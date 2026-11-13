"""Measure cached full-field inference with parallel heads and CUDA graphs."""

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@torch.no_grad()
def make_augmented_basis(means, modes):
    """Return contiguous B[sum_m(1+K_m), D], blocks [mean_m; modes_m.T]."""
    if not means or len(means) != len(modes):
        raise ValueError("Expected equal, nonempty sequences of means and modes.")
    first = means[0]
    if first.ndim != 1 or not first.is_floating_point():
        raise ValueError("Each mean must be a floating vector of output points.")
    blocks = []
    for mean, phi in zip(means, modes):
        if mean.shape != first.shape or phi.ndim != 2 or phi.shape[0] != len(first):
            raise ValueError("Mean and modes must share the complete output grid.")
        if mean.device != first.device or phi.device != first.device:
            raise ValueError("Basis tensors must already reside on one device.")
        if mean.dtype != first.dtype or phi.dtype != first.dtype:
            raise ValueError("Basis tensors must share dtype; no implicit precision changes.")
        blocks.extend((mean[None], phi.T))
    return torch.cat(blocks, dim=0).contiguous().detach()


@torch.no_grad()
def predict_mixture(model, x, k, B):
    """TEMPO's unchanged gate and branches, reconstructed by one full-grid matmul."""
    w = model.gating(x, k)
    coefficients = []
    for m, branch in enumerate(model.branches):
        wm = w[:, m : m + 1]
        coefficients.extend((wm, wm * branch(x, k)))
    return torch.cat(coefficients, dim=1) @ B


@torch.no_grad()
def predict_single(branch, x, k, B):
    """Standalone NeuralPOD/POD: [1, branch(x,k)] @ [mean; modes.T]."""
    coefficients = branch(x, k)
    return torch.cat((coefficients.new_ones((len(x), 1)), coefficients), dim=1) @ B


@torch.no_grad()
def cached_fused_field(model, x, k, prepackedbasis):
    """Reconstruct a TEMPO or single-branch field with a cached basis."""
    if hasattr(model, "gating") and hasattr(model, "branches"):
        return predict_mixture(model, x, k, prepackedbasis)
    return predict_single(getattr(model, "branch", model), x, k, prepackedbasis)


@dataclass(frozen=True)
class PackedHeads:
    heads: tuple
    d_kappa: int
    width: int
    ranks: tuple
    first_weight: torch.Tensor
    first_bias: torch.Tensor
    hidden_weights: tuple
    hidden_biases: tuple
    last_weight: torch.Tensor
    last_bias: torch.Tensor


@torch.no_grad()
def build_parallel_mlp_heads(heads, *, x_mean=0.0, x_scale=1.0, p_mean=0.0, p_scale=1.0):
    """Pack branch MLPs and fold their input normalization into the first layer.

    Normalization values are scalars or per-feature vectors; queries use raw x/k."""
    heads = tuple(heads)
    if not heads:
        raise ValueError("Require at least one BranchNet.")
    nets = [head.net for head in heads]
    for head, net in zip(heads, nets):
        if (
            head.training
            or not isinstance(net, nn.Sequential)
            or (len(net) != len(nets[0]) or len(net) < 3 or len(net) % 2 != 1)
        ):
            raise ValueError(
                "Expected eval-mode Sequential with uniform hidden layers and an output Linear."
            )
        if any(
            not isinstance(net[i], nn.Linear if i % 2 == 0 else nn.Tanh) for i in range(len(net))
        ):
            raise TypeError("Only the original alternating Linear/Tanh architecture is supported.")
    first = nets[0][0]
    width, inputs, d_kappa = first.out_features, first.in_features, heads[0].d_kappa
    for head, net in zip(heads, nets):
        if head.d_kappa != d_kappa or (net[0].in_features, net[0].out_features) != (inputs, width):
            raise ValueError("Heads must share input layout and hidden width.")
        for i in range(0, len(net), 2):
            layer = net[i]
            if (
                layer.bias is None
                or layer.weight.device != first.weight.device
                or layer.weight.dtype != first.weight.dtype
            ):
                raise ValueError("All Linear weights must have bias and share dtype/device.")
            if 0 < i < len(net) - 1 and (layer.in_features, layer.out_features) != (width, width):
                raise ValueError("Hidden layers must have uniform width.")
        if net[-1].in_features != width:
            raise ValueError("Output layer must consume the shared hidden width.")
    ranks = tuple(net[-1].out_features for net in nets)
    if not 0 <= d_kappa <= inputs:
        raise ValueError("Invalid parameter dimension.")

    def block(value, n):
        return torch.broadcast_to(
            torch.as_tensor(value, device=first.weight.device, dtype=first.weight.dtype), (n,)
        )

    mean = torch.cat((block(x_mean, inputs - d_kappa), block(p_mean, d_kappa)))
    scale = torch.cat((block(x_scale, inputs - d_kappa), block(p_scale, d_kappa)))
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Normalization requires finite means and finite positive scales.")
    first_weight = torch.cat([net[0].weight for net in nets], dim=0).detach() / scale[None, :]
    first_bias = torch.cat([net[0].bias for net in nets], dim=0).detach() - (
        first_weight * mean[None, :]
    ).sum(dim=1)
    last_weight = first.weight.new_zeros((len(nets), width, max(ranks)))
    last_bias = first.bias.new_zeros((len(nets), 1, max(ranks)))
    for h, (net, rank) in enumerate(zip(nets, ranks)):
        last_weight[h, :, :rank] = net[-1].weight.T
        last_bias[h, 0, :rank] = net[-1].bias
    return PackedHeads(
        heads,
        d_kappa,
        width,
        ranks,
        first_weight,
        first_bias,
        tuple(
            torch.stack([net[i].weight for net in nets]).transpose(1, 2).contiguous().detach()
            for i in range(2, len(nets[0]) - 1, 2)
        ),
        tuple(
            torch.stack([net[i].bias for net in nets])[:, None, :].detach()
            for i in range(2, len(nets[0]) - 1, 2)
        ),
        last_weight.detach(),
        last_bias.detach(),
    )


@torch.no_grad()
def parallel_head_outputs(packed, x, k):
    """Evaluate each packed head before gating and field reconstruction."""
    if any(head.training for head in packed.heads):
        raise ValueError("Packed heads require eval mode.")
    inp = torch.cat((x, k), dim=-1) if packed.d_kappa > 0 else x
    h = torch.tanh(F.linear(inp, packed.first_weight, packed.first_bias))
    h = h.reshape(len(x), len(packed.heads), packed.width).transpose(0, 1)
    for weight, bias in zip(packed.hidden_weights, packed.hidden_biases):
        h = torch.tanh(torch.bmm(h, weight) + bias)
    values = torch.bmm(h, packed.last_weight) + packed.last_bias
    return tuple(values[i, :, :rank] for i, rank in enumerate(packed.ranks))


@torch.no_grad()
def predict_parallel_mixture(model, x, k, B, packed):
    """Reconstruct the full soft-mixture field with the augmented basis."""
    expected = (model.gating.branch, *model.branches)
    if len(expected) != len(packed.heads) or any(
        a is not b for a, b in zip(expected, packed.heads)
    ):
        raise ValueError("Packed heads must belong to this unchanged model.")
    outputs = parallel_head_outputs(packed, x, k)
    w = F.softmax(outputs[0], dim=-1)
    coefficients = []
    for m, a in enumerate(outputs[1:]):
        wm = w[:, m : m + 1]
        coefficients.extend((wm, wm * a))
    return torch.cat(coefficients, dim=1) @ B


@torch.no_grad()
def predict_parallel_single(branch, x, k, B, packed):
    """Reconstruct one branch with its packed MLP and augmented basis."""
    if len(packed.heads) != 1 or packed.heads[0] is not branch:
        raise ValueError("Require a pack of this standalone branch.")
    a = parallel_head_outputs(packed, x, k)[0]
    return torch.cat((a.new_ones((len(x), 1)), a), dim=1) @ B


def supported_gpu_fields(help_text):
    fields = ["clocks.sm"] if "clocks.sm" in help_text else []
    for kind in ("hw", "sw"):
        supported = next(
            (
                f
                for prefix in ("clocks_event_reasons", "clocks_throttle_reasons")
                if (f := prefix + "." + kind + "_thermal_slowdown") in help_text
            ),
            None,
        )
        if supported:
            fields.append(supported)
    return fields


@lru_cache(maxsize=1)
def optional_gpu_fields():
    return supported_gpu_fields(
        subprocess.check_output(["nvidia-smi", "--help-query-gpu"], text=True)
    )


def normalize_thermal_status(value):
    token = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if token in ("active", "yes", "true", "1"):
        return "Active"
    if token in ("not active", "inactive", "no", "false", "0"):
        return "Inactive"
    return "Unknown"


def merge_thermal_usable(flags):
    if any(flag is False for flag in flags):
        return False
    if flags and all(flag is True for flag in flags):
        return True
    return None


def thermal_status(fields):
    statuses = {
        kind: normalize_thermal_status(
            next(
                (
                    fields[name]
                    for name in fields
                    if name.endswith("." + kind + "_thermal_slowdown")
                ),
                None,
            )
        )
        for kind in ("hw", "sw")
    }
    usable = merge_thermal_usable(
        [
            False if value == "Active" else True if value == "Inactive" else None
            for value in statuses.values()
        ]
    )
    return dict(
        hw=statuses["hw"],
        sw=statuses["sw"],
        usable=usable,
        aggregate="Active" if usable is False else "Inactive" if usable is True else "Unknown",
    )


def gpu_state(uuid):
    fields = (
        "uuid,name,driver_version,memory.total,memory.free,utilization.gpu,temperature.gpu,pstate"
    )
    fields += "".join("," + f for f in optional_gpu_fields())
    values = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + uuid,
                "--query-gpu=" + fields,
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .split(",")
    )
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    pids = [
        int(line.split(",")[1])
        for line in processes.splitlines()
        if line.split(",")[0].strip() == uuid
    ]
    if set(pids) - {os.getpid()}:
        raise RuntimeError("Timing rejected: selected GPU has another compute process.")
    result = dict(zip(fields.split(","), [v.strip() for v in values]))
    result.setdefault("clocks.sm", None)
    thermal = thermal_status(result)
    result["thermal_status"] = thermal
    result["thermal_slowdown"] = thermal["aggregate"]
    return dict(gpu=result, compute_pids=pids)


def statistics(rows):
    values = np.asarray(rows, dtype="f8")
    assert values.ndim == 2 and np.isfinite(values).all() and (values >= 0).all()
    q = np.quantile(values, [0.25, 0.5, 0.75])
    return dict(
        sample_count=int(values.size),
        repeats=len(values),
        queries_per_repeat=values.shape[1],
        mean_ms=float(values.mean()),
        median_ms=float(q[1]),
        q25_ms=float(q[0]),
        q75_ms=float(q[2]),
        iqr_ms=float(q[2] - q[0]),
        repeat_medians_ms=np.median(values, axis=1).tolist(),
        raw_ms=values.tolist(),
    )


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(p, d):
    with p.open("x") as f:
        json.dump(d, f, indent=2, allow_nan=False)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--case", choices=["burgers", "helmholtz"], required=True)
    a.add_argument("--checkpoint", type=Path, required=True)
    a.add_argument("--queries", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a = a.parse_args()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    os.environ["MAX_JOBS"] = "2"
    cache = Path(tempfile.gettempdir()) / "tempo-timing"
    for key, name in [
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCH_EXTENSIONS_DIR", "extensions"),
    ]:
        path = cache / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not uuid.startswith("GPU-") or "," in uuid:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain one GPU UUID.")
    before = gpu_state(uuid)
    assert not before["compute_pids"] and float(before["gpu"]["utilization.gpu"]) <= 5
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    _, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(0.30, 7e9 / total))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from scripts.paper.run_comparison import load_fit

    pins = {str(p): sha(p) for p in [ROOT / "scripts/paper/run_comparison.py", Path(__file__)]}
    if a.case == "burgers":
        checkpoint = a.checkpoint
        expected = "f453f93c973856c9cff5a3057fd2f7f0c5d4202156abed0031e9901e346ffbce"
        data = a.queries
        datahash = "b31de08f8966436863065f6467638ca673d53f3ce07030abd85b3f2e6e2cd8ee"
    else:
        checkpoint = a.checkpoint
        expected = "079400dcb94a6d527f3eb7536955d5dc4078ace71a38df1f725561e419f4027f"
        data = a.queries
        datahash = "7b173bc42617c42827d91931a17e06eec0e87d877ffd1de546d3da027d7d8969"
    assert sha(checkpoint) == expected and sha(data) == datahash
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    fit = load_fit(checkpoint, torch.device("cuda"))
    model = fit.model.eval()
    scale = ck["scale"]
    actual = model.state_dict()
    assert set(actual) == set(ck["online"])
    assert all(torch.equal(v.cpu(), ck["online"][k]) for k, v in actual.items())
    with np.load(data, allow_pickle=False) as z:
        if a.case == "burgers":
            x = torch.from_numpy(z["x"].copy()).cuda()
            k = torch.from_numpy(z["kappa"].copy()).cuda()
            coords = torch.from_numpy(z["coords"].copy())
            meta = json.loads(str(z["metadata_json"]))
            ids = list(range(64))
        else:
            assert not any(n.startswith(("test_", "heldout_")) for n in z.files)
            x = torch.from_numpy(z["val_x"][:64].copy()).float().cuda()
            k = torch.from_numpy(z["val_kappa"][:64].copy()).float().cuda()
            coords = torch.from_numpy(z["coords"].copy()).float()
            meta = json.loads(str(z["metadata"].item()))
            ids = z["val_id"][:64].astype(str).tolist()
            assert scale == meta["scales"]
    assert len(x) == len(k) == 64 and torch.equal(coords, ck["coords"])
    with torch.inference_mode():
        means, modes = fit.evaluate_bases(coords.cuda())
        if a.case == "burgers":
            norm = {q: scale[q] for q in ["x_mean", "x_scale", "p_mean", "p_scale"]}
            gain = scale["y_scale"]

            def standard(i):
                return (
                    model(
                        (x[i : i + 1] - scale["x_mean"]) / scale["x_scale"],
                        (k[i : i + 1] - scale["p_mean"]) / scale["p_scale"],
                        means,
                        modes,
                    )[0]
                    * gain
                )
        else:
            norm = {}
            gain = 1.0

            def standard(i):
                return model(x[i : i + 1], k[i : i + 1], means, modes)[0].reshape(-1, 101, 101)

        packed = build_parallel_mlp_heads([model.gating.branch, *model.branches], **norm)
        basis = make_augmented_basis(means, modes) * gain

        def eager_inputs(xb, kb):
            y = predict_parallel_mixture(model, xb, kb, basis, packed)
            return y if a.case == "burgers" else y.reshape(-1, 101, 101)

        def eager(i):
            return eager_inputs(x[i : i + 1], k[i : i + 1])

        sx = x[:1].clone()
        sk = k[:1].clone()
        compile_ms = 0.0
        if a.case == "burgers":
            t = time.perf_counter()
            forward = torch.compile(
                eager_inputs, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
            )
            probe = forward(sx, sk)
            torch.cuda.synchronize()
            compile_ms = (time.perf_counter() - t) * 1e3
            torch.testing.assert_close(probe, standard(0), rtol=2e-5, atol=2e-6)
        else:
            forward = eager_inputs
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                _warm = forward(sx, sk)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sy = forward(sx, sk)
        torch.cuda.synchronize()

        def replay(i):
            sx.copy_(x[i : i + 1])
            sk.copy_(k[i : i + 1])
            graph.replay()
            return sy.clone()

        methods = (
            {"parallel_eager": eager, "graph_compiled": replay}
            if a.case == "burgers"
            else {"standard": standard, "cuda_graph_clone": replay}
        )
        parity = {n: [] for n in methods}
        retained = replay(0)
        snapshot = retained.clone()
        hashes = []
        for i in range(64):
            ref = standard(i)
            assert ref.dtype == torch.float32 and torch.isfinite(ref).all()
            for n, fn in methods.items():
                y = fn(i)
                torch.testing.assert_close(y, ref, rtol=2e-5, atol=2e-6)
                parity[n].append(
                    {
                        "row": i,
                        "max_abs": float((y - ref).abs().max()),
                        "relative_l2": float(
                            (y - ref).double().norm() / ref.double().norm().clamp_min(1e-12)
                        ),
                    }
                )
            y = replay(i)
            assert y.data_ptr() != sy.data_ptr() and y.data_ptr() != retained.data_ptr()
            hashes.append(hashlib.sha256(y.cpu().numpy().tobytes()).hexdigest())
        torch.testing.assert_close(retained, snapshot, rtol=0, atol=0)
        assert len(set(hashes)) == 64
        assert not torch.equal(retained, replay(1))
        for fn in methods.values():
            for i in range(100):
                fn(i % 64)
        torch.cuda.synchronize()
        raw = {n: [] for n in methods}
        telemetry = []
        indices = []
        for rep in range(5):
            for n in list(methods) if rep % 2 == 0 else list(methods)[::-1]:
                first = gpu_state(uuid)
                values = []
                used = []
                for j in range(200):
                    i = (rep * 200 + j) % 64
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    y = methods[n](i)
                    torch.cuda.synchronize()
                    values.append((time.perf_counter_ns() - start) / 1e6)
                    used.append(i)
                    del y
                last = gpu_state(uuid)
                raw[n].append(values)
                indices.append({"repeat": rep, "method": n, "rows": used})
                telemetry.append(
                    {
                        "before": first,
                        "after": last,
                        "usable": merge_thermal_usable(
                            [
                                first["gpu"]["thermal_status"]["usable"],
                                last["gpu"]["thermal_status"]["usable"],
                            ]
                        ),
                    }
                )
    assert (
        sha(checkpoint) == expected
        and sha(data) == datahash
        and all(sha(p) == h for p, h in pins.items())
    )
    timing = {n: statistics(v) for n, v in raw.items()}
    report = {
        "status": "complete",
        "case": a.case,
        "training": False,
        "test_arrays_read": False,
        "checkpoint_sha256": expected,
        "checkpoint": str(checkpoint),
        "data_sha256": datahash,
        "data": str(data),
        "source_sha256": pins,
        "driver_sha256": sha(__file__),
        "online_state_exact": True,
        "grid": [201, 1024] if a.case == "burgers" else [1, 101, 101],
        "output_points": int(np.prod(coords.shape[:1])),
        "ranks": ck["fit_log"]["ranks"],
        "sample_ids": ids,
        "query_metadata": meta,
        "sample_input_indices": indices,
        "timings": timing,
        "all64_parity": parity,
        "fresh_output": {
            "all64_unique": True,
            "independent_clone": True,
            "retained_output_unchanged": True,
        },
        "gpu_before": before,
        "gpu_after": gpu_state(uuid),
        "telemetry": telemetry,
        "all_thermal_usable": merge_thermal_usable([t["usable"] for t in telemetry]),
        "compile_ms_excluded": compile_ms,
        "protocol": {
            "warmup_per_method": 100,
            "repeats": 5,
            "calls_per_repeat": 200,
            "queries": 64,
            "batch_size": 1,
            "dtype": "float32",
            "TF32": False,
            "clock": "perf_counter_ns with CUDA synchronization before and after every call",
            "included": "raw GPU input normalization and physical full-field output"
            if a.case == "burgers"
            else "prepared GPU input through full native RMS-normalized field",
            "excluded": "loading, host transfers, basis caching, packing, compilation, graph capture"
            + ("" if a.case == "burgers" else ", input normalization and physical output inverse"),
            "graph_included": "GPU input copy_, replay, independent output clone",
            "all_soft_experts_active": True,
        },
        "versions": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
        },
    }
    write(a.output, report)
    print(
        json.dumps(
            {
                "case": a.case,
                "checkpoint": expected,
                "median_ms": {n: v["median_ms"] for n, v in timing.items()},
                "thermal": report["all_thermal_usable"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
