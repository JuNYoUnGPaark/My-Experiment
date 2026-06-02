from __future__ import annotations

import os
import csv
import time
import math
import argparse
import psutil
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# Config
# ============================================================
WINDOW_SIZE = 128
PATCH_SIZE = 8
N_PATCHES = 16
NUM_CHANNELS = 9
NUM_CLASSES = 6
PHYS_DIM = 9

D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
D_FF = 128
DROPOUT = 0.1
GATE_HIDDEN = 64
NUM_FREQ_BANDS = 8


# ============================================================
# Model Definition
# ============================================================
class TimePatchEmbed(nn.Module):
    def __init__(self, num_channels, patch_size, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * num_channels, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        N = T // self.patch_size
        x = x[:, :N * self.patch_size, :]
        x = x.reshape(B, N, self.patch_size * C)
        return self.norm(self.proj(x))


class SpectralFilterbankEmbed(nn.Module):
    def __init__(self, num_channels, patch_size, num_bands, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.num_bins = patch_size // 2 + 1
        self.num_bands = num_bands

        self.filterbank = nn.Parameter(
            torch.randn(num_channels, self.num_bins, num_bands) * 0.02
        )
        self.proj = nn.Linear(num_channels * num_bands, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        N = T // self.patch_size
        x = x[:, :N * self.patch_size, :]
        x = x.reshape(B, N, self.patch_size, C)

        xf = torch.fft.rfft(x, dim=2)
        xf = torch.abs(xf)
        xf = xf.permute(0, 1, 3, 2)

        bands = torch.einsum("bncf,cfk->bnck", xf, self.filterbank)
        bands = bands.reshape(B, N, C * self.num_bands)

        return self.norm(self.proj(bands))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x1 = self.norm1(x)
        z, _ = self.attn(x1, x1, x1)
        x = x + z
        x = x + self.ffn(self.norm2(x))
        return x


class BranchEncoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class PatchPhysicsGate(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_phys_patch):
        return torch.sigmoid(self.mlp(z_phys_patch))


class PhysicsGuidedPatchDualTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_embed = TimePatchEmbed(NUM_CHANNELS, PATCH_SIZE, D_MODEL)
        self.freq_embed = SpectralFilterbankEmbed(
            NUM_CHANNELS, PATCH_SIZE, NUM_FREQ_BANDS, D_MODEL
        )

        self.time_pos = nn.Parameter(torch.zeros(1, N_PATCHES, D_MODEL))
        self.freq_pos = nn.Parameter(torch.zeros(1, N_PATCHES, D_MODEL))

        self.time_encoder = BranchEncoder(D_MODEL, NHEAD, D_FF, NUM_LAYERS, DROPOUT)
        self.freq_encoder = BranchEncoder(D_MODEL, NHEAD, D_FF, NUM_LAYERS, DROPOUT)

        self.gate = PatchPhysicsGate(PHYS_DIM, GATE_HIDDEN)

        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, NUM_CLASSES),
        )

    def forward(self, x, z_phys_patch):
        ht = self.time_embed(x) + self.time_pos
        hf = self.freq_embed(x) + self.freq_pos

        ht = self.time_encoder(ht)
        hf = self.freq_encoder(hf)

        alpha = self.gate(z_phys_patch)
        h_patch = alpha * ht + (1.0 - alpha) * hf
        h_pool = h_patch.mean(dim=1)

        logits = self.classifier(h_pool)
        return logits


# ============================================================
# Utils
# ============================================================
def get_state_dict(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]

    return ckpt


@torch.no_grad()
def evaluate_full_test(model, X, phys, y, device, batch_size=128):
    preds = []

    for start in range(0, len(X), batch_size):
        xb = np.asarray(X[start:start + batch_size], dtype=np.float32)
        pb = np.asarray(phys[start:start + batch_size], dtype=np.float32)

        xb = torch.from_numpy(xb).to(device)
        pb = torch.from_numpy(pb).to(device)

        logits = model(xb, pb)
        preds.append(logits.argmax(dim=1).cpu().numpy())

    y_pred = np.concatenate(preds)

    return {
        "Accuracy": float(accuracy_score(y, y_pred)),
        "Macro-F1": float(f1_score(y, y_pred, average="macro")),
    }


@torch.no_grad()
def benchmark_latency_ram(
    model,
    X,
    phys,
    indices,
    repeat_per_sample,
    warmup_samples,
    idle_rss,
    device,
):
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss

    warm_indices = indices[:min(warmup_samples, len(indices))]

    for idx in warm_indices:
        xb = torch.from_numpy(np.asarray(X[idx:idx + 1], dtype=np.float32)).to(device)
        pb = torch.from_numpy(np.asarray(phys[idx:idx + 1], dtype=np.float32)).to(device)
        _ = model(xb, pb)
        peak_rss = max(peak_rss, process.memory_info().rss)

    times = []

    for idx in indices:
        xb = torch.from_numpy(np.asarray(X[idx:idx + 1], dtype=np.float32)).to(device)
        pb = torch.from_numpy(np.asarray(phys[idx:idx + 1], dtype=np.float32)).to(device)

        for _ in range(repeat_per_sample):
            t0 = time.perf_counter()
            _ = model(xb, pb)
            t1 = time.perf_counter()

            times.append((t1 - t0) * 1000.0)
            peak_rss = max(peak_rss, process.memory_info().rss)

    return {
        "Latency_ms": float(np.mean(times)),
        "Latency_std_ms": float(np.std(times)),
        "Peak_RAM_MB": float((peak_rss - idle_rss) / (1024 ** 2)),
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="best_model.pt")
    parser.add_argument("--x_path", type=str, default="X_test.npy")
    parser.add_argument("--phys_path", type=str, default="phys_test.npy")
    parser.add_argument("--y_path", type=str, default="y_test.npy")
    parser.add_argument("--out_csv", type=str, default="rpi_physics_guided_results.csv")

    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--repeat_per_sample", type=int, default=5)
    parser.add_argument("--warmup_samples", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_prompt", action="store_true")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    device = torch.device("cpu")

    X = np.load(args.x_path, mmap_mode="r")
    phys = np.load(args.phys_path, mmap_mode="r")
    y = np.load(args.y_path, mmap_mode="r")

    model = PhysicsGuidedPatchDualTransformer()
    state = get_state_dict(args.model_path)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    process = psutil.Process(os.getpid())
    idle_rss = process.memory_info().rss

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(X), size=min(args.num_samples, len(X)), replace=False)

    print("\n================ RPi Physics-Guided HAR Benchmark ================")
    print(f"Model           : {args.model_path}")
    print(f"X_test          : {args.x_path} | shape={X.shape}")
    print(f"phys_test       : {args.phys_path} | shape={phys.shape}")
    print(f"y_test          : {args.y_path} | shape={y.shape}")
    print(f"Threads         : {torch.get_num_threads()}")
    print(f"Samples x repeat: {len(indices)} x {args.repeat_per_sample}")
    print(f"Idle RSS        : {idle_rss / (1024 ** 2):.4f} MB")
    print("===================================================================\n")

    if not args.no_prompt:
        input("Prepare power meter, then press Enter to start inference...")

    bench = benchmark_latency_ram(
        model=model,
        X=X,
        phys=phys,
        indices=indices,
        repeat_per_sample=args.repeat_per_sample,
        warmup_samples=args.warmup_samples,
        idle_rss=idle_rss,
        device=device,
    )

    if args.no_prompt:
        peak_power_w = ""
        energy_mj = ""
    else:
        text = input("Enter measured Peak Power(W), or press Enter to skip: ").strip()
        if text == "":
            peak_power_w = ""
            energy_mj = ""
        else:
            peak_power_w = float(text)
            energy_mj = peak_power_w * bench["Latency_ms"]

    metric = evaluate_full_test(
        model=model,
        X=X,
        phys=phys,
        y=np.asarray(y),
        device=device,
        batch_size=args.eval_batch_size,
    )

    params_m = count_parameters(model) / 1e6

    row = {
        "Dataset": "UCI-HAR",
        "Method": "PhysicsGuidedPatchDual",
        "Params(M)": params_m,
        "Latency(ms)": bench["Latency_ms"],
        "LatencyStd(ms)": bench["Latency_std_ms"],
        "Energy(mJ)": energy_mj,
        "PeakRAM(MB)": bench["Peak_RAM_MB"],
        "PeakPower(W)": peak_power_w,
        "Macro-F1": metric["Macro-F1"],
        "Accuracy": metric["Accuracy"],
        "Checkpoint": args.model_path,
    }

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print("\n================ Result ================")
    print(f"Params(M)      : {params_m:.4f}")
    print(f"Macro-F1       : {metric['Macro-F1']:.4f}")
    print(f"Accuracy       : {metric['Accuracy']:.4f}")
    print(f"Latency(ms)    : {bench['Latency_ms']:.4f}")
    print(f"Latency Std(ms): {bench['Latency_std_ms']:.4f}")
    print(f"Peak RAM(MB)   : {bench['Peak_RAM_MB']:.4f}")
    print(f"Peak Power(W)  : {peak_power_w}")
    print(f"Energy(mJ)     : {energy_mj if energy_mj == '' else f'{energy_mj:.4f}'}")
    print(f"Saved CSV      : {args.out_csv}")


if __name__ == "__main__":
    main()
