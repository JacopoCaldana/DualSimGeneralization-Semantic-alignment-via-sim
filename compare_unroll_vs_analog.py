"""
Deliverable: NMSE-vs-iteration/layer comparison between
  (a) the classic iterative Algorithm 2  -> dualsim.optimize_alternating  (ANALOG baseline)
  (b) the deep-unrolled SCA               -> sca_unrolled.DualSIMUnrolledSCA (UNROLLING)

Both act on the SAME fixed semantic operator A and the SAME SIM physics; the analog
baseline is averaged over a test set of channels H, and the unrolled net (trained on
random H with that fixed A) is evaluated on the same test channels. Saves a plot
(PDF+PNG) and a CSV (layer, nmse_analog, nmse_unroll) in results_sca/.

Usage (defaults give the fast "medium" 64-dim config that fits in <6 GB):
    python compare_unroll_vs_analog.py
    python compare_unroll_vs_analog.py --in-xy 16 12 --out-xy 24 16 -M 16 -L 5   # paper-like (needs lots of VRAM)
"""
import argparse
import contextlib
import os
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dualsim import DualSIMoptimizer, DualSIMoptimizerTorch
from sca_unrolled import train_sca_unrolled

OUTDIR = "results_sca"


# --------------------------------------------------------------------------- #
def build_sim(in_xy, out_xy, M, L, wavelength, slayer, device):
    ix, iy = in_xy
    ox, oy = out_xy
    cpu = DualSIMoptimizer(
        num_layers_TX=L, num_meta_atoms_TX_in_x=ix, num_meta_atoms_TX_in_y=iy,
        num_meta_atoms_TX_out_x=ox, num_meta_atoms_TX_out_y=oy,
        num_meta_atoms_TX_int_x=M, num_meta_atoms_TX_int_y=M, thickness_TX=slayer * L,
        num_layers_RX=L, num_meta_atoms_RX_in_x=ox, num_meta_atoms_RX_in_y=oy,
        num_meta_atoms_RX_out_x=ox, num_meta_atoms_RX_out_y=oy,
        num_meta_atoms_RX_int_x=M, num_meta_atoms_RX_int_y=M, thickness_RX=slayer * L,
        wavelength=wavelength, spacings=None, verbose=False,
    )
    return DualSIMoptimizerTorch(cpu).to(device)


def sim_dims(sim):
    theta = sim.get_buffer("W_T_0").shape[1]
    N_T = sim.get_buffer(f"W_T_{sim.L_T - 1}").shape[0]
    N_R = sim.get_buffer("W_R_0").shape[1]
    omega = sim.get_buffer(f"W_R_{sim.L_R - 1}").shape[0]
    return theta, N_T, N_R, omega


def make_A(theta, omega, pilots, gamma, seed, device):
    """Fixed supervised linear aligner A (paper eq. 10) from seeded random pilots."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(theta, pilots, generator=g) + 1j * torch.randn(theta, pilots, generator=g)
    Y = torch.randn(omega, pilots, generator=g) + 1j * torch.randn(omega, pilots, generator=g)
    inv = torch.linalg.inv(X @ X.conj().T + gamma * torch.eye(theta, dtype=torch.complex64))
    A = Y @ X.conj().T @ inv
    return A.to(torch.complex64).to(device)


def make_test_H(n, N_R, N_T, seed, device):
    g = torch.Generator().manual_seed(seed)
    H = (torch.randn(n, N_R, N_T, generator=g) + 1j * torch.randn(n, N_R, N_T, generator=g)) / (2 ** 0.5)
    return H.to(torch.complex64).to(device)


@torch.no_grad()
def _zero_phases(sim):
    for p in sim.xi_T:
        p.zero_()
    for p in sim.xi_R:
        p.zero_()


@torch.no_grad()
def _nmse_now(sim, A, H, A2):
    Z, _ = sim.get_effective_cascade(H)
    num = (Z.conj() * A).sum()
    den = (Z.conj() * Z).sum().real + 1e-12
    beta = num / den
    return ((beta * Z - A).abs() ** 2).sum().item() / A2


def analog_curve(sim, A, H_test, K, lr, warmup_frac):
    """Run optimize_alternating per test channel (reset to zero phases); average NMSE/layer."""
    A2 = (torch.norm(A) ** 2).item()
    curves = []
    devnull = open(os.devnull, "w")
    for i in range(H_test.shape[0]):
        H = H_test[i]
        _zero_phases(sim)
        nmse0 = _nmse_now(sim, A, H, A2)  # xi^0 (zeros)
        with contextlib.redirect_stdout(devnull):
            loss_hist = sim.optimize_alternating(
                A, H, max_iters=K, lr=lr, warmup_frac=warmup_frac,
                clip_value=1.0, rx_steps_per_tx=1,
            )
        curves.append(np.concatenate([[nmse0], np.asarray(loss_hist) / A2]))  # len K+1
    devnull.close()
    _zero_phases(sim)
    return np.mean(np.stack(curves, axis=0), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-xy", type=int, nargs=2, default=[8, 8], help="TX input meta-atoms (x y)")
    ap.add_argument("--out-xy", type=int, nargs=2, default=[8, 8], help="antenna / RX-out meta-atoms (x y)")
    ap.add_argument("-M", type=int, default=8, help="intermediate-layer resolution (M x M atoms)")
    ap.add_argument("-L", type=int, default=5, help="SIM layers per side")
    ap.add_argument("-K", type=int, default=100, help="iterations / unrolled layers")
    ap.add_argument("--wavelength", type=float, default=0.005)
    ap.add_argument("--slayer", type=float, default=0.025)
    # unrolling training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--iters-per-epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-w", type=float, default=1e-2)
    ap.add_argument("--init-w", type=float, default=0.1)
    ap.add_argument("--coupling", choices=["scalar", "diagonal"], default="diagonal")
    ap.add_argument("--innovation-norm", choices=["rms", "sign", "unit", "none"], default="rms")
    ap.add_argument("--supervision", choices=["final", "all"], default="all")
    ap.add_argument("--momentum", action=argparse.BooleanOptionalAction, default=True,
                    help="Adam-style learnable momentum on the innovation (--no-momentum to disable)")
    ap.add_argument("--init-mu", type=float, default=0.9)
    ap.add_argument("--learn-s", action=argparse.BooleanOptionalAction, default=True,
                    help="learn S^k (xi^{k+1}=S^k xi^k + W^k g^k); init=I, lr_S=lr_W/ratio, ||S-I||^2 reg")
    ap.add_argument("--s-lr-ratio", type=float, default=50.0)
    ap.add_argument("--reg-s", type=float, default=1e-2)
    ap.add_argument("--first-order", action="store_true",
                    help="detached-innovation training (no double-backprop); needed for large M/L")
    ap.add_argument("--analytic", action="store_true",
                    help="analytic innovation + gradient checkpointing -> deep-sup 2nd-order at any scale")
    # analog baseline
    ap.add_argument("--analog-lr", type=float, default=0.1)
    ap.add_argument("--analog-warmup", type=float, default=0.05)
    ap.add_argument("--n-test", type=int, default=32, help="test channels to average over")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"device={device} | config: in={args.in_xy} out={args.out_xy} M={args.M} "
          f"L={args.L} K={args.K}")
    sim = build_sim(args.in_xy, args.out_xy, args.M, args.L, args.wavelength, args.slayer, device)
    theta, N_T, N_R, omega = sim_dims(sim)
    n_T = sum(p.shape[0] for p in sim.xi_T)
    n_R = sum(p.shape[0] for p in sim.xi_R)
    print(f"  dims: theta={theta} N_T={N_T} N_R={N_R} omega={omega} | n_T={n_T} n_R={n_R}")

    A = make_A(theta, omega, pilots=max(2 * theta, 128), gamma=1e-4, seed=args.seed, device=device)
    H_test = make_test_H(args.n_test, N_R, N_T, seed=args.seed + 1, device=device)
    print(f"  A: {tuple(A.shape)} ||A||_F={torch.norm(A).item():.3f} | "
          f"H_test: {tuple(H_test.shape)}")

    # ---- ANALOG ----
    print("\n[1/2] Analog baseline (optimize_alternating, rx_steps_per_tx=1)...")
    t = time.time()
    c_analog = analog_curve(sim, A, H_test, args.K, args.analog_lr, args.analog_warmup)
    print(f"  done in {time.time() - t:.1f}s | NMSE: init={c_analog[0]:.4f} -> final={c_analog[-1]:.4f}")

    # ---- UNROLLING ----
    print("\n[2/2] Deep unrolling (training the per-layer step sizes)...")
    t = time.time()
    model, _ = train_sca_unrolled(
        sim, A, args.K, N_T=N_T, N_R=N_R,
        epochs=args.epochs, iters_per_epoch=args.iters_per_epoch, batch=args.batch,
        lr_w=args.lr_w, init_w=args.init_w, coupling=args.coupling,
        innovation_norm=args.innovation_norm, supervision=args.supervision,
        momentum=args.momentum, init_mu=args.init_mu,
        learn_S=args.learn_s, s_lr_ratio=args.s_lr_ratio, reg_S=args.reg_s,
        first_order=args.first_order, analytic=args.analytic, checkpoint=args.analytic,
        device=device, seed=args.seed, val_H=H_test,
    )
    c_unroll = np.asarray(model.eval_curve(H_test, A))
    print(f"  done in {time.time() - t:.1f}s | NMSE: init={c_unroll[0]:.4f} -> final={c_unroll[-1]:.4f}")

    # ---- report + save ----
    layers = np.arange(len(c_analog))
    tag = args.tag or f"M{args.M}_L{args.L}_K{args.K}_{theta}x{omega}"
    print("\n=== NMSE vs iteration/layer ===")
    print(f"{'k':>5} {'analog':>10} {'unroll':>10} {'unroll<=analog':>16}")
    for k in [1, 5, 10, 25, 50, args.K]:
        if k < len(layers):
            mark = "OK" if c_unroll[k] <= c_analog[k] + 1e-9 else "x"
            print(f"{k:>5} {c_analog[k]:>10.4f} {c_unroll[k]:>10.4f} {mark:>16}")
    frac_below = float(np.mean(c_unroll <= c_analog + 1e-9))
    print(f"unrolling <= analog on {100 * frac_below:.0f}% of layers | "
          f"final gap (analog-unroll) = {c_analog[-1] - c_unroll[-1]:+.4f}")

    csv_path = os.path.join(OUTDIR, f"nmse_vs_layer_{tag}.csv")
    np.savetxt(csv_path, np.column_stack([layers, c_analog, c_unroll]),
               delimiter=",", header="layer,nmse_analog,nmse_unroll", comments="")
    print(f"\nsaved CSV -> {csv_path}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(layers, c_analog, "-", color="tab:red", lw=2, label="Analog (Algorithm 2)")
    ax.plot(layers, c_unroll, "-", color="tab:blue", lw=2, label="Deep unrolling (SCA)")
    ax.set_xlabel("iteration / unrolled layer  k")
    ax.set_ylabel("NMSE")
    ax.set_title(f"NMSE vs layer  ({tag})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(OUTDIR, f"nmse_vs_layer_{tag}.{ext}")
        fig.savefig(p, dpi=150)
        print(f"saved plot -> {p}")


if __name__ == "__main__":
    main()
