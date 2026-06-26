"""
Downstream-accuracy sweep for the DEEP-UNROLLED dual-SIM (clean-room SCA stack).

This is the unrolling counterpart of the classic dual-SIM accuracy experiment
`experiment_runner.run_experiment_layers` / `Dual_classification_multiple.py`:
instead of running the iterative Algorithm 2 (`optimize_alternating`, 3000 iters)
to configure the SIM, we run a SINGLE forward pass of the K-layer unrolled SCA
network (`sca_unrolled.DualSIMUnrolledSCA`) to read the TX/RX phases, then evaluate
the exact same over-the-air CIFAR-10 classification pipeline (`inference.run_evaluation`).

For every (M=32, L in {2,5,10,15}) we:
  1. build the SAME SIM physics as the classic dual-SIM experiment (TX in 16x12 -> out 24x16,
     RX in 24x16 -> out 24x16, M intermediate atoms/side, spacing lambda/2),
  2. train the per-layer step schedule on the FIXED real A_target with random channels,
  3. single-shot infer the phases on the real (A_target, H_mimo),
  4. write those phases into the sim and call `run_evaluation` verbatim -> accuracy.

The real semantic operator A_target (PPFE zero-shot or Linear ridge) and the real
Rayleigh channel H_mimo are built exactly as in `Dual_classification_multiple.py`,
so the unrolled numbers are directly comparable to the classic dual-SIM rows already saved in
`final_results_PPFE.csv` (overlaid in the plot).

Run:
    source venv/bin/activate
    python accuracy_unrolled.py                 # M=32, L=2,5,10,15, PPFE, seed 42
    python accuracy_unrolled.py -M 32 -L 2 5 10 15 -K 100 --with-classic --classic-iters 100
                                                 # equal-iteration: unroll K vs classic dual-SIM @ N it
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import argparse
import csv
import gc
import random
import time
import contextlib
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dualsim import DualSIMoptimizer, DualSIMoptimizerTorch
from sca_unrolled import train_sca_unrolled, DualSIMUnrolledSCA
from inference import run_evaluation
from oracle_test import run_oracle_test

from alignment import DataModuleAlignmentClassification
from classifier import DataModuleClassifier
from alignment_utils import ridge_regression, ppfe
from utils import complex_compressed_tensor, prewhiten, complex_gaussian_matrix
from models_tasks.classification import Classifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models"
OUTDIR = BASE_DIR / "results_accuracy_unrolled"

TX_NAME = "vit_small_patch16_224"
RX_NAME = "vit_base_patch16_224"

# SIM geometry (identical to experiment_runner.run_sim_configuration)
WAVELENGTH = 0.005
SLAYER = 5 * WAVELENGTH          # 0.025
DX = WAVELENGTH / 2              # 0.0025  (== default 0.5*lambda spacing)


# --------------------------------------------------------------------------- #
def build_sim(L, M_int, device):
    """Dual-SIM physics matching the classic dual-SIM accuracy experiment exactly."""
    sim_cpu = DualSIMoptimizer(
        num_layers_TX=L,
        num_meta_atoms_TX_in_x=16, num_meta_atoms_TX_in_y=12,
        num_meta_atoms_TX_out_x=24, num_meta_atoms_TX_out_y=16,
        num_meta_atoms_TX_int_x=M_int, num_meta_atoms_TX_int_y=M_int,
        thickness_TX=SLAYER * L,
        num_layers_RX=L,
        num_meta_atoms_RX_in_x=24, num_meta_atoms_RX_in_y=16,
        num_meta_atoms_RX_out_x=24, num_meta_atoms_RX_out_y=16,
        num_meta_atoms_RX_int_x=M_int, num_meta_atoms_RX_int_y=M_int,
        thickness_RX=SLAYER * L,
        wavelength=WAVELENGTH,
        spacings={'tx_in': DX, 'tx_out': DX, 'tx_int': DX,
                  'rx_in': DX, 'rx_out': DX, 'rx_int': DX},
        verbose=False,
    )
    return DualSIMoptimizerTorch(sim_cpu).to(device)


@torch.no_grad()
def set_sim_phases(sim, xT, xR):
    """Write the unrolled net's inferred per-layer phases (lists of [1, m_l]) into the
    sim's xi_T / xi_R parameters, so run_evaluation reads them via get_effective_cascade."""
    for p, x in zip(sim.xi_T, xT):
        p.copy_(x.reshape(-1).to(p.dtype))
    for p, x in zip(sim.xi_R, xR):
        p.copy_(x.reshape(-1).to(p.dtype))


@torch.no_grad()
def beta_and_nmse(sim, A, H):
    """Closed-form beta (eq. 42) and emulation NMSE for the current sim phases on (A, H)."""
    Z, _ = sim.get_effective_cascade(H)
    num = torch.sum(Z.conj() * A)
    den = torch.sum(Z.conj() * Z).real + 1e-12
    beta = num / den
    nmse = ((beta * Z - A).abs() ** 2).sum().item() / (torch.norm(A) ** 2).item()
    return beta, nmse


# --------------------------------------------------------------------------- #
def load_classic_converged_accuracy(strategy, M_int, seed):
    """Classic dual-SIM 'Accuracy SIM Mimo' per L at CONVERGENCE (3000 iters) -- reference only.

    Prefers the authoritative per-seed sweep results_layers_<strategy>_seed<seed>.json
    ({"32x32": {"2": 46.2, ...}}); falls back to the seed-averaged final_results CSV."""
    import json
    jpath = BASE_DIR / f"results_layers_{strategy}_seed{seed}.json"
    if jpath.exists():
        d = json.load(open(jpath)).get(f"{M_int}x{M_int}", {})
        out = {int(k): float(v) for k, v in d.items()}
        if out:
            return out
    path = BASE_DIR / f"final_results_{strategy}.csv"
    if not path.exists():
        return {}
    by_L = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("SIM Meta Atoms Intermediate X") != str(M_int):
                continue
            try:
                L = int(r["SIM Layers"]); acc = float(r["Accuracy SIM Mimo"])
            except (KeyError, ValueError, TypeError):
                continue
            by_L.setdefault(L, []).append(acc)
    return {L: float(np.mean(v)) for L, v in by_L.items()}


def classic_accuracy_inline(sim, A_target, H_mimo, dm_task, clf,
                            L_in, mu_in, L_out, mu_out, device, L, max_iters, seed):
    """Run the classic dual-SIM (alternating Algorithm 2) for `max_iters` iterations on (A, H)
    and evaluate accuracy. Use max_iters=100 for an equal-iteration comparison vs unrolling
    (same lr schedule 0.1/L^1.3 as the saved sweep, just stopped early instead of at 3000)."""
    A_t = torch.as_tensor(A_target, dtype=torch.complex64, device=device)
    H_t = torch.as_tensor(H_mimo, dtype=torch.complex64, device=device)
    lr = 0.1 / (L ** 1.3)                     # same lr schedule as run_experiment_layers
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        sim.optimize_alternating(A_target=A_target, H_mimo=H_mimo,
                                 max_iters=max_iters, lr=lr)
    beta, nmse = beta_and_nmse(sim, A_t, H_t)
    acc = run_evaluation(sim, dm_task.test_dataloader(), H_mimo, None, beta,
                         L_in, mu_in, L_out, mu_out, clf, device)
    return acc * 100, nmse


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=32, help="intermediate meta-atoms/side")
    ap.add_argument("-L", type=int, nargs="+", default=[2, 5, 10, 15])
    ap.add_argument("-K", type=int, default=400, help="unrolled layers / iterations")
    ap.add_argument("--alignment", choices=["PPFE", "Linear"], default="PPFE")
    ap.add_argument("--seed", type=int, default=42)
    # unrolled training. The untrained-momentum schedule already matches/beats the classic dual-SIM@3000
    # NMSE floor at K=400 (see diag_nmse_vs_k.py), so training defaults OFF (epochs=0); set
    # --epochs > 0 only to try squeezing the curve further (slow: per-layer autograd).
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--iters-per-epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-w", type=float, default=1e-2)
    ap.add_argument("--init-w", type=float, default=0.1)
    ap.add_argument("--init-mu", type=float, default=0.9)
    ap.add_argument("--no-first-order", dest="first_order", action="store_false",
                    help="use 2nd-order deep-sup (only fits for small L); default first-order")
    ap.set_defaults(first_order=True)
    ap.add_argument("--with-classic", "--with-analog", dest="with_classic", action="store_true",
                    help="also run the classic dual-SIM (alternating Algorithm 2) inline for a "
                         "same-run, equal-iteration comparison")
    ap.add_argument("--classic-iters", "--analog-iters", dest="classic_iters", type=int,
                    default=3000, help="iterations for the inline classic dual-SIM run "
                                       "(set 100 to compare at equal iterations with K=100)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTDIR.mkdir(exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    print(f"device={device} | M={args.M} L={args.L} K={args.K} | "
          f"alignment={args.alignment} seed={args.seed} first_order={args.first_order}")

    # ---- data, classifier, A_target, H_mimo (mirror Dual_classification_multiple.py) ----
    clf_path = MODEL_PATH / "data" / "classifiers" / "cifar10" / RX_NAME / f"seed_{args.seed}.ckpt"
    if not clf_path.exists():
        raise FileNotFoundError(f"classifier checkpoint missing: {clf_path}")

    print("loading datamodules + classifier ...")
    dm_align = DataModuleAlignmentClassification(
        dataset="cifar10", tx_enc=TX_NAME, rx_enc=RX_NAME,
        train_label_size=4200, method="centroid", batch_size=128, seed=args.seed)
    dm_align.setup()
    dm_task = DataModuleClassifier(dataset="cifar10", rx_enc=TX_NAME, batch_size=128)
    dm_task.setup()
    clf = Classifier.load_from_checkpoint(clf_path).to(device).eval()

    input_c = complex_compressed_tensor(dm_align.train_data.z_tx.T, device=device)
    output_c = complex_compressed_tensor(dm_align.train_data.z_rx.T, device=device)
    input_w, L_in, mu_in = prewhiten(input_c, device=device)
    output_w, L_out, mu_out = prewhiten(output_c, device=device)

    if args.alignment == "Linear":
        A_target = ridge_regression(input_w, output_w, lmb=1e-3)
    else:
        A_target = ppfe(input_w, output_w, output_real=dm_align.train_data.z_rx,
                        n_clusters=20, n_proto=1000, seed=args.seed)
    A_target = A_target.to(torch.complex64).to(device)
    H_mimo = complex_gaussian_matrix(0, 1, size=(384, 384)).to(device)
    print(f"A_target: {tuple(A_target.shape)} ||A||_F={torch.norm(A_target).item():.3f} | "
          f"H_mimo: {tuple(H_mimo.shape)}")

    oracle_acc = run_oracle_test(dm_task, A_target, L_in, mu_in, L_out, mu_out, clf, device) * 100
    print(f"oracle (ideal A, no SIM/channel): {oracle_acc:.2f}%")
    classic_conv_csv = load_classic_converged_accuracy(args.alignment, args.M, args.seed)

    # ---- unrolled sweep over L ----
    rows = []
    for L in args.L:
        print("\n" + "=" * 60)
        print(f"M={args.M} | L={L} | K={args.K}  (unrolled SCA)")
        print("=" * 60)
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
        sim = build_sim(L, args.M, device)
        N_T = sim.get_buffer(f"W_T_{sim.L_T - 1}").shape[0]
        N_R = sim.get_buffer("W_R_0").shape[1]

        t = time.time()
        if args.epochs > 0:
            model, _ = train_sca_unrolled(
                sim, A_target, args.K, N_T=N_T, N_R=N_R,
                epochs=args.epochs, iters_per_epoch=args.iters_per_epoch, batch=args.batch,
                lr_w=args.lr_w, init_w=args.init_w, coupling="diagonal", innovation_norm="rms",
                supervision="all", momentum=True, init_mu=args.init_mu,
                learn_S=True, first_order=args.first_order,
                device=device, seed=args.seed,
                val_H=torch.stack([H_mimo]),   # validate the schedule on the REAL channel
            )
            mode = f"trained {args.epochs}ep"
        else:
            # untrained-momentum schedule: already matches/beats classic dual-SIM@3000 at K=400.
            model = DualSIMUnrolledSCA(
                sim, args.K, init_w=args.init_w, coupling="diagonal", innovation_norm="rms",
                momentum=True, init_mu=args.init_mu, learn_S=True,
                first_order=args.first_order).to(device)
            mode = "untrained-momentum"
        train_s = time.time() - t

        ti = time.time()
        H1 = H_mimo.unsqueeze(0)
        xT, xR = model.infer_phases(H1, A_target)   # single K-layer forward -> phases
        set_sim_phases(sim, xT, xR)
        beta, nmse_u = beta_and_nmse(sim, A_target, H_mimo)
        acc_u = run_evaluation(sim, dm_task.test_dataloader(), H_mimo, None, beta,
                               L_in, mu_in, L_out, mu_out, clf, device) * 100
        print(f">> UNROLLED  L={L}: accuracy={acc_u:.2f}%  emulation_NMSE={nmse_u:.4f}  "
              f"|beta|={beta.abs().item():.2e}  [{mode}, build {train_s:.0f}s, "
              f"infer+eval {time.time() - ti:.0f}s]", flush=True)

        acc_c, nmse_c = (None, None)
        if args.with_classic:
            set_sim_phases(sim, [torch.zeros_like(p) for p in sim.xi_T],
                           [torch.zeros_like(p) for p in sim.xi_R])
            acc_c, nmse_c = classic_accuracy_inline(
                sim, A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out,
                device, L, args.classic_iters, args.seed)
            print(f">> CLASSIC dual-SIM L={L} ({args.classic_iters} it): "
                  f"accuracy={acc_c:.2f}%  emulation_NMSE={nmse_c:.4f}")

        acc_c_conv = classic_conv_csv.get(L)
        rows.append(dict(M=args.M, L=L, K=args.K, acc_unroll=acc_u, nmse_unroll=nmse_u,
                         classic_iters=(args.classic_iters if args.with_classic else None),
                         acc_classic_inline=acc_c, nmse_classic_inline=nmse_c,
                         acc_classic_conv_csv=acc_c_conv, oracle=oracle_acc))
        del model, sim
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- save + plot ----
    # include the training mode in the tag so a trained run (epochs>0) never overwrites the
    # untrained-momentum results/plots (and vice-versa); previous files keep their own names.
    train_tag = f"ep{args.epochs}" if args.epochs > 0 else "untrained"
    cmp_tag = f"_vsclassic{args.classic_iters}it" if args.with_classic else ""
    tag = f"{args.alignment}_M{args.M}_K{args.K}_{train_tag}{cmp_tag}_seed{args.seed}"
    csv_path = OUTDIR / f"accuracy_unrolled_{tag}.csv"
    fields = ["M", "L", "K", "acc_unroll", "nmse_unroll", "classic_iters",
              "acc_classic_inline", "nmse_classic_inline", "acc_classic_conv_csv", "oracle"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\nsaved CSV -> {csv_path}")

    print(f"\n=== Accuracy vs L (M={args.M}, {args.alignment}) ===")
    print(f"{'L':>4} {'unroll%':>9} {'classic%':>10} {'NMSE_u':>8}")
    for r in rows:
        c = r["acc_classic_inline"] if r["acc_classic_inline"] is not None else r["acc_classic_conv_csv"]
        print(f"{r['L']:>4} {r['acc_unroll']:>9.2f} "
              f"{(f'{c:.2f}' if c is not None else '--'):>10} {r['nmse_unroll']:>8.4f}")

    Ls = [r["L"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(Ls, [r["acc_unroll"] for r in rows], "o-", color="tab:blue", lw=2,
            label=f"Deep unrolling (SCA, K={args.K})")
    if args.with_classic and any(r["acc_classic_inline"] is not None for r in rows):
        # only the equal-iteration classic dual-SIM (NOT the converged 3000-iter reference)
        ax.plot(Ls, [r["acc_classic_inline"] for r in rows], "s--", color="tab:red", lw=2,
                label=f"Classic dual-SIM (alt. opt., {args.classic_iters} it)")
    ax.axhline(oracle_acc, color="gray", ls=":", lw=1.5, label=f"Oracle ({oracle_acc:.1f}%)")
    ax.set_xlabel("SIM layers  L")
    ax.set_ylabel("CIFAR-10 accuracy [%]")
    ax.set_title(f"Accuracy vs L  (M={args.M}, {args.alignment}, K={args.K})")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = OUTDIR / f"accuracy_unrolled_{tag}.{ext}"
        fig.savefig(p, dpi=150)
        print(f"saved plot -> {p}")


if __name__ == "__main__":
    main()
