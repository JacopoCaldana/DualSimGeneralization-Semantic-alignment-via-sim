"""
Accuracy vs unrolled depth K (== complexity / number of iterations).

Two parametric sweeps, ONE run:
  (1) curves parametrised by SIM-layer count L  (atoms/layer M fixed)   -> plot "byL"
  (2) curves parametrised by atoms/layer M       (SIM layers L fixed)   -> plot "byM"

For every config we evaluate the deep-unrolled SCA network at several K checkpoints
(x-axis) and, as a dashed reference of the SAME colour, the CLASSIC dual-SIM  run for the SAME iteration budget. The message: how FEW unrolled layers reach
the accuracy the classic dual-SIM needs many more iterations for.

Why a fresh untrained model per K is exact: the untrained-momentum schedule uses a CONSTANT
per-layer step (w_T = full(K, init_w), mu = full(K, init_mu)), so a K=k model's trajectory is
exactly the first k steps of any larger model -> evaluating at each checkpoint is consistent.

Reuses verbatim the CIFAR-10 over-the-air pipeline, the real A_target (PPFE/Linear) and the
real channel H_mimo of `accuracy_unrolled.py` (and its build_sim / set_sim_phases /
beta_and_nmse / classic_accuracy_inline helpers), so numbers are directly comparable.

Run:
    source venv/bin/activate
    python accuracy_vs_k.py            # byL: M=32,L=2,5,10,15 ; byM: L=10,M=16,32,64 ; PPFE seed42
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import argparse
import csv
import gc
import random
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sca_unrolled import DualSIMUnrolledSCA, train_sca_unrolled
from inference import run_evaluation
from oracle_test import run_oracle_test

# reuse the exact helpers + constants of the accuracy experiment (no duplication, no edits)
from accuracy_unrolled import (
    build_sim, set_sim_phases, beta_and_nmse, classic_accuracy_inline,
    TX_NAME, RX_NAME, MODEL_PATH, OUTDIR,
)

from alignment import DataModuleAlignmentClassification
from classifier import DataModuleClassifier
from alignment_utils import ridge_regression, ppfe
from utils import complex_compressed_tensor, prewhiten, complex_gaussian_matrix
from models_tasks.classification import Classifier

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
def setup_task(alignment, seed, device, n_test=1, n_val=4):
    """Build datamodules, classifier, real A_target, and a TRAIN/TEST split of synthetic
    channels: H_test (n_test held-out channels, seed+1000) for final accuracy, H_val (n_val,
    seed+500) for best-restore during training. Training H are sampled INSIDE train_sca_unrolled
    from the base `seed` -> a disjoint i.i.d. stream, so train and test channels never overlap."""
    clf_path = MODEL_PATH / "data" / "classifiers" / "cifar10" / RX_NAME / f"seed_{seed}.ckpt"
    if not clf_path.exists():
        raise FileNotFoundError(f"classifier checkpoint missing: {clf_path}")

    print("loading datamodules + classifier ...")
    dm_align = DataModuleAlignmentClassification(
        dataset="cifar10", tx_enc=TX_NAME, rx_enc=RX_NAME,
        train_label_size=4200, method="centroid", batch_size=128, seed=seed)
    dm_align.setup()
    dm_task = DataModuleClassifier(dataset="cifar10", rx_enc=TX_NAME, batch_size=128)
    dm_task.setup()
    clf = Classifier.load_from_checkpoint(clf_path).to(device).eval()

    input_c = complex_compressed_tensor(dm_align.train_data.z_tx.T, device=device)
    output_c = complex_compressed_tensor(dm_align.train_data.z_rx.T, device=device)
    input_w, L_in, mu_in = prewhiten(input_c, device=device)
    output_w, L_out, mu_out = prewhiten(output_c, device=device)

    if alignment == "Linear":
        A_target = ridge_regression(input_w, output_w, lmb=1e-3)
    else:
        A_target = ppfe(input_w, output_w, output_real=dm_align.train_data.z_rx,
                        n_clusters=20, n_proto=1000, seed=seed)
    A_target = A_target.to(torch.complex64).to(device)
    def _channels(n, sd):
        g = torch.Generator().manual_seed(sd)
        H = (torch.randn(n, 384, 384, generator=g) + 1j * torch.randn(n, 384, 384, generator=g))
        return (H / (2 ** 0.5)).to(torch.complex64).to(device)
    H_test = _channels(n_test, seed + 1000)   # held-out: final accuracy
    H_val = _channels(n_val, seed + 500)       # held-out: best-restore during training
    print(f"A_target: {tuple(A_target.shape)} ||A||_F={torch.norm(A_target).item():.3f} | "
          f"H_test: {tuple(H_test.shape)} (seed {seed + 1000}) | H_val: {tuple(H_val.shape)} (seed {seed + 500})")
    return dict(dm_task=dm_task, clf=clf, A_target=A_target, H_test=H_test, H_val=H_val,
                L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out)


@torch.no_grad()
def _zero_phases(sim):
    for p in sim.xi_T:
        p.zero_()
    for p in sim.xi_R:
        p.zero_()


def eval_unroll_at_k(sim, k, L, t, args, device):
    """Unrolled net of depth k -> phases per held-out test channel -> accuracy + NMSE,
    averaged over t['H_test']. If args.epochs>0 the per-layer schedule is TRAINED first on
    random training channels (sampled inside train_sca_unrolled from args.seed, a stream
    disjoint from the test/val seeds); otherwise the untrained-momentum schedule is used."""
    if args.epochs > 0:
        N_T = sim.get_buffer(f"W_T_{sim.L_T - 1}").shape[0]
        N_R = sim.get_buffer("W_R_0").shape[1]
        model, _ = train_sca_unrolled(
            sim, t["A_target"], k, N_T=N_T, N_R=N_R,
            epochs=args.epochs, iters_per_epoch=args.iters_per_epoch, batch=args.batch,
            lr_w=args.lr_w, init_w=args.init_w, coupling="diagonal", innovation_norm="rms",
            supervision="all", momentum=True, init_mu=args.init_mu, learn_S=True,
            first_order=args.first_order, device=device, seed=args.seed, val_H=t["H_val"])
    else:
        model = DualSIMUnrolledSCA(
            sim, k, init_w=args.init_w, coupling="diagonal", innovation_norm="rms",
            momentum=True, init_mu=args.init_mu, learn_S=True, first_order=args.first_order).to(device)
    accs, nmses = [], []
    for H in t["H_test"]:
        xT, xR = model.infer_phases(H.unsqueeze(0), t["A_target"])
        set_sim_phases(sim, xT, xR)
        beta, nmse = beta_and_nmse(sim, t["A_target"], H)
        acc = run_evaluation(sim, t["dm_task"].test_dataloader(), H, None, beta,
                             t["L_in"], t["mu_in"], t["L_out"], t["mu_out"], t["clf"], device) * 100
        accs.append(acc); nmses.append(nmse)
    del model
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(nmses))


def eval_classic_at_k(sim, k, L, t, device, seed):
    """Classic dual-SIM for k iterations -> accuracy + NMSE, averaged over t['H_test']."""
    accs, nmses = [], []
    for H in t["H_test"]:
        _zero_phases(sim)
        acc, nmse = classic_accuracy_inline(
            sim, t["A_target"], H, t["dm_task"], t["clf"],
            t["L_in"], t["mu_in"], t["L_out"], t["mu_out"], device, L, k, seed)
        accs.append(acc); nmses.append(nmse)
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(nmses))


# --------------------------------------------------------------------------- #
def sweep(configs, ks, t, args, device):
    """configs: list of (label_value, L, M). Returns rows (one per config x k)."""
    rows = []
    for val, L, M in configs:
        print("\n" + "=" * 64)
        print(f"config L={L} M={M}  ->  K sweep {ks}")
        print("=" * 64)
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
        sim = build_sim(L, M, device)
        for k in ks:
            tu = time.time()
            acc_u, std_u, nmse_u = eval_unroll_at_k(sim, k, L, t, args, device)
            acc_c, std_c, nmse_c = (None, None, None)
            if args.classic:
                acc_c, std_c, nmse_c = eval_classic_at_k(sim, k, L, t, device, args.seed)
            print(f"  K={k:>4} | unroll {acc_u:6.2f}+-{std_u:4.2f}% (NMSE {nmse_u:.4f})"
                  + (f" | classic {acc_c:6.2f}+-{std_c:4.2f}% (NMSE {nmse_c:.4f})" if args.classic else "")
                  + f"   [{time.time() - tu:.0f}s]", flush=True)
            rows.append(dict(L=L, M=M, K=k, acc_unroll=acc_u, acc_unroll_std=std_u,
                             nmse_unroll=nmse_u, acc_classic=acc_c, acc_classic_std=std_c,
                             nmse_classic=nmse_c))
        del sim
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def write_csv(rows, tag):
    csv_path = OUTDIR / f"acc_vs_k_{tag}.csv"
    fields = ["L", "M", "K", "acc_unroll", "acc_unroll_std", "nmse_unroll",
              "acc_classic", "acc_classic_std", "nmse_classic"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\nsaved CSV -> {csv_path}")


def read_csv_rows(tag):
    """Load rows back from a previously-saved acc_vs_k CSV (for --replot, no re-run)."""
    csv_path = OUTDIR / f"acc_vs_k_{tag}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found for replot: {csv_path}")
    def fnum(x):
        return None if x in (None, "", "None") else float(x)
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(L=int(r["L"]), M=int(r["M"]), K=int(r["K"]),
                             acc_unroll=fnum(r["acc_unroll"]), acc_unroll_std=fnum(r.get("acc_unroll_std")),
                             nmse_unroll=fnum(r["nmse_unroll"]),
                             acc_classic=fnum(r["acc_classic"]), acc_classic_std=fnum(r.get("acc_classic_std")),
                             nmse_classic=fnum(r["nmse_classic"])))
    return rows


def plot_curves(rows, configs, ks, oracle_acc, tag, param_name, args):
    """One accuracy-vs-K figure from `rows`, restricted to the K values in `ks`
    (solid unroll, dashed classic, per config). Does NOT write the CSV."""
    kset = set(ks)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    cmap = plt.get_cmap("viridis")
    n = max(len(configs) - 1, 1)
    for i, (val, L, M) in enumerate(configs):
        color = cmap(i / n)
        sub = sorted([r for r in rows if r["L"] == L and r["M"] == M and r["K"] in kset],
                     key=lambda r: r["K"])
        if not sub:
            continue
        xs = [r["K"] for r in sub]
        ax.plot(xs, [r["acc_unroll"] for r in sub], "o-", color=color, lw=2,
                label=f"unroll  {param_name}={val}")
        if all(r["acc_classic"] is not None for r in sub):
            ax.plot(xs, [r["acc_classic"] for r in sub], "s--", color=color, lw=1.6, alpha=0.8,
                    label=f"classic {param_name}={val}")
    if oracle_acc is not None:
        ax.axhline(oracle_acc, color="gray", ls=":", lw=1.5, label=f"Oracle ({oracle_acc:.1f}%)")
    ax.set_xscale("log")
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("unrolled layers / iterations  K  (complexity)")
    ax.set_ylabel("CIFAR-10 accuracy [%]")
    fixed_txt = (f"M={args.fixed_M}" if param_name == "L" else f"L={args.fixed_L}")
    ax.set_title(f"Accuracy vs K  ({args.alignment}, {fixed_txt})  -- unroll (solid) vs classic dual-SIM (dashed)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = OUTDIR / f"acc_vs_k_{tag}.{ext}"
        fig.savefig(p, dpi=150)
        print(f"saved plot -> {p}")
    plt.close(fig)


def do_sweep(name, configs, param_name, base_tag, t, oracle_acc, args, device):
    """Run (or, with --replot, load) one sweep, then plot it capped at --kmax.
    When --kmax is set the plot tag gets a _kmax<N> suffix so the full-range plots are kept."""
    if args.replot:
        try:
            rows = read_csv_rows(base_tag)
        except FileNotFoundError as e:
            print(f"[skip {name}] {e}")
            return
        ks_avail = sorted({r["K"] for r in rows})
        oracle = args.oracle  # not stored in the CSV; pass via --oracle to draw the line
    else:
        rows = sweep(configs, args.ks, t, args, device)
        write_csv(rows, base_tag)
        ks_avail = list(args.ks)
        oracle = oracle_acc
    ks_plot = [k for k in ks_avail if args.kmax is None or k <= args.kmax]
    plot_tag = base_tag + (f"_kmax{args.kmax}" if args.kmax is not None else "")
    plot_curves(rows, configs, ks_plot, oracle, plot_tag, param_name, args)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignment", choices=["PPFE", "Linear"], default="PPFE")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 20, 50, 100, 200, 400],
                    help="K checkpoints on the x-axis (unrolled layers / classic iterations)")
    # sweep (1): curves per L at fixed atoms/layer
    ap.add_argument("--ls", type=int, nargs="+", default=[2, 5, 10, 15], help="L values for the byL plot")
    ap.add_argument("--fixed-M", type=int, default=32, help="atoms/layer held fixed in the byL plot")
    # sweep (2): curves per atoms/layer at fixed L
    ap.add_argument("--ms", type=int, nargs="+", default=[16, 32, 64], help="atoms/layer for the byM plot")
    ap.add_argument("--fixed-L", type=int, default=10, help="SIM layers held fixed in the byM plot")
    # unrolled net (untrained-momentum, same recipe as accuracy_unrolled defaults)
    ap.add_argument("--init-w", type=float, default=0.1)
    ap.add_argument("--init-mu", type=float, default=0.9)
    ap.add_argument("--no-first-order", dest="first_order", action="store_false")
    ap.set_defaults(first_order=True)
    # held-out evaluation + optional training (train/test split of channels)
    ap.add_argument("--n-channels", type=int, default=1,
                    help="held-out TEST channels to average accuracy over (seed+1000)")
    ap.add_argument("--n-val", type=int, default=4,
                    help="held-out VAL channels for best-restore during training (seed+500)")
    ap.add_argument("--epochs", type=int, default=0,
                    help="train the per-layer schedule for this many epochs (0 = untrained-momentum); "
                         ">0 -> per-K training on random channels, eval on held-out TEST channels")
    ap.add_argument("--iters-per-epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-w", type=float, default=1e-2)
    ap.add_argument("--classic", action=argparse.BooleanOptionalAction, default=True,
                    help="also evaluate the classic dual-SIM at each K (dashed); --no-classic to skip")
    ap.add_argument("--only", choices=["byL", "byM", "both"], default="both")
    # replot / cap without re-running the (slow) sweep
    ap.add_argument("--kmax", type=int, default=None,
                    help="cap K for plotting (keep only K<=kmax); plot files get a _kmax<N> suffix "
                         "so the full-range plots are NOT overwritten")
    ap.add_argument("--replot", action="store_true",
                    help="regenerate the plot(s) from the existing acc_vs_k CSV(s) WITHOUT re-running")
    ap.add_argument("--oracle", type=float, default=None,
                    help="oracle %% for the dotted reference line in --replot mode (not stored in CSV)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTDIR.mkdir(exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    print(f"device={device} | alignment={args.alignment} seed={args.seed} | ks={args.ks}\n"
          f"  byL: L={args.ls} (M={args.fixed_M}) | byM: M={args.ms} (L={args.fixed_L}) | "
          f"classic={args.classic} first_order={args.first_order}")

    if args.replot:
        t, oracle_acc = None, None
        print("REPLOT mode: regenerating plots from existing CSV(s) -- NO simulation re-run."
              + ("" if args.oracle is not None else "  (no --oracle -> oracle line omitted)"))
    else:
        t = setup_task(args.alignment, args.seed, device, args.n_channels, args.n_val)
        oracle_acc = run_oracle_test(t["dm_task"], t["A_target"], t["L_in"], t["mu_in"],
                                     t["L_out"], t["mu_out"], t["clf"], device) * 100
        print(f"oracle (ideal A, no SIM/channel): {oracle_acc:.2f}%")

    # variant suffix so trained / multi-channel runs do NOT overwrite the untrained single-H files
    variant = (f"_ep{args.epochs}" if args.epochs > 0 else "") \
              + (f"_nch{args.n_channels}" if args.n_channels > 1 else "")

    if args.only in ("byL", "both"):
        print("\n######## SWEEP 1: accuracy vs K, curves per L (atoms/layer fixed) ########")
        configs = [(L, L, args.fixed_M) for L in args.ls]
        do_sweep("byL", configs, "L", f"byL_{args.alignment}_M{args.fixed_M}_seed{args.seed}{variant}",
                 t, oracle_acc, args, device)

    if args.only in ("byM", "both"):
        print("\n######## SWEEP 2: accuracy vs K, curves per atoms/layer (L fixed) ########")
        configs = [(M, args.fixed_L, M) for M in args.ms]
        do_sweep("byM", configs, "M", f"byM_{args.alignment}_L{args.fixed_L}_seed{args.seed}{variant}",
                 t, oracle_acc, args, device)


if __name__ == "__main__":
    main()
