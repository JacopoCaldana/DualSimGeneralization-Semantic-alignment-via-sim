"""Fast feasibility probe: how far does the UNTRAINED-momentum unrolled SCA net descend
on the REAL (A_target, H_mimo), as a function of unrolled depth K? One forward per L
(no training loop), so it's cheap -- it tells us the K each L needs to reach the analog
NMSE floor (analog@3000: L2~0.92, L5~0.73, L10~0.62, L15~0.67 for PPFE M32)."""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import random
import numpy as np
import torch

from accuracy_unrolled import build_sim
from sca_unrolled import DualSIMUnrolledSCA
from alignment import DataModuleAlignmentClassification
from alignment_utils import ppfe
from utils import complex_compressed_tensor, prewhiten, complex_gaussian_matrix

SEED = 42
K = 600
TX_NAME, RX_NAME = "vit_small_patch16_224", "vit_base_patch16_224"
ANALOG = {2: 0.92, 5: 0.73, 10: 0.62, 15: 0.67}   # analog@3000 NMSE floor (PPFE M32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

dm = DataModuleAlignmentClassification(dataset="cifar10", tx_enc=TX_NAME, rx_enc=RX_NAME,
                                       train_label_size=4200, method="centroid",
                                       batch_size=128, seed=SEED)
dm.setup()
ic = complex_compressed_tensor(dm.train_data.z_tx.T, device=device)
oc = complex_compressed_tensor(dm.train_data.z_rx.T, device=device)
iw, _, _ = prewhiten(ic, device=device)
ow, _, _ = prewhiten(oc, device=device)
A = ppfe(iw, ow, output_real=dm.train_data.z_rx, n_clusters=20, n_proto=1000,
         seed=SEED).to(torch.complex64).to(device)
H = complex_gaussian_matrix(0, 1, size=(384, 384)).to(device).unsqueeze(0)
print(f"A {tuple(A.shape)} ||A||_F={torch.norm(A).item():.3f} | H {tuple(H.shape)}", flush=True)

ck = [10, 25, 50, 100, 200, 300, 400, 600]
print(f"\n{'L':>3} " + " ".join(f"k={k:<4}" for k in ck) + "   analog@3000", flush=True)
for L in (2, 5, 10, 15):
    torch.manual_seed(SEED)
    sim = build_sim(L, 32, device)
    model = DualSIMUnrolledSCA(sim, K, init_w=0.1, coupling="diagonal", innovation_norm="rms",
                               momentum=True, init_mu=0.9, learn_S=True,
                               first_order=False).to(device)
    with torch.enable_grad():
        curve = model.eval_curve(H, A)            # len K+1, NMSE per layer on the real H
    vals = " ".join(f"{curve[k]:<6.3f}" for k in ck)
    print(f"{L:>3} {vals}   {ANALOG[L]:.2f}", flush=True)
    del sim, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
