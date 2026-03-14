import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Local Modules Import ---
from dualsim import DualSIMoptimizer, DualSIMoptimizerTorch
from alignment import DataModuleAlignmentClassification
from classifier import DataModuleClassifier
from alignment_utils import ridge_regression, ppfe
from utils import (
    complex_compressed_tensor, 
    prewhiten, 
    complex_gaussian_matrix,
)
from download_utils import download_models_ckpt
from inference import run_evaluation, run_evaluation_mmse
from models_tasks.classification import Classifier
from oracle_test import run_oracle_test
from experiment_runner import run_experiment_layers, run_experiment_snr, run_experiment_1_mono_sim


# ============================================================
# 1. ENVIRONMENT SETUP AND PATHS (Pre-flight Check)
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Compute device activated: {device}")

# Absolute paths to avoid ambiguity
#BASE_DIR = Path('/Users/jacopocaldana/Desktop/Università/Tesi')
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models'

# Encoder names to construct the checkpoint path
TX_NAME = "vit_small_patch16_224" 
RX_NAME = "vit_base_patch16_224" 

# Path to the pre-trained classifier checkpoint
CLF_PATH = MODEL_PATH / "data" / "classifiers" / "cifar10" / RX_NAME / f"seed_{SEED}.ckpt"

print(f"🔍 Preliminary classifier check...")
if not CLF_PATH.exists():
    raise FileNotFoundError(f"Critical Error: Checkpoint does not exist at {CLF_PATH}. "
                            "Verify the folder before starting the optimization!")
print("✅ Checkpoint found. Proceeding with data loading...")

# ============================================================
# 2. DATA LOADING (Dual Datamodule)
# ============================================================
# Datamodule for Semantic Alignment training
dm_align = DataModuleAlignmentClassification(
    dataset="cifar10",
    tx_enc=TX_NAME,      
    rx_enc=RX_NAME,      
    train_label_size=100, # Number of Semantic Pilots
    method='centroid',    
    batch_size=128,
    seed=SEED
)
dm_align.setup()

# Datamodule for the classification task evaluation
dm_task = DataModuleClassifier(
    dataset="cifar10",
    rx_enc=TX_NAME,     
    batch_size=128
)
dm_task.setup()

# ============================================================
# 3. TARGET MATRIX A CALCULATION (Strategy)
# ============================================================

# Complex compression of semantic pilots
# input_c: From ViT-Small -> 384 real -> 192 complex
input_c = complex_compressed_tensor(dm_align.train_data.z_tx.T, device=device)
# output_c: From ViT-Base -> 768 real -> 384 complex
output_c = complex_compressed_tensor(dm_align.train_data.z_rx.T, device=device)

# --- Pre-whitening Process ---
# TX Whitening (Dimension 192)
input_w, L_in, mu_in = prewhiten(input_c, device=device) 
# RX Whitening (Dimension 384)
output_w, L_out, mu_out = prewhiten(output_c, device=device)

# --- ALIGNMENT STRATEGY SELECTION ---
ALIGNMENT_TYPE = 'Linear'  # Change to 'PPFE' for Zero-Shot mode

if ALIGNMENT_TYPE == 'Linear':
    # Ridge Regression (Supervised) to project from 192 to 384
    A_target = ridge_regression(input_w, output_w, lmb=1e-3)
    print(f"✅ Target matrix A calculated via Ridge Regression (Supervised)")
else:
    # PPFE (Zero-Shot) based on prototypes and clustering
    A_target = ppfe(
        input_w, output_w, 
        output_real=dm_align.train_data.z_rx, 
        n_clusters=10, n_proto=10, seed=SEED
    )
    print(f"✅ Target matrix A calculated via PPFE (Zero-Shot)")


# ============================================================
# 4. PHYSICAL DUAL-SIM AND CHANNEL INITIALIZATION
# ============================================================
wavelength = 0.005  # lambda = 5mm (60 GHz)
slayer = 2 * wavelength 
dx = wavelength / 2 

# Initialize the physical SIM parameters (NumPy-based)
sim_cpu = DualSIMoptimizer(
    num_layers_TX=10, 
    num_meta_atoms_TX_in_x=16, num_meta_atoms_TX_in_y=12,    # 192 semantic complexity
    num_meta_atoms_TX_out_x=24, num_meta_atoms_TX_out_y=16,  # 384 antennas
    num_meta_atoms_TX_int_x=32, num_meta_atoms_TX_int_y=32,  
    thickness_TX=slayer * 10,
    
    num_layers_RX=10,
    num_meta_atoms_RX_in_x=24, num_meta_atoms_RX_in_y=16,    # 384 antennas
    num_meta_atoms_RX_out_x=24, num_meta_atoms_RX_out_y=16,  # 384 semantic reconstruction
    num_meta_atoms_RX_int_x=32, num_meta_atoms_RX_int_y=32,
    thickness_RX=slayer * 10,
    
    wavelength=wavelength,
    spacings={'tx_in': dx, 'tx_out': dx, 'tx_int': dx, 'rx_in': dx, 'rx_out': dx, 'rx_int': dx},
    verbose=True # Shows progress of W propagation matrices calculation
)

# Convert to PyTorch Model
model = DualSIMoptimizerTorch(sim_cpu).to(device)

# Rayleigh Fading MIMO Channel Initialization (Eq. 8)
H_mimo = complex_gaussian_matrix(0, 1, size=(384, 384)).to(device)

# ============================================================
# 5. ALTERNATING OPTIMIZATION (ALGORITHM 2)
# ============================================================
# Optimization process for phase shifts xi_T and xi_R
loss_history = model.optimize_alternating(
    A_target=A_target, 
    H_mimo=H_mimo, 
    max_iters=500, 
    lr=0.1, 
    lambda_reg=1e-4
)

# Final calculation of the optimal scaling factor beta (Eq. 17)
with torch.no_grad():
    Z_final, _ = model.get_effective_cascade(H_mimo)
    beta_opt = torch.sum(torch.conj(Z_final) * A_target) / torch.sum(torch.conj(Z_final) * Z_final)

# ============================================================
# 6. FINAL EVALUATION (SNR SWEEP: SIM Performance)
# ============================================================
# Load the classifier for the downstream task
clf = Classifier.load_from_checkpoint(CLF_PATH).to(device).eval()

# SNR [dB] range for thesis evaluation
snr_range = [-10, 0, 10, 20, 30, 40]
results_sim = []

print(f"\n📊 Starting Comparative Evaluation ({ALIGNMENT_TYPE})...")

for snr in snr_range:
    # Dual-SIM Over-the-Air Evaluation
    acc_sim = run_evaluation(
        model=model, 
        dataloader=dm_task.test_dataloader(), 
        H_mimo=H_mimo, 
        snr_db=float(snr), 
        beta_opt=beta_opt, 
        L_in=L_in, mu_in=mu_in, 
        L_out=L_out, mu_out=mu_out, 
        clf=clf, 
        device=device
    )
    results_sim.append(acc_sim * 100)
    
    print(f"📡 SNR: {snr:3}dB | SIM Accuracy: {acc_sim*100:6.2f}%")

# ============================================================
# 7. PERFORMANCE VISUALIZATION
# ============================================================
plt.figure(figsize=(12, 5))

# Plot 1: Loss Convergence
plt.subplot(1, 2, 1)
plt.plot(loss_history, color='green', lw=2)
plt.yscale('log')
plt.title('Semantic Alignment Convergence')
plt.xlabel('Iterations'), plt.ylabel('Loss (Log Scale)')
plt.grid(True, alpha=0.3)

# Plot 2: Accuracy vs SNR 
plt.subplot(1, 2, 2)
plt.plot(snr_range, results_sim, marker='o', linestyle='-', color='blue', lw=2)
plt.title(f'Dual-SIM Performance ({ALIGNMENT_TYPE})')
plt.xlabel('SNR [dB]'), plt.ylabel('Accuracy [%]')
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(BASE_DIR / 'final_results_plot.png') # Automatic plot save
plt.show()

# Final Oracle Test (Upper bound verification)
oracle_acc = run_oracle_test(dm_task, A_target, L_in, mu_in, L_out, mu_out, clf, device)
print(f"🎯 Oracle Accuracy (Upper Bound): {oracle_acc * 100:.2f}%")