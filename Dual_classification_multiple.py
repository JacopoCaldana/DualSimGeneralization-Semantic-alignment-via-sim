import torch
import numpy as np
from pathlib import Path

# --- Local Modules Import ---
from alignment import DataModuleAlignmentClassification
from classifier import DataModuleClassifier
from alignment_utils import ridge_regression, ppfe
from utils import (
    complex_compressed_tensor, 
    prewhiten, 
    complex_gaussian_matrix,
)
from models_tasks.classification import Classifier
from oracle_test import run_oracle_test

# --- Experiment Execution Functions ---
from experiment_runner import run_experiment_layers, run_experiment_snr, run_experiment_1_mono_sim, run_sim_configuration_asymmetric,run_experiment_rx_depth,run_experiment_tx_depth, run_experiment_layers_disjoint, run_experiment_snr_disjoint


# ============================================================
# 1. ENVIRONMENT SETUP AND PATHS
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Absolute paths for project structure
#BASE_DIR = Path('/Users/jacopocaldana/Desktop/Università/Tesi')
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models'

# Encoder names used to build the checkpoint path
TX_NAME = "vit_small_patch16_224" 
RX_NAME = "vit_base_patch16_224" 
CLF_PATH = MODEL_PATH / "data" / "classifiers" / "cifar10" / RX_NAME / f"seed_{SEED}.ckpt"

print(f"🔍 Preliminary classifier check...")
if not CLF_PATH.exists():
    raise FileNotFoundError(f"Critical Error: Checkpoint does not exist at {CLF_PATH}. "
                            "Please verify the model folder before starting.")
print("✅ Checkpoint found. Proceeding...")


# ============================================================
# 2. DATA AND CLASSIFIER LOADING
# ============================================================
print("⏳ Loading Datamodules...")

TRAIN_SIZE_PER_CLASS = 4200
# Datamodule for Semantic Alignment (using a limited set of pilots)
dm_align = DataModuleAlignmentClassification(
    dataset="cifar10", tx_enc=TX_NAME, rx_enc=RX_NAME,      
    train_label_size=TRAIN_SIZE_PER_CLASS, method='centroid', batch_size=128, seed=SEED
)
dm_align.setup()

# Datamodule for the classification task evaluation
dm_task = DataModuleClassifier(dataset="cifar10", rx_enc=TX_NAME, batch_size=128)
dm_task.setup()

# Load pre-trained classifier to the compute device
clf = Classifier.load_from_checkpoint(CLF_PATH).to(device).eval()


# ============================================================
# 3. CALCULATION OF TARGET MATRIX A AND CHANNEL H
# ============================================================
# Complex Compression: Mapping real features to the complex domain
input_c = complex_compressed_tensor(dm_align.train_data.z_tx.T, device=device)
output_c = complex_compressed_tensor(dm_align.train_data.z_rx.T, device=device)

# Pre-whitening: Normalizing the latent spaces
input_w, L_in, mu_in = prewhiten(input_c, device=device) 
output_w, L_out, mu_out = prewhiten(output_c, device=device)

# --- ALIGNMENT STRATEGY SELECTION ---
ALIGNMENT_TYPE = 'PPFE'  # Set to 'PPFE' for Zero-Shot alignment

if ALIGNMENT_TYPE == 'Linear':
    # Supervised Ridge Regression to find the mapping between spaces
    A_target = ridge_regression(input_w, output_w, lmb=1e-3)
    print(f"✅ Target matrix A calculated (Supervised - Linear)")
else:
    # Prototype-based Parseval Frame Equalization (Zero-Shot)
    A_target = ppfe(
        input_w, output_w, 
        output_real=dm_align.train_data.z_rx, 
        n_clusters=20, n_proto=1000, seed=SEED
    )
    print(f"✅ Target matrix A calculated (Zero-Shot - PPFE)")

# Real Rayleigh Fading MIMO Channel Generation (as described in Eq. 8)
H_mimo = complex_gaussian_matrix(0, 1, size=(384, 384)).to(device)
# ----No Channel---
#H_mimo = torch.eye(384, dtype=torch.complex64, device=device)


# ============================================================
# 4. BASELINE CALCULATION (ORACLE)
# ============================================================
print("\n🔮 Calculating Baseline Accuracy (Oracle)...")
# The Oracle represents the ideal software-only performance without the SIM hardware constraints
oracle_acc = run_oracle_test(dm_task, A_target, L_in, mu_in, L_out, mu_out, clf, device)
print(f"🎯 Baseline (Ideal A_target without channel): {oracle_acc * 100:.2f}%")


# ============================================================
# 5. EXPERIMENT EXECUTION
# ============================================================
# Decomment the experiment you wish to run

# --- EXPERIMENT 1: Impact of Meta-surface Layers (L) ---
print(f"🚀 Starting Experiment 1: Layers Variation ({ALIGNMENT_TYPE})...")
data_layers = run_experiment_layers(
      A_target=A_target, H_mimo=H_mimo, dm_task=dm_task, clf=clf, 
      L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, device=device,
      strategy_name=ALIGNMENT_TYPE
)

# --- EXPERIMENT 2: Impact of Signal-to-Noise Ratio (SNR) ---
# print(f"\n 🚀 Starting Experiment 2: SNR Sweep ({ALIGNMENT_TYPE})...")
# data_snr = run_experiment_snr(
#     A_target=A_target, H_mimo=H_mimo, dm_task=dm_task, clf=clf, 
#     L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, device=device,
#     strategy_name=ALIGNMENT_TYPE
# )

# --- MONO-SIM ABLATION STUDY ---
# print(f"\n 🚀 Starting Mono-SIM Ablation Study ({ALIGNMENT_TYPE})...")
# data_mono = run_experiment_1_mono_sim(
#     A_target=A_target, dm_task=dm_task, clf=clf, 
#     L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, device=device,
#     strategy_name=ALIGNMENT_TYPE
# )

# --- ASYMMETRIC RX DEPTH ---
# print(f"🚀 Starting Asymmetric RX experiment ({ALIGNMENT_TYPE})..")
# data_asym = run_experiment_rx_depth(
#     A_target=A_target, H_mimo=H_mimo, dm_task=dm_task, clf=clf, 
#     L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, device=device,
#     strategy_name=ALIGNMENT_TYPE
# )

# --- ASYMMETRIC TX DEPTH ---
# print(f"🚀 Starting Asymmetric TX experiment ({ALIGNMENT_TYPE})..")
# data_tx_var = run_experiment_tx_depth(
#     A_target=A_target, H_mimo=H_mimo, dm_task=dm_task, clf=clf, 
#     L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, device=device,
#     strategy_name=ALIGNMENT_TYPE
# )

# ============================================================
#  DISJOINT OPTIMIZATION EXPERIMENTS 
# ============================================================

# --- DISJOINT EXP 1: Impact of Layers (L) (without Noise) ---
# print("\n 🧩 Starting Disjoint Experiment 1: Layers Variation...")
# data_layers_disjoint = run_experiment_layers_disjoint(
#     A_target=A_target, H_mimo=H_mimo, dm_task=dm_task, clf=clf, 
#     L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, device=device
# )

# --- DISJOINT EXP 2: Impact of SNR (Dynamic RX Re-training) ---
#print("\n 🧩 Starting Disjoint Experiment 2: SNR Sweep...")
#data_snr_disjoint = run_experiment_snr_disjoint(
 #    A_target=A_target, H_mimo=H_mimo, dm_task=dm_task, clf=clf, 
  #   L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, device=device
#)