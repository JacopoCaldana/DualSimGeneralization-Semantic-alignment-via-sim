import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
import random
from pathlib import Path

# --- Moduli Core della Pipeline ---
from alignment import DataModuleAlignmentClassification
from classifier import DataModuleClassifier
from alignment_utils import ridge_regression, ppfe
from utils import complex_compressed_tensor, prewhiten, complex_gaussian_matrix
from models_tasks.classification import Classifier
from oracle_test import run_oracle_test

# --- Import dell'Esperimento della Parte 2 Separata ---
from unrolled_runner import run_unrolled_experiment_layers

SEED_LIST = [27, 42, 123]
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models'
TX_NAME = "vit_small_patch16_224" 
RX_NAME = "vit_base_patch16_224" 

for current_seed in SEED_LIST:
    print("\n" + "#"*60)
    print(f"🌍 INIZIO RUN COMPLETA UNROLLED PER SEED: {current_seed}")
    print("#"*60)

    torch.manual_seed(current_seed)
    np.random.seed(current_seed)
    random.seed(current_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CLF_PATH = MODEL_PATH / "data" / "classifiers" / "cifar10" / RX_NAME / f"seed_{current_seed}.ckpt"
    if not CLF_PATH.exists():
        raise FileNotFoundError(f"Errore: Checkpoint mancante a {CLF_PATH}")

    # Caricamento Datamodules e Classificatore
    dm_align = DataModuleAlignmentClassification(
        dataset="cifar10", tx_enc=TX_NAME, rx_enc=RX_NAME,      
        train_label_size=4200, method='centroid', batch_size=128, seed=current_seed
    )
    dm_align.setup()

    dm_task = DataModuleClassifier(dataset="cifar10", rx_enc=TX_NAME, batch_size=128)
    dm_task.setup()
    clf = Classifier.load_from_checkpoint(CLF_PATH).to(device).eval()

    # Pre-whitening e compressione
    input_c = complex_compressed_tensor(dm_align.train_data.z_tx.T, device=device)
    output_c = complex_compressed_tensor(dm_align.train_data.z_rx.T, device=device)
    input_w, L_in, mu_in = prewhiten(input_c, device=device) 
    output_w, L_out, mu_out = prewhiten(output_c, device=device)

    ALIGNMENT_TYPE = 'PPFE' 

    if ALIGNMENT_TYPE == 'Linear':
        A_target = ridge_regression(input_w, output_w, lmb=1e-3)
    else:
        A_target = ppfe(input_w, output_w, output_real=dm_align.train_data.z_rx, n_clusters=20, n_proto=1000, seed=current_seed)

    H_mimo = complex_gaussian_matrix(0, 1, size=(384, 384)).to(device)

    # Calcolo baseline Oracle
    oracle_acc = run_oracle_test(dm_task, A_target, L_in, mu_in, L_out, mu_out, clf, device)
    print(f"🎯 Baseline Oracle (Ideal): {oracle_acc * 100:.2f}%")

    # ============================================================
    # EXECUTION PARTE 2: DEEP UNROLLING EXPERIMENT
    # ============================================================
    run_unrolled_experiment_layers(
        A_target=A_target, H_mimo=H_mimo, dm_task=dm_task, clf=clf, 
        L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, device=device,
        strategy_name=ALIGNMENT_TYPE, seed=current_seed
    )