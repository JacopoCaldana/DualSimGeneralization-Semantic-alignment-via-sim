import os
import sys
import torch
import json
from pathlib import Path

# 1. FORZA IL PERCORSO LOCALE
sys.path.insert(0, os.path.abspath(os.getcwd()))

# Importa solo gli strumenti necessari (senza attivare i cicli del main)
from alignment import DataModuleAlignmentClassification
from classifier import DataModuleClassifier
from alignment_utils import ridge_regression, ppfe
from utils import complex_compressed_tensor, prewhiten
from models_tasks.classification import Classifier
from oracle_test import run_oracle_test

def run_baselines(seeds=[27, 42, 123]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configurazione percorsi
    BASE_DIR = Path(__file__).resolve().parent
    TX_NAME = "vit_small_patch16_224" 
    RX_NAME = "vit_base_patch16_224" 
    
    final_results = {}

    for seed in seeds:
        print(f"\n--- 🔮 ELABORAZIONE SEED {seed} ---")
        
        # Setup Dati
        dm_align = DataModuleAlignmentClassification(
            dataset="cifar10", tx_enc=TX_NAME, rx_enc=RX_NAME,      
            train_label_size=4200, method='centroid', batch_size=128, seed=seed
        )
        dm_align.setup()
        dm_task = DataModuleClassifier(dataset="cifar10", rx_enc=TX_NAME, batch_size=128)
        dm_task.setup()
        
        # Caricamento Classificatore
        clf_path = BASE_DIR / "models" / "data" / "classifiers" / "cifar10" / RX_NAME / f"seed_{seed}.ckpt"
        clf = Classifier.load_from_checkpoint(clf_path).to(device).eval()

        # Whitening
        input_c = complex_compressed_tensor(dm_align.train_data.z_tx.T, device=device)
        output_c = complex_compressed_tensor(dm_align.train_data.z_rx.T, device=device)
        input_w, L_in, mu_in = prewhiten(input_c, device=device) 
        output_w, L_out, mu_out = prewhiten(output_c, device=device)

        # --- CALCOLO LINEAR ---
        A_linear = ridge_regression(input_w, output_w, lmb=1e-3)
        acc_linear = run_oracle_test(dm_task, A_linear, L_in, mu_in, L_out, mu_out, clf, device)

        # --- CALCOLO PPFE ---
        A_ppfe = ppfe(input_w, output_w, output_real=dm_align.train_data.z_rx, n_clusters=20, n_proto=1000, seed=seed)
        acc_ppfe = run_oracle_test(dm_task, A_ppfe, L_in, mu_in, L_out, mu_out, clf, device)

        final_results[str(seed)] = {
            "Linear": acc_linear * 100,
            "PPFE": acc_ppfe * 100
        }

    # Salva i risultati in un unico JSON per i tuoi plot
    with open("all_seeds_baselines.json", "w") as f:
        json.dump(final_results, f, indent=4)

    print("\n" + "="*40)
    print("✅ CALCOLO COMPLETATO")
    for s, res in final_results.items():
        print(f"Seed {s} -> Linear: {res['Linear']:.2f}% | PPFE: {res['PPFE']:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_baselines()