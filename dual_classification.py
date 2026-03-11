import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Import moduli locali ---
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

#############################################
####ISPEZIONE SIM###########à
####################################################
def inspect_sim_health(model):
    print("\n🩺 --- CONTROLLO SALUTE DUAL-SIM ---")
    
    # 1. Verifica le matrici di propagazione (Il sospetto n. 1)
    print(f"Norma W0_T (Input TX): {torch.norm(model.W0_T).item():.6f}")
    print(f"Norma WL_T (Output TX): {torch.norm(model.WL_T).item():.6f}")
    
    # Controllo se le matrici interne sono vuote
    if len(model.W_int_T) > 0:
        norm_int = torch.norm(model.W_int_T[0]).item()
        print(f"Norma W_int_T[0]:      {norm_int:.6f}")
    
    # 2. Verifica i parametri ottimizzabili (Fasi)
    norm_xi = torch.norm(model.xi_T[0]).item()
    print(f"Norma Fasi TX (xi_T):   {norm_xi:.6f}")

    # 3. Test di propagazione (Z = GR * H * GT)
    with torch.no_grad():
        H_id = torch.eye(384, device=model.W0_T.device).to(torch.complex64)
        Z, _ = model.get_effective_cascade(H_id)
        norm_Z = torch.norm(Z).item()
        print(f"\n🌊 NORMA TOTALE CASCATA (Z): {norm_Z:.6f}")
        
    if norm_Z == 0:
        print("🚨 ERRORE CRITICO: La SIM è 'morta'. Il segnale non passa.")
        print("👉 Controlla il calcolo di 'slayer' e 'A_cell' in DualSIMoptimizer.")
    else:
        print("✅ La SIM è viva. Se la loss scende poco, è un problema di scala.")

# ============================================================
# 1. SETUP AMBIENTE E PERCORSI (Pre-flight Check)
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Percorsi assoluti per evitare ambiguità
BASE_DIR = Path('/Users/jacopocaldana/Desktop/Università/Tesi')
MODEL_PATH = BASE_DIR / 'models'

# Nomi encoder per costruire il percorso del checkpoint
TX_NAME = "vit_small_patch16_224" 
RX_NAME = "vit_base_patch16_224" 

# IL PERCORSO CHE ABBIAMO VERIFICATO INSIEME
CLF_PATH = MODEL_PATH / "data" / "classifiers" / "cifar10" / RX_NAME / f"seed_{SEED}.ckpt"

print(f"🔍 Verifica preliminare del classificatore...")
if not CLF_PATH.exists():
    raise FileNotFoundError(f"Errore critico: Il checkpoint non esiste in {CLF_PATH}. "
                            "Verifica la cartella prima di iniziare l'ottimizzazione!")
print("✅ Checkpoint trovato correttamente. Procedo con il caricamento dati...")

# ============================================================
# 2. CARICAMENTO DATI (DOPPIO DATAMODULE)
# ============================================================
dm_align = DataModuleAlignmentClassification(
    dataset="cifar10",
    tx_enc=TX_NAME,      
    rx_enc=RX_NAME,      
    train_label_size=100, 
    method='centroid',    
    batch_size=128,
    seed=SEED
)
dm_align.setup()

dm_task = DataModuleClassifier(
    dataset="cifar10",
    rx_enc=TX_NAME,     
    batch_size=128
)
dm_task.setup()

# ============================================================
# 3. CALCOLO DELLA MATRICE TARGET A (STRATEGIA)
# ============================================================

# Trasformazione complessa dei piloti
# input_c: da ViT-Small -> 384 reali -> 192 complessi
input_c = complex_compressed_tensor(dm_align.train_data.z_tx.T, device=device)
# output_c: da ViT-Base -> 768 reali -> 384 complessi
output_c = complex_compressed_tensor(dm_align.train_data.z_rx.T, device=device)

# --- Sbiancamento separato (Pre-whitening) ---
# Sbiancamento TX (Dimensione 192)
input_w, L_in, mu_in = prewhiten(input_c, device=device) 
# Sbiancamento RX (Dimensione 384)
output_w, L_out, mu_out = prewhiten(output_c, device=device)

# --- SCELTA DELLA STRATEGIA DI ALLINEAMENTO ---
ALIGNMENT_TYPE = 'Linear'  # Cambia in 'PPFE' per la modalità Zero-Shot

if ALIGNMENT_TYPE == 'Linear':
    # Ridge Regression (Supervisionata) per proiettare da 192 a 384
    A_target = ridge_regression(input_w, output_w, lmb=1e-3)
    print(f"✅ Target A calcolato con Ridge Regression (Supervisionato)")
else:
    # PPFE (Zero-Shot) basato su prototipi e clustering
    A_target = ppfe(
        input_w, output_w, 
        output_real=dm_align.train_data.z_rx, 
        n_clusters=10, n_proto=10, seed=SEED
    )
    print(f"✅ Target A calcolato con PPFE (Zero-Shot)")

# ============================================================
# CHECK PRE-OTTIMIZZAZIONE (Debug Dimensioni)
# ============================================================
print("\n--- 🕵️ DEBUG SANITY CHECK ---")
print(f"TX (Small) Complex Dim: {input_c.shape[0]} (Atteso: 192)")
print(f"RX (Base) Complex Dim: {output_c.shape[0]} (Atteso: 384)")
print(f"L_in Shape: {L_in.shape} (Atteso: [192, 192])")
print(f"L_out Shape: {L_out.shape} (Atteso: [384, 384])")
print(f"A_target Shape: {A_target.shape} (Atteso: [384, 192])")

# Verifica bloccante per prevenire il RuntimeError in inference.py
if L_in.shape[0] != 192:
    print(f" ERRORE: L_in ha dimensione {L_in.shape[0]}, ma l'input TX è 192.")
    exit()

if A_target.shape != (384, 192):
    print(f" ERRORE: A_target non è [384, 192]. Shape attuale: {A_target.shape}")
    exit()

print("✅ Dimensioni coerenti! Procedo con la fisica Dual-SIM...")

# ============================================================
# 4. INIZIALIZZAZIONE FISICA DUAL-SIM E CANALE
# ============================================================
wavelength = 0.005  # lambda = 5mm (60 GHz)
slayer = 5 * wavelength 
dx = wavelength / 2 

sim_cpu = DualSIMoptimizer(
    num_layers_TX=10, 
    num_meta_atoms_TX_in_x=16, num_meta_atoms_TX_in_y=12,    # 192 complessità semantica
    num_meta_atoms_TX_out_x=24, num_meta_atoms_TX_out_y=16,  # 384 antenne
    num_meta_atoms_TX_int_x=32, num_meta_atoms_TX_int_y=32,  
    thickness_TX=slayer * 10,
    
    num_layers_RX=10,
    num_meta_atoms_RX_in_x=24, num_meta_atoms_RX_in_y=16,    # 384 antenne
    num_meta_atoms_RX_out_x=24, num_meta_atoms_RX_out_y=16,  # 384 ricostruzione semantica
    num_meta_atoms_RX_int_x=32, num_meta_atoms_RX_int_y=32,
    thickness_RX=slayer * 10,
    
    wavelength=wavelength,
    spacings={'tx_in': dx, 'tx_out': dx, 'tx_int': dx, 'rx_in': dx, 'rx_out': dx, 'rx_int': dx},
    verbose=True # Vedrai il progresso del calcolo delle matrici W
)

model = DualSIMoptimizerTorch(sim_cpu).to(device)
#H_mimo = complex_gaussian_matrix(0, 1, size=(384, 384)).to(device)
H_mimo = torch.eye(384, 384, device=device).to(torch.complex64)

##############################
#####BLOCCO DIAGNOSI##############################
######################
inspect_sim_health(model)

with torch.no_grad():
    # 1. Calcola la norma del target e della SIM iniziale
    z_init, _ = model.get_effective_cascade(H_mimo)
    norm_target = torch.norm(A_target).item()
    norm_sim = torch.norm(z_init).item()
    
    print(f"\n--- DIAGNOSI INTOPO ---")
    print(f"Norma Target (Docente): {norm_target:.2f}")
    print(f"Norma SIM (Studente):  {norm_sim:.2f}")
    print(f"Rapporto: {norm_target/norm_sim:.2f}")
    
    if norm_target/norm_sim > 2.0:
        print("❌ SOSPETTO: Il target è troppo 'grande' per la SIM. La loss non scenderà mai.")
        # Prova a riscalare: A_target = A_target * (norm_sim / norm_target)

# ============================================================
# 5. OTTIMIZZAZIONE ALTERNATA (ALGORITMO 2)
# ============================================================
#print(f" Inizio ottimizzazione ({ALIGNMENT_TYPE})...")
loss_history = model.optimize_alternating(
    A_target=A_target, 
    H_mimo=H_mimo, 
    max_iters=500, 
    lr=0.1, 
    lambda_reg=1e-4
)

with torch.no_grad():
    Z_final, _ = model.get_effective_cascade(H_mimo)
    beta_opt = torch.sum(torch.conj(Z_final) * A_target) / torch.sum(torch.conj(Z_final) * Z_final)

# ============================================================
# 6. VALUTAZIONE FINALE (LOOP SNR: SIM vs MMSE)
# ============================================================
clf = Classifier.load_from_checkpoint(CLF_PATH).to(device).eval()

# Range di SNR [dB] da testare per la tesi
snr_range = [-10, 0, 10, 20, 30, 40]
results_sim = []
results_mmse = []

print(f"\n📊 Inizio valutazione comparativa ({ALIGNMENT_TYPE})...")

for snr in snr_range:
    # 1. Valutazione Dual-SIM 
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
    
    print(f"📡 SNR: {snr:3}dB | SIM: {acc_sim*100:6.2f}%")
# ============================================================
# 7. GRAFICI 
# ============================================================
plt.figure(figsize=(12, 5))

# Plot 1: Convergenza Loss
plt.subplot(1, 2, 1)
plt.plot(loss_history, color='green', lw=2)
plt.yscale('log')
plt.title('Convergenza Semantica')
plt.xlabel('Iterazioni'), plt.ylabel('Loss (Log)')
plt.grid(True, alpha=0.3)

# Plot 2: Accuracy vs SNR 
plt.subplot(1, 2, 2)
plt.plot(snr_range, results_sim, marker='o', linestyle='-', color='blue', lw=2)
plt.title(f'Performance Dual-SIM ({ALIGNMENT_TYPE})')
plt.xlabel('SNR [dB]'), plt.ylabel('Accuratezza [%]')
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(BASE_DIR / 'risultato_finale.png') # Salva il grafico automaticamente
plt.show()


oracle_acc = run_oracle_test(dm_task, A_target, L_in, mu_in, L_out, mu_out, clf, device)