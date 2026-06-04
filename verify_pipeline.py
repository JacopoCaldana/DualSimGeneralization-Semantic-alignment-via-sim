# verify_pipeline.py
import torch
from model import DualSIMUnrolledTorch

def run_phase4_verification():
    print("=== FASE 4: VERIFICA CON CONFIGURAZIONE REALE AIR-SIM ===")

    # Parametri esatti dal tuo script analogico
    L = 3               # Numero di layer per SIM
    M_int = 12          # Esempio: num_meta_atoms_int_x/y
    
    N_T = 384           # 24 * 16 (Antenne TX)
    N_R = 384           # 24 * 16 (Antenne RX)
    theta_bar = 16      # Spazio latente TX complanare d'ingresso
    omega_bar = 16      # Spazio latente RX complanare d'uscita

    # Elementi interni totali per ciascun layer del SIM
    M_total_layer = M_int * M_int  # 12 * 12 = 144 textel per strato
    
    # Dimensione piatta totale per la rete neurale (144 * 3 = 432 elementi di fase)
    d_1 = L * M_total_layer
    d_2 = L * M_total_layer

    print(f"Dimensioni Canale Configurate: {N_R}x{N_T}")
    print(f"Elementi di fase totali srotolati per SIM (d_1, d_2): {d_1}")

    # Generazione delle matrici fisse imitando quelle estratte dall'oggetto DualSIMoptimizer
    # TX (G1): Connette l'ingresso latente (16) alle antenne (384) passando per i layer interni (144)
    W_list_1 = [
        torch.randn(M_total_layer, theta_bar, dtype=torch.complex64) / torch.sqrt(torch.tensor(theta_bar)),
        torch.randn(M_total_layer, M_total_layer, dtype=torch.complex64) / torch.sqrt(torch.tensor(M_total_layer)),
        torch.randn(M_total_layer, M_total_layer, dtype=torch.complex64) / torch.sqrt(torch.tensor(M_total_layer)),
        torch.randn(N_T, M_total_layer, dtype=torch.complex64) / torch.sqrt(torch.tensor(M_total_layer))
    ]

    # RX (G2): Connette le antenne (384) allo spazio latente d'uscita (16)
    W_list_2 = [
        torch.randn(M_total_layer, N_R, dtype=torch.complex64) / torch.sqrt(torch.tensor(N_R)),
        torch.randn(M_total_layer, M_total_layer, dtype=torch.complex64) / torch.sqrt(torch.tensor(M_total_layer)),
        torch.randn(M_total_layer, M_total_layer, dtype=torch.complex64) / torch.sqrt(torch.tensor(M_total_layer)),
        torch.randn(omega_bar, M_total_layer, dtype=torch.complex64) / torch.sqrt(torch.tensor(M_total_layer))
    ]

    # Inizializziamo la rete con K=4 strati srotolati
    net = DualSIMUnrolledTorch(num_layers=4, d_1=d_1, d_2=d_2, W_list_1=W_list_1, W_list_2=W_list_2)

    # Istanza reale del Canale H (384 x 384) e del Target A (16 x 16)
    H_mimo = (torch.randn(N_R, N_T, dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0)))
    A_target = torch.randn(omega_bar, theta_bar, dtype=torch.complex64)

    # Vettori iniziali piatti delle fasi (432,)
    xi_1_init = torch.rand(d_1) * 2 * torch.pi
    xi_2_init = torch.rand(d_2) * 2 * torch.pi

    print("Esecuzione del forward pass...")
    try:
        xi_1_opt, xi_2_opt, beta = net(A_target, H_mimo, xi_1_init, xi_2_init, layers=L, M_int=M_int)
        print("\n=== VERIFICA PIPELINE EMULAZIONE STRUTTURATA RIUSCITA! ===")
        print(f"Fasi TX ottimizzate (flat): {xi_1_opt.shape}")
        print(f"Fasi RX ottimizzate (flat): {xi_2_opt.shape}")
        print(f"Valore beta finale: {beta.item():.4f}")
    except Exception as e:
        print(f"\n[ERRORE]: {e}")

if __name__ == "__main__":
    run_phase4_verification()