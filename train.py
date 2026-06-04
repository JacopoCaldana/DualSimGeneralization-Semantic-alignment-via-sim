# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import math

# Importiamo i moduli dell'architettura Unrolled (Fase 1 e Fase 3)
from model import DualSIMUnrolledTorch
from physics_engine import get_sim_transfer_matrix
from data_utils import SIMDataset


from dualsim import DualSIMoptimizer,DualSIMoptimizerTorch
from utils import complex_gaussian_matrix




def train_unrolled_dualsim():
    print("=== FASE 5: ADDESTRAMENTO DEEP UNROLLING (PARAMETRI ORIGINARI SIM) ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device rilevato ed in uso: {device}")

    # 1. Configurazione Fisica REALE presa dal tuo codice
    L_T = 5             # Corrisponde a L per il TX
    L_R = 5             # Corrisponde a L per l'RX
    M_int = 16          
    wavelength = 0.005  
    slayer = 5 * wavelength 
    dx = wavelength / 2  
    
    # PARAMETRI ALLINEATI ALLA TUA FUNZIONE:
    # TX in: 16x12 = 192  |  TX out: 24x16 = 384 (Antenne N_T)
    # RX in: 24x16 = 384 (Antenne N_R)  |  RX out: 24x16 = 384 
    theta_bar = 16 * 12 # = 192 (Dimensione spaziale d'ingresso)
    omega_bar = 24 * 16 # = 384 (Dimensione spaziale d'uscita)
    N_T = 24 * 16       # = 384 Antenne TX
    N_R = 24 * 16       # = 384 Antenne RX

    # Iperparametri di addestramento della rete
    K_layers = 4         
    num_epochs = 10      
    batch_size = 4       
    learning_rate = 1e-3

    print("\nInizializzazione DualSIMoptimizer con le tue griglie reali...")
    sim_cpu = DualSIMoptimizer(
        num_layers_TX=L_T, 
        num_meta_atoms_TX_in_x=16,  num_meta_atoms_TX_in_y=12,   # 16x12 = 192 (theta_bar)
        num_meta_atoms_TX_out_x=24, num_meta_atoms_TX_out_y=16,  # 24x16 = 384 (N_T)
        num_meta_atoms_TX_int_x=M_int, num_meta_atoms_TX_int_y=M_int,  
        thickness_TX=slayer * L_T,
        num_layers_RX=L_R,
        num_meta_atoms_RX_in_x=24, num_meta_atoms_RX_in_y=16,  # 24x16 = 384 (N_R)
        num_meta_atoms_RX_out_x=24, num_meta_atoms_RX_out_y=16, # 24x16 = 384 (omega_bar)
        num_meta_atoms_RX_int_x=M_int, num_meta_atoms_RX_int_y=M_int,
        thickness_RX=slayer * L_R,
        wavelength=wavelength,
        spacings={'tx_in': dx, 'tx_out': dx, 'tx_int': dx, 'rx_in': dx, 'rx_out': dx, 'rx_int': dx},
        verbose=False 
    )

    print("Conversione in DualSIMoptimizerTorch...")
    sim_torch = DualSIMoptimizerTorch(sim_cpu).to(device)

    # Estrazione dei buffer
    W_list_1 = sim_torch._get_W_list('W_T', sim_torch.L_T)
    W_list_2 = sim_torch._get_W_list('W_R', sim_torch.L_R)

    # Shape dei singoli layer per lo slicing dinamico
    layer_sizes_1 = [p.numel() for p in sim_torch.xi_T]
    layer_sizes_2 = [p.numel() for p in sim_torch.xi_R]
    
    d_1 = sum(layer_sizes_1)
    d_2 = sum(layer_sizes_2)
    print(f" -> Dimensione totale delle fasi TX (d_1): {d_1} | RX (d_2): {d_2}")
    print(f" -> Ripartizione layer TX: {layer_sizes_1} | RX: {layer_sizes_2}")

    # Generazione del dataset con le dimensioni semantiche allineate (384 x 192)
    print(f"\nGenerazione del dataset con target A di dimensione {omega_bar}x{theta_bar}...")
    dataset = SIMDataset(num_samples=200, N_T=N_T, N_R=N_R, theta_bar=theta_bar, omega_bar=omega_bar)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Istanziazione della Rete Deep Unrolled
    net = DualSIMUnrolledTorch(
        num_layers=K_layers, d_1=d_1, d_2=d_2, 
        W_list_1=W_list_1, W_list_2=W_list_2,
        layer_sizes_1=layer_sizes_1, layer_sizes_2=layer_sizes_2
    ).to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Loop di Addestramento
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (A_batch, H_batch) in enumerate(dataloader):
            A_batch = A_batch.to(device)
            H_batch = H_batch.to(device)
            
            optimizer.zero_grad()
            batch_loss = 0.0
            current_batch_size = A_batch.size(0)
            
            for i in range(current_batch_size):
                A_i = A_batch[i]
                H_i = H_batch[i]
                
                # Inizializziamo lo stato a zero (inizializzazione trasparente)
                xi_1_init = (torch.zeros(d_1)).to(device)
                xi_2_init = (torch.zeros(d_2)).to(device)
                
                # Forward Pass srotolato della rete neurale
                xi_1_opt, xi_2_opt, beta_opt = net(A_i, H_i, xi_1_init, xi_2_init)
                
                # Calcolo della cascata fisica per l'errore di emulazione
                G1 = get_sim_transfer_matrix(xi_1_opt, W_list_1, layer_sizes_1)
                G2 = get_sim_transfer_matrix(xi_2_opt, W_list_2, layer_sizes_2)
                Z_final = G2 @ H_i @ G1
                
                # Loss di allineamento (Equation 43 con lambda=0)
                loss_i = torch.norm(beta_opt * Z_final - A_i, p='fro')**2
                batch_loss += loss_i
                
            batch_loss = batch_loss / current_batch_size
            batch_loss.backward()
            # NUOVO: Taglia i gradienti troppo alti per stabilizzare l'architettura unrolled
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}] | Average Frobenius Loss: {epoch_loss / len(dataloader):.5f}")

    print("\n=== ADDESTRAMENTO COMPLETATO CON SUCCESSO ===")
    torch.save(net.state_dict(), "dualsim_unrolled_weights.pth")
    print("Pesi salvati in 'dualsim_unrolled_weights.pth'")

if __name__ == "__main__":
    train_unrolled_dualsim()