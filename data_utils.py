import torch
from torch.utils.data import Dataset, DataLoader

class SIMDataset(Dataset):
    def __init__(self, num_samples, N_T, N_R, theta_bar, omega_bar):
        """
        Dataset sintetico per l'addestramento della rete Dual-SIM.
        
        Args:
            num_samples (int): Numero totale di campioni nel dataset.
            N_T (int): Numero di antenne in trasmissione.
            N_R (int): Numero di antenne in ricezione.
            theta_bar (int): Dimensione dello spazio latente compresso al TX.
            omega_bar (int): Dimensione dello spazio latente compresso all'RX.
        """
        self.num_samples = num_samples
        
        # Generiamo i canali fisici H (MIMO Rayleigh fading channel)
        # Distribuzione Gaussiana complessa standard N(0, 1) per ogni elemento
        real_part_H = torch.randn(num_samples, N_R, N_T)
        imag_part_H = torch.randn(num_samples, N_R, N_T)
        self.H_data = torch.complex(real_part_H, imag_part_H) / torch.sqrt(torch.tensor(2.0))
        
        # Generiamo gli operatori semantici target A
        # In uno scenario reale verrebbero calcolati tramite i Semantic Pilots o Frame di Parseval
        # Dentro la generazione di A in SIMDataset (data_utils.py)
        real_part_A = torch.randn(num_samples, omega_bar, theta_bar)
        imag_part_A = torch.randn(num_samples, omega_bar, theta_bar)
        A_unnormalized = torch.complex(real_part_A, imag_part_A)

# Normalizzazione per campione (Frobenius norm = 1)
        self.A_data = torch.zeros_like(A_unnormalized)
        for i in range(num_samples):
            self.A_data[i] = A_unnormalized[i] / torch.norm(A_unnormalized[i], p='fro')


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.A_data[idx], self.H_data[idx]

# --- Esempio di Inizializzazione ---
# Definiamo le dimensioni del sistema
N_T = 64         # Antenne TX
N_R = 64         # Antenne RX
theta_bar = 16   # Dimensione latente complessa TX
omega_bar = 16   # Dimensione latente complessa RX

# Creiamo il dataset sintetico di 1000 campioni
train_dataset = SIMDataset(num_samples=1000, N_T=N_T, N_R=N_R, theta_bar=theta_bar, omega_bar=omega_bar)

# Creiamo il DataLoader per iterare a blocchi (batch)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Verifica rapida del primo batch
A_batch, H_batch = next(iter(train_loader))
print(f"Shape del batch H: {H_batch.shape} (Dovrebbe essere {batch_size}, {N_R}, {N_T})")
print(f"Shape del batch A: {A_batch.shape} (Dovrebbe essere {batch_size}, {omega_bar}, {theta_bar})")
print(f"Tipo di dato: {H_batch.dtype}")