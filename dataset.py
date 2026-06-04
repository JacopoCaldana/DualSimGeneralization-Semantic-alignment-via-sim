import torch
from torch.utils.data import Dataset, DataLoader

class SemanticAlignmentDataset(Dataset):
    def __init__(self, num_samples, num_pilots=128, gamma=1e-4):
        self.num_samples = num_samples
        self.theta_bar = 192  # 16 x 12
        self.N_T = 384        # 24 x 16
        self.N_R = 384        # 24 x 16
        self.omega_bar = 384  # 24 x 16
        
        self.H_data = []
        self.A_data = []
        
        for _ in range(num_samples):
            H = (torch.randn(self.N_R, self.N_T) + 1j * torch.randn(self.N_R, self.N_T)) / torch.sqrt(torch.tensor(2.0))
            self.H_data.append(H)
            
            X = torch.randn(self.theta_bar, num_pilots) + 1j * torch.randn(self.theta_bar, num_pilots)
            Y = torch.randn(self.omega_bar, num_pilots) + 1j * torch.randn(self.omega_bar, num_pilots)
            inv_term = torch.linalg.inv(X @ X.conj().T + gamma * torch.eye(self.theta_bar, dtype=torch.complex64))
            A = Y @ X.conj().T @ inv_term
            self.A_data.append(A)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.A_data[idx], self.H_data[idx]

# --- CHECK DIAGNOSTICO DEL DATASET ---
if __name__ == "__main__":
    print("🔍 --- DIAGNOSTICA DATASET ---")
    dataset = SemanticAlignmentDataset(num_samples=10)
    loader = DataLoader(dataset, batch_size=2)
    A_b, H_b = next(iter(loader))
    
    print(f"A_target Batch Shape: {A_b.shape} | Tipo: {A_b.dtype}")
    print(f"H_mimo Batch Shape:   {H_b.shape} | Tipo: {H_b.dtype}")
    print(f"Norma Frobenius Media A: {torch.norm(A_b[0], p='fro').item():.4f}")
    print(f"Norma Frobenius Media H: {torch.norm(H_b[0], p='fro').item():.4f}")
    
    # Controllo NaN o Inf
    if torch.isnan(A_b).any() or torch.isnan(H_b).any():
        print("❌ ERROR: Il dataset contiene dei NaN!")
    else:
        print("✅ DATASET: OK (Nessun NaN rilevato, forme coerenti)")