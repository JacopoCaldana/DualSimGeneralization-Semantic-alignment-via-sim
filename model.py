# model.py
import torch
import torch.nn as nn
import math
from physics_engine import get_sim_transfer_matrix

# model.py
import torch
import torch.nn as nn
from physics_engine import get_sim_transfer_matrix

class DualSIMUnrolledTorch(nn.Module):
    def __init__(self, num_layers, d_1, d_2, W_list_1, W_list_2, layer_sizes_1, layer_sizes_2):
        super(DualSIMUnrolledTorch, self).__init__()
        self.num_layers = num_layers
        self.d_1 = d_1
        self.d_2 = d_2
        self.W_list_1 = W_list_1
        self.W_list_2 = W_list_2
        self.layer_sizes_1 = layer_sizes_1
        self.layer_sizes_2 = layer_sizes_2
        
        # Inizializzazione bilanciata per facilitare la discesa del gradiente
        self.S_1 = nn.ParameterList([nn.Parameter(torch.eye(d_1) * 0.5) for _ in range(num_layers)])
        self.W_1 = nn.ParameterList([nn.Parameter(torch.eye(d_1) * 0.1) for _ in range(num_layers)])
        
        self.S_2 = nn.ParameterList([nn.Parameter(torch.eye(d_2) * 0.5) for _ in range(num_layers)])
        self.W_2 = nn.ParameterList([nn.Parameter(torch.eye(d_2) * 0.1) for _ in range(num_layers)])

    def forward(self, A, H, xi_1_init, xi_2_init):
        # Abilitiamo l'autograd preservando la connessione al grafo radice (Nessun detach)
        xi_1 = xi_1_init.flatten().clone().requires_grad_(True)
        xi_2 = xi_2_init.flatten().clone().requires_grad_(True)
        
        for k in range(self.num_layers):
            # --- STEP 1: OTTIMIZZAZIONE TX (G1) ---
            G1 = get_sim_transfer_matrix(xi_1, self.W_list_1, self.layer_sizes_1)
            G2 = get_sim_transfer_matrix(xi_2, self.W_list_2, self.layer_sizes_2)
            Z = G2 @ H @ G1
            
            # Proiezione geometrica di beta (Emula il comportamento analogico con no_grad)
            with torch.no_grad():
                numerator = torch.sum(torch.conj(Z) * A)
                denominator = torch.sum(torch.conj(Z) * Z) + 1e-12
                beta = numerator / denominator
            
            loss_1 = torch.norm(beta * Z - A, p='fro')**2
            g_1 = torch.autograd.grad(loss_1, xi_1, create_graph=True)[0].real
            
            # Normalizzazione del gradiente (Previene l'evanescenza all'inizializzazione)
            g_1 = g_1 / (torch.norm(g_1) + 1e-8)
            
            xi_1_next = torch.matmul(self.S_1[k], xi_1) - torch.matmul(self.W_1[k], g_1)
            xi_1 = torch.remainder(xi_1_next, 2.0 * torch.pi)
            
            # --- STEP 2: OTTIMIZZAZIONE RX (G2) ---
            G1_updated = get_sim_transfer_matrix(xi_1, self.W_list_1, self.layer_sizes_1)
            G2_current = get_sim_transfer_matrix(xi_2, self.W_list_2, self.layer_sizes_2)
            Z_half = G2_current @ H @ G1_updated
            
            with torch.no_grad():
                numerator_half = torch.sum(torch.conj(Z_half) * A)
                denominator_half = torch.sum(torch.conj(Z_half) * Z_half) + 1e-12
                beta_half = numerator_half / denominator_half
            
            loss_2 = torch.norm(beta_half * Z_half - A, p='fro')**2
            g_2 = torch.autograd.grad(loss_2, xi_2, create_graph=True)[0].real
            
            g_2 = g_2 / (torch.norm(g_2) + 1e-8)
            
            xi_2_next = torch.matmul(self.S_2[k], xi_2) - torch.matmul(self.W_2[k], g_2)
            xi_2 = torch.remainder(xi_2_next, 2.0 * torch.pi)
            
        return xi_1, xi_2, beta_half