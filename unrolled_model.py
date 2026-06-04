import torch
import torch.nn as nn

class DualSIMUnrolledTorch(nn.Module):
    def __init__(self, d_T, d_R, K_layers, init_gamma=0.1):
        super().__init__()
        self.K = K_layers
        self.d_T = d_T
        self.d_R = d_R

        self.S_T = nn.ParameterList([nn.Parameter(torch.ones(d_T)) for _ in range(self.K)])
        self.W_T = nn.ParameterList([nn.Parameter(torch.ones(d_T) * init_gamma) for _ in range(self.K)])
        self.S_R = nn.ParameterList([nn.Parameter(torch.ones(d_R)) for _ in range(self.K)])
        self.W_R = nn.ParameterList([nn.Parameter(torch.ones(d_R) * init_gamma) for _ in range(self.K)])

    def _compute_vjp(self, z_vec, xi, v):
        dot_product = torch.vdot(z_vec, v).real
        if not dot_product.requires_grad:
            return torch.zeros_like(xi)
        return torch.autograd.grad(dot_product, xi, create_graph=True)[0]

    def forward(self, xi_T_flat, xi_R_flat, H, A, forward_cascade_fn):
        a = A.flatten()

        xi_T_flat = xi_T_flat.clone().detach().requires_grad_(True)
        xi_R_flat = xi_R_flat.clone().detach().requires_grad_(True)

        for k in range(self.K):
            xi_T_flat = xi_T_flat.clone().requires_grad_(True)
            xi_R_flat = xi_R_flat.clone().requires_grad_(True)

            # --- STEP TX ---
            Z_k = forward_cascade_fn(xi_T_flat, xi_R_flat, H)
            z_k = Z_k.flatten()
            
            beta_k = torch.vdot(z_k, a) / (torch.norm(z_k)**2 + 1e-12)
            r_k = a - beta_k * z_k
            v_T = beta_k.conj() * r_k
            
            g_T_k = self._compute_vjp(z_k, xi_T_flat, v_T.detach())
            g_T_k = g_T_k / (torch.norm(g_T_k) + 1e-8)  # Normalizzazione del VJP
            
            xi_T_next = self.S_T[k] * xi_T_flat + self.W_T[k] * g_T_k

            # --- STEP RX ---
            Z_k_half = forward_cascade_fn(xi_T_next, xi_R_flat, H)
            z_k_half = Z_k_half.flatten()
            r_k_half = a - beta_k * z_k_half
            
            v_R = beta_k.conj() * r_k_half
            g_R_k = self._compute_vjp(z_k_half, xi_R_flat, v_R.detach())
            g_R_k = g_R_k / (torch.norm(g_R_k) + 1e-8)  # Normalizzazione del VJP
            
            xi_R_next = self.S_R[k] * xi_R_flat + self.W_R[k] * g_R_k

            xi_T_flat, xi_R_flat = xi_T_next, xi_R_next

        Z_final = forward_cascade_fn(xi_T_flat, xi_R_flat, H)
        beta_final = torch.vdot(Z_final.flatten(), a) / (torch.norm(Z_final)**2 + 1e-12)
        
        return xi_T_flat, xi_R_flat, Z_final, beta_final