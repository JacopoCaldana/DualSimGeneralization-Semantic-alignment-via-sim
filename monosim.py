import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim

import numpy as np

class MonoSIMoptimizer:
    """
    Single Stacked Intelligent Metasurface (SIM) Optimizer.
    Can be instantiated as either TX or RX.
    """
    def __init__(
        self,
        num_layers: int,
        num_meta_atoms_in_x: int,
        num_meta_atoms_in_y: int,
        num_meta_atoms_out_x: int,
        num_meta_atoms_out_y: int,
        num_meta_atoms_int_x: int,
        num_meta_atoms_int_y: int,
        thickness: float,
        wavelength: float,
        verbose: bool = False,
        spacings: dict = None 
    ):
        if num_layers <= 0:
            raise ValueError("Number of layers must be positive.")

        self.L = num_layers
        self.N_in = num_meta_atoms_in_x * num_meta_atoms_in_y
        self.N_out = num_meta_atoms_out_x * num_meta_atoms_out_y
        self.M_int = num_meta_atoms_int_x * num_meta_atoms_int_y
        self.slayer = thickness / num_layers

        self.wavelength = wavelength
        self.kappa = 2 * np.pi / wavelength
        self.verbose = verbose
        
        # Spaziature uniformate per la singola SIM
        self.spacings = spacings or {
            'in': 0.5 * wavelength, 'out': 0.5 * wavelength, 'int': 0.5 * wavelength
        }

        # Inizializzazione fasi (deviazione standard piccola)
        std_dev = 0.01 
        self._phase_shifts = [
            np.random.normal(0, std_dev, self.M_int) for _ in range(self.L - 1)
        ] + [np.random.normal(0, std_dev, self.N_out)]

        # Pre-calcolo matrici di propagazione W
        self.W = self._build_W_list(self.L)

    def _build_W_list(self, L: int) -> list:
        W_list = []
        if L == 1:
            W_list.append(self._calculate_W('input', 'output'))
        else:
            W_list.append(self._calculate_W('input', 'intermediate'))
            for _ in range(L - 2):
                W_list.append(self._calculate_W('intermediate', 'intermediate'))
            W_list.append(self._calculate_W('intermediate', 'output'))
        return W_list

    def _get_2d_coords(self, linear_idx: int, num_atoms_x: int) -> tuple[int, int]:
        my = int(np.ceil(linear_idx / num_atoms_x))
        mx = linear_idx - (my - 1) * num_atoms_x
        return mx, my
    
    def _calculate_propagation_distance(self, nx_from, idx_from, nx_to, idx_to, sx, sy, slayer) -> float:
        x_from, y_from = self._get_2d_coords(idx_from, nx_from)
        x_to, y_to = self._get_2d_coords(idx_to, nx_to)
        return np.sqrt(((x_from - x_to)*sx)**2 + ((y_from - y_to)*sy)**2 + slayer**2)
    
    def _calculate_W(self, from_type: str, to_type: str) -> np.ndarray:
        params = {
            'input':        (self.N_in,  int(np.sqrt(self.N_in)),  self.spacings['in']),
            'intermediate': (self.M_int, int(np.sqrt(self.M_int)), self.spacings['int']),
            'output':       (self.N_out, int(np.sqrt(self.N_out)), self.spacings['out']),
        }

        from_total, from_nx, from_sx = params[from_type]
        to_total, to_nx, _ = params[to_type]
        A_cell = from_sx * from_sx 
        W = np.zeros((to_total, from_total), dtype=complex)

        for to_idx in range(1, to_total + 1):
            for from_idx in range(1, from_total + 1):
                d = self._calculate_propagation_distance(from_nx, from_idx, to_nx, to_idx, from_sx, from_sx, self.slayer)
                if d > 0:
                    W[to_idx - 1, from_idx - 1] = (self.slayer * A_cell / (d**2)) * \
                                                  (1 / (2 * np.pi * d) - 1j / self.wavelength) * \
                                                  np.exp(1j * self.kappa * d)
        return W





class MonoSIMoptimizerTorch(nn.Module):
    def __init__(self, sim_cpu):
        super().__init__()
        self.verbose = sim_cpu.verbose
        self.L = sim_cpu.L
        self.input_dim = sim_cpu.N_in

        # Registrazione buffer per la GPU
        for i, W in enumerate(sim_cpu.W):
            self.register_buffer(f'W_{i}', torch.tensor(W, dtype=torch.complex64))

        # Parametri addestrabili: fasi xi
        self.xi = nn.ParameterList([
            nn.Parameter(torch.tensor(p, dtype=torch.float32)) for p in sim_cpu._phase_shifts
        ])

    def get_cascade(self):
        device = self.xi[0].device
        G = torch.eye(self.input_dim, dtype=torch.complex64, device=device)
        for l in range(self.L):
            W_l = getattr(self, f'W_{l}')
            Y_l = torch.diag(torch.exp(1j * self.xi[l]))
            G = Y_l @ W_l @ G
        return G

    def optimize_sim(self, target_matrix, max_iters=5000, lr=0.05, lambda_reg=0.0):
        """
        Ottimizzazione TX usando Adam per gestire la profondità dei layer.
        """
        # Fix per il warning PyTorch
        if isinstance(target_matrix, torch.Tensor):
            Target_torch = target_matrix.clone().detach().to(self.xi[0].device).to(torch.complex64)
        else:
            Target_torch = torch.tensor(target_matrix, dtype=torch.complex64, device=self.xi[0].device)
        
        opt = optim.Adam(self.xi.parameters(), lr=lr)
        loss_history = []

        for k in range(max_iters):
            with torch.no_grad():
                G_k = self.get_cascade()
                num_beta = torch.sum(torch.conj(G_k) * Target_torch)
                den_beta = torch.sum(torch.conj(G_k) * G_k) + 1e-12
                beta_k = num_beta / den_beta

            opt.zero_grad()
            G_current = self.get_cascade()
            
            loss = torch.norm(beta_k * G_current - Target_torch, p='fro')**2
            if lambda_reg > 0:
                loss += lambda_reg * torch.norm(G_current, p='fro')**2
                
            loss.backward()
            opt.step() # NESSUNA normalizzazione manuale, lasciamo lavorare Adam

            with torch.no_grad():
                for p in self.xi: 
                    p.copy_(p % (2 * torch.pi))

            current_loss = loss.item()
            loss_history.append(current_loss)
            
            if k % 500 == 0:
                print(f"      [Iter {k:4d}/{max_iters}] Loss: {current_loss:.4e} | Beta Mag: {torch.abs(beta_k).item():.2e}")
            
        with torch.no_grad():
            G_final = self.get_cascade()
            beta_f = torch.sum(torch.conj(G_final) * Target_torch) / (torch.sum(torch.conj(G_final) * G_final) + 1e-12)
            fro_err = torch.norm(beta_f * G_final - Target_torch) / (torch.norm(Target_torch) + 1e-12)
            print(f"   🏁 Optimization Done. Final Relative Error: {fro_err.item():.4f}\n")
            
        return loss_history, beta_f.item()


    def optimize_rx_sequential(self, H_eff, A_target, max_iters=5000, lr=0.05, lambda_reg=1e-4):
        """
        Ottimizzazione Sequenziale RX usando Adam. Essenziale per L > 10.
        """
        # Fix per il warning PyTorch
        if isinstance(H_eff, torch.Tensor):
            H_eff_torch = H_eff.clone().detach().to(self.xi[0].device).to(torch.complex64)
        else:
            H_eff_torch = torch.tensor(H_eff, dtype=torch.complex64, device=self.xi[0].device)
            
        if isinstance(A_target, torch.Tensor):
            A_torch = A_target.clone().detach().to(self.xi[0].device).to(torch.complex64)
        else:
            A_torch = torch.tensor(A_target, dtype=torch.complex64, device=self.xi[0].device)

        opt = optim.Adam(self.xi.parameters(), lr=lr)
        loss_history = []

        for k in range(max_iters):
            with torch.no_grad():
                G_R = self.get_cascade()
                Z = G_R @ H_eff_torch
                num_beta = torch.sum(torch.conj(Z) * A_torch)
                den_beta = torch.sum(torch.conj(Z) * Z) + 1e-12
                beta_k = num_beta / den_beta

            opt.zero_grad()
            G_R = self.get_cascade()
            Z_current = G_R @ H_eff_torch
            
            loss = torch.norm(beta_k * Z_current - A_torch, p='fro')**2
            if lambda_reg > 0:
                loss += lambda_reg * torch.norm(G_R, p='fro')**2

            loss.backward()
            opt.step() # Adam fa la magia qui

            with torch.no_grad():
                for p in self.xi:
                    p.copy_(p % (2 * torch.pi))

            current_loss = loss.item()
            loss_history.append(current_loss)

            if k % 500 == 0:
                print(f"      [Iter {k:4d}/{max_iters}] Eq. Loss: {current_loss:.4e} | Beta Mag: {torch.abs(beta_k).item():.2e}")

        return loss_history, beta_k.item()