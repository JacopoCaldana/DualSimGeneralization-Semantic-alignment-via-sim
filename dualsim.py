import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim

class DualSIMoptimizer:
    """
    Dual Stacked Intelligent Metasurface (Dual-SIM) Optimizer.
    Implements the architecture with SIMs at both the transmitter (TX) and the receiver (RX)
    for over-the-air semantic alignment.
    """

    def __init__(
        self,
        # --- TX-SIM Parameters (L_T layers) ---
        num_layers_TX: int,                 # L_T 
        num_meta_atoms_TX_in_x: int,        # Input semantic dimension (theta_bar) 
        num_meta_atoms_TX_in_y: int,
        num_meta_atoms_TX_out_x: int,       # Number of TX antennas (N_T) 
        num_meta_atoms_TX_out_y: int,
        num_meta_atoms_TX_int_x: int,       # Meta-atoms in TX intermediate layers
        num_meta_atoms_TX_int_y: int,
        thickness_TX: float,

        # --- RX-SIM Parameters (L_R layers) ---
        num_layers_RX: int,                 # L_R 
        num_meta_atoms_RX_in_x: int,        # Number of RX antennas (N_R) 
        num_meta_atoms_RX_in_y: int,
        num_meta_atoms_RX_out_x: int,       # Output semantic dimension (omega_bar)
        num_meta_atoms_RX_out_y: int,
        num_meta_atoms_RX_int_x: int,       # Meta-atoms in RX intermediate layers
        num_meta_atoms_RX_int_y: int,
        thickness_RX: float,

        # --- Common Physical Parameters ---
        wavelength: float,                  # lambda 
        verbose: bool = False,
        # Spacings (dx, dy, sx, sy) - can be passed as a dictionary
        spacings: dict = None 
    ):
        # Validate thickness and layer counts
        if num_layers_TX <= 0 or num_layers_RX <= 0:
            raise ValueError("Number of layers must be positive for both SIMs.")

        # 1. TX-SIM Configuration
        self.L_T = num_layers_TX
        self.N_in_T = num_meta_atoms_TX_in_x * num_meta_atoms_TX_in_y    # theta_bar 
        self.N_out_T = num_meta_atoms_TX_out_x * num_meta_atoms_TX_out_y  # N_T 
        self.M_int_T = num_meta_atoms_TX_int_x * num_meta_atoms_TX_int_y
        self.slayer_TX = thickness_TX / num_layers_TX # Spacing between TX layers 

        # 2. RX-SIM Configuration
        self.L_R = num_layers_RX
        self.N_in_R = num_meta_atoms_RX_in_x * num_meta_atoms_RX_in_y    # N_R 
        self.N_out_R = num_meta_atoms_RX_out_x * num_meta_atoms_RX_out_y  # omega_bar 
        self.M_int_R = num_meta_atoms_RX_int_x * num_meta_atoms_RX_int_y
        self.slayer_RX = thickness_RX / num_layers_RX # Spacing between RX layers 

        # 3. EM Parameters
        self.wavelength = wavelength
        self.kappa = 2 * np.pi / wavelength # Wavenumber 
        self.verbose = verbose
        
        # Save spacings (using default values if not provided)
        self.spacings = spacings or {
            'tx_in': 0.5 * wavelength, 'tx_out': 0.5 * wavelength, 'tx_int': 0.5 * wavelength,
            'rx_in': 0.5 * wavelength, 'rx_out': 0.5 * wavelength, 'rx_int': 0.5 * wavelength
        }

        # 4. Phase Shift Initialization (xi_T and xi_R) 
        # The first L-1 layers have intermediate dimensions, the last layer has output dimensions
        #self._phase_shifts_TX = [
         #   np.random.uniform(0, 2 * np.pi, self.M_int_T) for _ in range(self.L_T - 1)
        #] + [np.random.uniform(0, 2 * np.pi, self.N_out_T)]

        #self._phase_shifts_RX = [
         #   np.random.uniform(0, 2 * np.pi, self.M_int_R) for _ in range(self.L_R - 1)
        #] + [np.random.uniform(0, 2 * np.pi, self.N_out_R)]

        # 4. Phase Shift Initialization (xi_T and xi_R)
        # INIZIALIZZAZIONE CORRETTA: Fasi vicine allo zero per evitare interferenza distruttiva massiva.
        # Usa una deviazione standard molto piccola (es. 0.01 radianti).
        std_dev = 0.01 
        
        self._phase_shifts_TX = [
            np.random.normal(0, std_dev, self.M_int_T) for _ in range(self.L_T - 1)
        ] + [np.random.normal(0, std_dev, self.N_out_T)]

        self._phase_shifts_RX = [
            np.random.normal(0, std_dev, self.M_int_R) for _ in range(self.L_R - 1)
        ] + [np.random.normal(0, std_dev, self.N_out_R)]

        # 5. Pre-calculation of Fixed Attenuation Matrices (W) 
        # Create exactly L matrices to satisfy the product \prod_{l=1}^{L}
        self.W_TX = self._build_W_list(self.L_T, 'TX_input', 'TX_intermediate', 'TX_output')
        self.W_RX = self._build_W_list(self.L_R, 'RX_input', 'RX_intermediate', 'RX_output')

    def _build_W_list(self, L: int, type_in: str, type_int: str, type_out: str) -> list:
        """Constructs the exact sequence of L propagation matrices."""
        W_list = []
        if L == 1:
            W_list.append(self._calculate_W(type_in, type_out))
        else:
            W_list.append(self._calculate_W(type_in, type_int))
            for _ in range(L - 2):
                W_list.append(self._calculate_W(type_int, type_int))
            W_list.append(self._calculate_W(type_int, type_out))
        return W_list

    def _get_2d_coords(self, linear_idx: int, num_atoms_x: int) -> tuple[int, int]:
        """Convert 1-indexed linear index to 1-indexed (x, y) coordinates."""
        my = int(np.ceil(linear_idx / num_atoms_x))
        mx = linear_idx - (my - 1) * num_atoms_x
        return mx, my
    
    def _calculate_propagation_distance(
        self,
        from_layer_num_atoms_x: int,
        from_linear_idx: int,
        to_layer_num_atoms_x: int,
        to_linear_idx: int,
        spacing_x: float,
        spacing_y: float,
        slayer: float 
    ) -> float:
        """Calculates propagation distance d_m,m' between two meta-atoms."""
        from_x, from_y = self._get_2d_coords(from_linear_idx, from_layer_num_atoms_x)
        to_x, to_y = self._get_2d_coords(to_linear_idx, to_layer_num_atoms_x)

        dx_diff = (from_x - to_x) * spacing_x
        dy_diff = (from_y - to_y) * spacing_y

        distance = np.sqrt(dx_diff**2 + dy_diff**2 + slayer**2)
        return distance
    
    def _calculate_W(self, from_layer_type: str, to_layer_type: str) -> np.ndarray:
        """Calculates the attenuation matrix W for a pair of layers (Eq. 3 or 5)."""
        params = {
            'TX_input':        (self.N_in_T,  int(np.sqrt(self.N_in_T)),  self.spacings['tx_in'],  self.slayer_TX),
            'TX_intermediate': (self.M_int_T, int(np.sqrt(self.M_int_T)), self.spacings['tx_int'], self.slayer_TX),
            'TX_output':       (self.N_out_T, int(np.sqrt(self.N_out_T)), self.spacings['tx_out'], self.slayer_TX),
            'RX_input':        (self.N_in_R,  int(np.sqrt(self.N_in_R)),  self.spacings['rx_in'],  self.slayer_RX),
            'RX_intermediate': (self.M_int_R, int(np.sqrt(self.M_int_R)), self.spacings['rx_int'], self.slayer_RX),
            'RX_output':       (self.N_out_R, int(np.sqrt(self.N_out_R)), self.spacings['rx_out'], self.slayer_RX),
        }

        from_total, from_nx, from_sx, slayer = params[from_layer_type]
        to_total, to_nx, _, _ = params[to_layer_type]
        
        A_cell = from_sx * from_sx 
        W = np.zeros((to_total, from_total), dtype=complex)

        for to_idx in range(1, to_total + 1):
            for from_idx in range(1, from_total + 1):
                d = self._calculate_propagation_distance(
                    from_nx, from_idx, to_nx, to_idx, from_sx, from_sx, slayer
                )
                
                if d > 0:
                    attenuation = (slayer * A_cell / (d**2)) * \
                                  (1 / (2 * np.pi * d) - 1j / self.wavelength) * \
                                  np.exp(1j * self.kappa * d)
                    W[to_idx - 1, from_idx - 1] = attenuation
        return W
    
    def _calculate_G_T(self) -> np.ndarray:
        r"""Eq. 3: G_T = \prod_{l=1}^{L_T} \Upsilon_l^{(T)} W_l^{(T)}"""
        G = np.eye(self.N_in_T, dtype=complex)
        for l in range(self.L_T):
            Y_l = np.diag(np.exp(1j * self._phase_shifts_TX[l]))
            G = Y_l @ self.W_TX[l] @ G
        return G

    def _calculate_G_R(self) -> np.ndarray:
        r"""Eq. 5: G_R = \prod_{l=1}^{L_R} \Upsilon_l^{(R)} W_l^{(R)}"""
        G = np.eye(self.N_in_R, dtype=complex)
        for l in range(self.L_R):
            Y_l = np.diag(np.exp(1j * self._phase_shifts_RX[l]))
            G = Y_l @ self.W_RX[l] @ G
        return G
    
    def calculate_effective_cascade(self, H_mimo: np.ndarray) -> np.ndarray:
        """Calculates the total operator Z = G_R * H * G_T (Eq. 7)."""
        GT = self._calculate_G_T()
        GR = self._calculate_G_R()
        return GR @ H_mimo @ GT

    def calculate_optimal_beta(self, Z: np.ndarray, A: np.ndarray) -> complex:
        """Calculates the optimal scaling factor beta (Eq. 17)."""
        z_vec = Z.flatten()
        a_vec = A.flatten()
        return np.vdot(z_vec, a_vec) / np.vdot(z_vec, z_vec)

class DualSIMoptimizerTorch(nn.Module):
    def __init__(self, sim_cpu):
        """
        PyTorch implementation of the DualSIMoptimizer.
        Uses register_buffer to ensure fixed matrices follow the model to the GPU.
        """
        super().__init__()
        self.verbose = sim_cpu.verbose
        self.L_T = sim_cpu.L_T
        self.L_R = sim_cpu.L_R

        # 1. Register fixed matrices as BUFFERS (This is the key for GPU compatibility)
        for i, W in enumerate(sim_cpu.W_TX):
            self.register_buffer(f'W_T_{i}', torch.tensor(W, dtype=torch.complex64))
        
        for i, W in enumerate(sim_cpu.W_RX):
            self.register_buffer(f'W_R_{i}', torch.tensor(W, dtype=torch.complex64))

        # 2. Trainable Parameters: phase shifts xi_T and xi_R
        self.xi_T = nn.ParameterList([
            nn.Parameter(torch.tensor(p, dtype=torch.float32)) for p in sim_cpu._phase_shifts_TX
        ])
        self.xi_R = nn.ParameterList([
            nn.Parameter(torch.tensor(p, dtype=torch.float32)) for p in sim_cpu._phase_shifts_RX
        ])

        

    def _get_W_list(self, prefix, L):
        """Helper to retrieve registered buffers as a list."""
        return [getattr(self, f'{prefix}_{i}') for i in range(L)]

    def _calculate_G_T(self):
        W_T_list = self._get_W_list('W_T', self.L_T)
        return self._calculate_G(W_T_list, self.xi_T, self.L_T, W_T_list[0].shape[1])

    def _calculate_G_R(self):
        W_R_list = self._get_W_list('W_R', self.L_R)
        return self._calculate_G(W_R_list, self.xi_R, self.L_R, W_R_list[0].shape[1])
    
    def _calculate_G(self, W_list, phases, L, input_dim):
        # Now W_list elements are on the same device as the model automatically
        device = phases[0].device
        G = torch.eye(input_dim, dtype=torch.complex64, device=device)
        
        for l in range(L):
            # Applying phase shifts
            Y_l = torch.diag(torch.exp(1j * phases[l]))
            # Both Y_l, W_list[l] and G are now on the same device (CUDA)
            G = Y_l @ W_list[l] @ G
        return G

    def get_effective_cascade(self, H):
        """Calculates Z = G_R * H * G_T (Eq. 7)."""
        GT = self._calculate_G_T()
        GR = self._calculate_G_R()
        Z = GR @ H @ GT
        return Z, GR

    def get_TX_cascade(self):
        """Restituisce la matrice di propagazione G_T della SIM Trasmettitore."""
        return self._calculate_G_T()

    def get_RX_cascade(self):
        """Restituisce la matrice di propagazione G_R della SIM Ricevitore."""
        return self._calculate_G_R()    

    #def optimize_alternating(self, A_target, H_mimo, max_iters, lr=0.1, lambda_reg=1e-4):
        """
        Alternating Optimization Algorithm with explicit Beta calculation.
        Minimizes Frobenius norm error between scaled cascade and target A.
        """
     
        #A_torch = torch.as_tensor(A_target, dtype=torch.complex64)
        #H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64)
        
        #opt_T = optim.Adam(self.xi_T.parameters(), lr=lr)
       # opt_R = optim.Adam(self.xi_R.parameters(), lr=lr)
        
       # loss_history = []

        #for k in tqdm(range(max_iters), disable=not self.verbose, desc="Alternating Optimization"):
            # --- STEP 1: Update Beta (Closed form) ---
         #   with torch.no_grad():
          #      Z_current, _ = self.get_effective_cascade(H_torch)
                # Compute beta = <Z, A> / <Z, Z>
           #     beta = torch.sum(torch.conj(Z_current) * A_torch) / torch.sum(torch.conj(Z_current) * Z_current)

            # --- STEP 2: TX Optimization (xi_T) ---
            #opt_T.zero_grad()
            #Z_t, _ = self.get_effective_cascade(H_torch)
            #loss_J_T = torch.norm(beta * Z_t - A_torch, p='fro')**2
            #loss_J_T.backward()
            #opt_T.step()

            # --- STEP 3: RX Optimization (xi_R) + Noise Regularization ---
            #opt_R.zero_grad()
            #Z_r, GR = self.get_effective_cascade(H_torch)
           # loss_J_R = torch.norm(beta * Z_r - A_torch, p='fro')**2
            #loss_N = torch.norm(GR, p='fro')**2  # Regularization to control noise enhancement
           # total_loss_R = loss_J_R + lambda_reg * loss_N
            #total_loss_R.backward()
            #opt_R.step()

            # --- STEP 4: Phase Projection [0, 2*pi) ---
           # with torch.no_grad():
           #     for p in self.xi_T: p.copy_(p % (2 * torch.pi))
            #    for p in self.xi_R: p.copy_(p % (2 * torch.pi))

           # current_loss=loss_J_R.item()
           # loss_history.append(current_loss)
            #if k % 50 == 0:
             #   print(f"      [Iter {k:4d}/{max_iters}] Semantic Loss: {current_loss:.6f}")
            
        # --- CALCOLO FINALE ERRORE DI FROBENIUS ---
        #with torch.no_grad():
         #   Z_final, _ = self.get_effective_cascade(H_torch)
          #  beta_f = torch.sum(torch.conj(Z_final) * A_torch) / (torch.sum(torch.conj(Z_final) * Z_final) + 1e-12)
           # fro_err = torch.norm(beta_f * Z_final - A_torch) / torch.norm(A_torch)
            #print(f"   🏁 Optimization Done. Final Relative Frobenius Error: {fro_err.item():.4f}\n")
            
        #return loss_history


    def optimize_alternating(self, A_target, H_mimo, max_iters, lr=0.1, lambda_reg=1e-4):
        """
        Alternating Optimization Algorithm (Projected Gradient Descent).
        Segue rigorosamente le equazioni del paper:
        a) Update beta in forma chiusa (fissato per TX e RX nello stesso step).
        b) Update xi_T tramite gradiente e proiezione [0, 2pi).
        c) Update xi_R tramite gradiente (usando xi_T aggiornato) e proiezione [0, 2pi).
        """
        import torch.optim as optim

        A_torch = torch.as_tensor(A_target, dtype=torch.complex64)
        H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64)
        
        # Gradient Descent PURO (SGD) 
        opt_T = optim.SGD(self.xi_T.parameters(), lr=lr)
        opt_R = optim.SGD(self.xi_R.parameters(), lr=lr)
        
        loss_history = []

        for k in range(max_iters):
            # --- a) Closed-form update for beta^k (fissato per l'intera iterazione k) ---
            with torch.no_grad():
                Z_k, _ = self.get_effective_cascade(H_torch)
                # beta^k = <Z, A> / ||Z||^2
                num_beta = torch.sum(torch.conj(Z_k) * A_torch)
                den_beta = torch.sum(torch.conj(Z_k) * Z_k) + 1e-12
                beta_k = num_beta / den_beta

            # --- b) TX Optimization ---
            opt_T.zero_grad()
            Z_T, _ = self.get_effective_cascade(H_torch)
            loss_J_T = torch.norm(beta_k * Z_T - A_torch, p='fro')**2
            loss_J_T.backward()
            
            # TRUCCO DI STABILITÀ: Normalizzazione del gradiente
            with torch.no_grad():
                for p in self.xi_T:
                    if p.grad is not None:
                        # Dividiamo per la norma per forzare il passo a essere grande quanto 'lr'
                        grad_norm = torch.norm(p.grad) + 1e-12
                        p.grad /= grad_norm 
            opt_T.step()

            # Proiezione [0, 2pi)
            with torch.no_grad():
                for p in self.xi_T: p.copy_(p % (2 * torch.pi))

            # --- c) RX Optimization ---
            opt_R.zero_grad()
            Z_R, G_R = self.get_effective_cascade(H_torch)
            loss_J_R = torch.norm(beta_k * Z_R - A_torch, p='fro')**2
            total_loss_R = loss_J_R + lambda_reg * torch.norm(G_R, p='fro')**2
            total_loss_R.backward()
            
            # TRUCCO DI STABILITÀ: Normalizzazione del gradiente
            with torch.no_grad():
                for p in self.xi_R:
                    if p.grad is not None:
                        grad_norm = torch.norm(p.grad) + 1e-12
                        p.grad /= grad_norm
            opt_R.step()

            with torch.no_grad():
                for p in self.xi_R: p.copy_(p % (2 * torch.pi))

            current_loss = loss_J_R.item()
            loss_history.append(current_loss)
            if k % 50 == 0:
                print(f"      [Iter {k:4d}/{max_iters}] Semantic Loss: {current_loss:.6f} | Beta Mag: {torch.abs(beta_k).item():.2e}")
            
        # --- CALCOLO FINALE ERRORE DI FROBENIUS ---
        with torch.no_grad():
            Z_final, _ = self.get_effective_cascade(H_torch)
            beta_f = torch.sum(torch.conj(Z_final) * A_torch) / (torch.sum(torch.conj(Z_final) * Z_final) + 1e-12)
            fro_err = torch.norm(beta_f * Z_final - A_torch) / torch.norm(A_torch)
            print(f"   🏁 Optimization Done. Final Relative Frobenius Error: {fro_err.item():.4f}\n")
            
        return loss_history    
      
    def optimize_disjoint(self, A_target, H_mimo, noise_var, max_iters, lr=0.1, lambda_reg=1e-4):
        """
        True Disjoint Optimization (Separation Principle).
        - TX punta ad A (Allineamento Semantico)
        - RX punta a I (Diagonalizzazione del Canale: G_R * H \approx I)
        """
        import torch.optim as optim
        
        A_torch = torch.as_tensor(A_target, dtype=torch.complex64)
        H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64)
        
        # --- NUOVO TARGET RX: MATRICE IDENTITA' ---
        # Vogliamo che il canale equivalente G_R * H diventi trasparente (Identity)
        I_target = torch.eye(H_torch.shape[1], dtype=torch.complex64, device=H_torch.device)

        opt_T = optim.Adam(self.xi_T.parameters(), lr=lr)
        opt_R = optim.Adam(self.xi_R.parameters(), lr=lr)
        
        loss_history_T = []
        loss_history_R = []

        for k in range(max_iters):
            
            # ==========================================
            # 1. TX SIM: Allineamento Semantico (Identico a Mono-SIM)
            # ==========================================
            opt_T.zero_grad()
            G_T = self.get_TX_cascade() 
            
            with torch.no_grad():
                beta_t = torch.sum(torch.conj(G_T) * A_torch) / (torch.sum(torch.conj(G_T) * G_T) + 1e-12)
            
            loss_T = torch.norm(beta_t * G_T - A_torch, p='fro')**2
            loss_T.backward()
            opt_T.step()

            # ==========================================
            # 2. RX SIM: Inversione / Diagonalizzazione del Canale
            # Vogliamo che (G_R * H) sia proporzionale a I
            # ==========================================
            opt_R.zero_grad()
            G_R = self.get_RX_cascade() 
            H_eq = G_R @ H_torch # Canale Equivalente
            
            # Calcolo di beta_r rispetto all'Identità
            with torch.no_grad():
                beta_r = torch.sum(torch.conj(H_eq) * I_target) / (torch.sum(torch.conj(H_eq) * H_eq) + 1e-12)
            
            # Loss RX: Quanto il canale equivalente è lontano dall'identità perfetta
            loss_R = torch.norm(beta_r * H_eq - I_target, p='fro')**2
            
            # Regolarizzazione: Evitare che G_R amplifichi il rumore
            total_loss_R = loss_R + lambda_reg * torch.norm(G_R, p='fro')**2
            
            total_loss_R.backward()
            opt_R.step()

            # ==========================================
            # 3. Proiezione per entrambe le SIM
            # ==========================================
            with torch.no_grad():
                for p in self.xi_T: p.copy_(p % (2 * torch.pi))
                for p in self.xi_R: p.copy_(p % (2 * torch.pi))
                
            loss_history_T.append(loss_T.item())
            loss_history_R.append(loss_R.item())
            
            if k % 50 == 0:
                print(f"      [Iter {k:4d}/{max_iters}] Loss TX (A): {loss_T.item():.2f} | Loss RX (H->I): {loss_R.item():.2f}")

        # --- CALCOLI FINALI PER INFERENZA ---
        with torch.no_grad():
            G_T_final = self.get_TX_cascade()
            G_R_final = self.get_RX_cascade()
            
            # Beta per TX (rispetto ad A)
            beta_t_final = torch.sum(torch.conj(G_T_final) * A_torch) / (torch.sum(torch.conj(G_T_final) * G_T_final) + 1e-12)
            
            # Beta per RX (rispetto a I)
            H_eq_final = G_R_final @ H_torch
            beta_r_final = torch.sum(torch.conj(H_eq_final) * I_target) / (torch.sum(torch.conj(H_eq_final) * H_eq_final) + 1e-12)
            
            # Errore combinato finale: Z = (beta_r * G_R) * H * (beta_t * G_T)
            # Se RX ha funzionato: (beta_r * G_R * H) ≈ I
            # Se TX ha funzionato: (beta_t * G_T) ≈ A
            # Allora Z_totale ≈ I * A = A
            Z_final = H_eq_final @ G_T_final
            fro_err = torch.norm((beta_t_final * beta_r_final) * Z_final - A_torch) / torch.norm(A_torch)
            print(f"   🏁 Disjoint Opt Done. Final Combined Error: {fro_err.item():.4f}\n")

        # Restituire il prodotto dei beta per l'inferenza
        return loss_history_T, loss_history_R, (beta_t_final * beta_r_final)


      