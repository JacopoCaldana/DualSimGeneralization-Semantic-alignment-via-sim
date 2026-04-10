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

        # 2. Trainable Parameters: phase shifts xi_T and xi_R (TRANSPARENT INIT)
        self.xi_T = nn.ParameterList([
            # Creiamo un tensore di ZERI con la stessa identica forma di 'p'
            nn.Parameter(torch.zeros_like(torch.tensor(p, dtype=torch.float32))) 
            for p in sim_cpu._phase_shifts_TX
        ])
        
        self.xi_R = nn.ParameterList([
            # Creiamo un tensore di ZERI con la stessa identica forma di 'p'
            nn.Parameter(torch.zeros_like(torch.tensor(p, dtype=torch.float32))) 
            for p in sim_cpu._phase_shifts_RX
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




    def optimize_alternating(
        self,
        A_target,
        H_mimo,
        max_iters: int = 3000,
        lr: float = 0.3,
        lambda_reg: float = 0.0,
        warmup_frac: float = 0.05,
        clip_value: float = 1.0,
        rx_steps_per_tx: int = 2,
        patience: int = 300,
        rel_tol: float = 1e-5,
    ):
        """
        Alternating Optimization for Dual-SIM Phase Configuration.

        Bug fixes vs previous version
        ------------------------------
        BUG 1 – Scheduler step count mismatch (LR appearing flat in logs):
            LambdaLR.step() was called once per outer iteration, but the lambda
            function was written as if `step` == outer iteration index.  With
            rx_steps_per_tx=2 the RX scheduler was called at the right pace, but
            the TX scheduler was called at the same pace — both received the same
            monotone `step` counter from PyTorch's internal epoch counter, so the
            lambda was evaluated correctly only if sched.step() is called exactly
            once per iteration.  The real issue was that the flat-LR seen in the
            logs (e.g. 0.049482 constant for 2000 iters) came from the previous
            SGD+MultiStepLR code being invoked instead of this one.  To make the
            schedule robust and readable, we now use CosineAnnealingLR directly,
            which takes T_max (total steps) explicitly and needs no lambda.

        BUG 2 – Beta explosion with deep cascades (|β| → 10^5 for L=20):
            Each W matrix attenuates by ~(slayer·Acell/d²), so ‖G_T‖ and ‖G_R‖
            decay exponentially in L.  The cascade Z = G_R H G_T therefore has
            ‖Z‖_F ≈ ε_L → 0 for large L, causing β = <Z,A>/‖Z‖² → ∞.
            The gradient of ‖βZ − A‖² w.r.t. ξ is 2β·Re(∂Z/∂ξ)^H(βZ−A), which
            explodes with β even after clipping, because clip_grad_norm_ cannot
            know that the true signal is in the direction of ∂Z/∂ξ, not β.
            FIX: work with the *normalised* cascade Ẑ = Z/‖Z‖_F.  Then β̂ = <Ẑ,A>
            is bounded by ‖A‖_F, gradients stay well-scaled at all depths, and
            the loss ‖β̂Ẑ − A‖² is invariant to the overall SIM attenuation.
            The true β (needed for inference) is recovered as β = β̂/‖Z‖_F.

        Design rationale (unchanged)
        -----------------------------
        - Adam with per-parameter second moments as implicit Hessian preconditioner.
        - CosineAnnealingLR with linear warm-up for smooth LR decay.
        - Global gradient clipping (preserves inter-layer gradient ratios).
        - Asymmetric RX/TX update budget (RX landscape is harder).
        - Best-iterate tracking + patience-based early stopping.

        Parameters
        ----------
        A_target        : array-like, complex  – target semantic operator A
        H_mimo          : array-like, complex  – MIMO channel matrix H
        max_iters       : int                  – maximum outer iterations
        lr              : float                – peak learning rate
        lambda_reg      : float                – RX noise regularisation weight λ
        warmup_frac     : float                – fraction of iters for LR warm-up
        clip_value      : float                – global gradient clip norm
        rx_steps_per_tx : int                  – RX inner steps per TX outer step
        patience        : int                  – early-stop window (iterations)
        rel_tol         : float                – relative improvement threshold

        Returns
        -------
        loss_history : list[float]  – normalised emulation loss per outer step
        """

        print(f"\n⚡ ADAM ENGINE STARTING: L={len(self.xi_T)} | lr={lr}")
        import math

        A_torch = torch.as_tensor(A_target, dtype=torch.complex64)
        H_torch = torch.as_tensor(H_mimo,   dtype=torch.complex64)

        device  = next(self.parameters()).device
        A_torch = A_torch.to(device)
        H_torch = H_torch.to(device)

        # Pre-compute ‖A‖_F once — used for relative error reporting only
        A_norm = torch.norm(A_torch, p='fro').item()

        # ------------------------------------------------------------------ #
        #  Optimisers                                                         #
        # ------------------------------------------------------------------ #
        opt_T = optim.Adam(self.xi_T.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        opt_R = optim.Adam(self.xi_R.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

        # ------------------------------------------------------------------ #
        #  Schedulers — CosineAnnealingLR is explicit about T_max,           #
        #  no lambda arithmetic needed, step() called once per outer iter.   #
        #  Linear warm-up handled via a multiplicative warm-up scheduler      #
        #  chained with CosineAnnealingLR via SequentialLR.                  #
        # ------------------------------------------------------------------ #
        warmup_iters = max(1, int(max_iters * warmup_frac))
        cosine_iters = max(1, max_iters - warmup_iters)
        lr_min       = lr * 1e-3

        def _make_schedulers(opt):
            warmup = optim.lr_scheduler.LinearLR(
                opt,
                start_factor=1e-3,   # begin at lr * 1e-3
                end_factor=1.0,      # reach full lr at end of warm-up
                total_iters=warmup_iters,
            )
            cosine = optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=cosine_iters,
                eta_min=lr_min,
            )
            return optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[warmup, cosine],
                milestones=[warmup_iters],
            )

        sched_T = _make_schedulers(opt_T)
        sched_R = _make_schedulers(opt_R)

        # ------------------------------------------------------------------ #
        #  Helper: normalised beta (BUG 2 fix)                               #
        #  Returns (beta_hat, Z_norm) where beta_hat = <Ẑ, A> is O(‖A‖)    #
        #  and Z_norm = ‖Z‖_F.  The true beta = beta_hat / Z_norm.          #
        # ------------------------------------------------------------------ #
        def _normalised_beta(Z):
            Z_norm = torch.norm(Z, p='fro') + 1e-12
            Z_hat  = Z / Z_norm
            beta_hat = torch.sum(torch.conj(Z_hat) * A_torch).detach()
            return beta_hat, Z_norm.detach()

        # ------------------------------------------------------------------ #
        #  Helper: one gradient step on a SIM side using normalised loss     #
        # ------------------------------------------------------------------ #
        def _step_side(opt, params, extra_loss=None):
            opt.zero_grad()
            Z, G_R = self.get_effective_cascade(H_torch)
            # Normalise cascade to decouple β from phase gradients
            Z_norm   = torch.norm(Z, p='fro') + 1e-12
            Z_hat    = Z / Z_norm
            beta_hat = torch.sum(torch.conj(Z_hat) * A_torch)
            loss_J   = torch.norm(beta_hat * Z_hat - A_torch, p='fro') ** 2
            total    = loss_J + (extra_loss(G_R) if extra_loss else 0.0)
            total.backward()
            nn.utils.clip_grad_norm_(params, max_norm=clip_value)
            opt.step()
            with torch.no_grad():
                for p in params:
                    p.copy_(p % (2.0 * math.pi))
            return loss_J.item(), (beta_hat / Z_norm).detach(), Z_norm.detach()

        noise_reg = (lambda G_R: lambda_reg * torch.norm(G_R, p='fro') ** 2) \
                    if lambda_reg > 0 else None

        # ------------------------------------------------------------------ #
        #  Main loop                                                          #
        # ------------------------------------------------------------------ #
        loss_history = []
        best_loss    = float('inf')
        best_xi_T    = [p.data.clone() for p in self.xi_T]
        best_xi_R    = [p.data.clone() for p in self.xi_R]
        beta_k       = torch.tensor(1.0 + 0j, device=device)
        Z_norm_k     = torch.tensor(1.0,      device=device)

        for k in range(max_iters):

            # --- 1. TX step ---
            _, _, _ = _step_side(opt_T, list(self.xi_T.parameters()))

            # --- 2. RX step(s) ---
            current_loss = 0.0
            for _ in range(rx_steps_per_tx):
                current_loss, beta_k, Z_norm_k = _step_side(
                    opt_R, list(self.xi_R.parameters()), noise_reg
                )

            # --- 3. Scheduler step (once per outer iteration – BUG 1 fix) ---
            sched_T.step()
            sched_R.step()

            loss_history.append(current_loss)

            # --- 4. Best-iterate tracking ---
            if current_loss < best_loss:
                best_loss = current_loss
                best_xi_T = [p.data.clone() for p in self.xi_T]
                best_xi_R = [p.data.clone() for p in self.xi_R]

            # --- 5. Logging ---
            if k % 100 == 0:
                cur_lr     = opt_T.param_groups[0]['lr']
                true_beta  = torch.abs(beta_k).item()
                print(
                    f"  [Iter {k:4d}/{max_iters}] "
                    f"Loss: {current_loss:.6f} | "
                    f"LR: {cur_lr:.2e} | "
                    f"|β|: {true_beta:.2e} | "
                    f"‖Z‖: {Z_norm_k.item():.2e}"
                )

            # --- 6. Early stopping ---
            if k >= patience:
                window_start = loss_history[-patience]
                if window_start > 1e-12:
                    rel_improvement = (window_start - current_loss) / window_start
                    if rel_improvement < rel_tol:
                        print(
                            f"  [Early stop @ iter {k}]  "
                            f"Δloss/loss = {rel_improvement:.2e} < tol {rel_tol:.2e}"
                        )
                        break

        # ------------------------------------------------------------------ #
        #  Restore best parameters                                           #
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            for p, best in zip(self.xi_T, best_xi_T):
                p.copy_(best)
            for p, best in zip(self.xi_R, best_xi_R):
                p.copy_(best)

        # ------------------------------------------------------------------ #
        #  Final report                                                       #
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            Z_f, _   = self.get_effective_cascade(H_torch)
            Z_norm_f = torch.norm(Z_f, p='fro') + 1e-12
            Z_hat_f  = Z_f / Z_norm_f
            beta_f   = torch.sum(torch.conj(Z_hat_f) * A_torch) / Z_norm_f
            fro_err  = torch.norm(beta_f * Z_f - A_torch) / A_norm
            print(
                f"  ✓ Done. Best normalised loss: {best_loss:.6f} | "
                f"Relative Frobenius error: {fro_err.item():.4f} | "
                f"‖Z‖_F: {Z_norm_f.item():.2e}\n"
            )

        return loss_history  
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

        # 2. Trainable Parameters: phase shifts xi_T and xi_R (TRANSPARENT INIT)
        self.xi_T = nn.ParameterList([
            # Creiamo un tensore di ZERI con la stessa identica forma di 'p'
            nn.Parameter(torch.zeros_like(torch.tensor(p, dtype=torch.float32))) 
            for p in sim_cpu._phase_shifts_TX
        ])
        
        self.xi_R = nn.ParameterList([
            # Creiamo un tensore di ZERI con la stessa identica forma di 'p'
            nn.Parameter(torch.zeros_like(torch.tensor(p, dtype=torch.float32))) 
            for p in sim_cpu._phase_shifts_RX
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




    def optimize_alternating(
        self,
        A_target,
        H_mimo,
        max_iters: int = 3000,
        lr: float = 0.3,
        lambda_reg: float = 0.0,
        warmup_frac: float = 0.05,
        clip_value: float = 1.0,
        rx_steps_per_tx: int = 2,
        patience: int = 300,
        rel_tol: float = 1e-5,
    ):
        """
        Alternating Optimization for Dual-SIM Phase Configuration.

        Bug fixes vs previous version
        ------------------------------
        BUG 1 – Scheduler step count mismatch (LR appearing flat in logs):
            LambdaLR.step() was called once per outer iteration, but the lambda
            function was written as if `step` == outer iteration index.  With
            rx_steps_per_tx=2 the RX scheduler was called at the right pace, but
            the TX scheduler was called at the same pace — both received the same
            monotone `step` counter from PyTorch's internal epoch counter, so the
            lambda was evaluated correctly only if sched.step() is called exactly
            once per iteration.  The real issue was that the flat-LR seen in the
            logs (e.g. 0.049482 constant for 2000 iters) came from the previous
            SGD+MultiStepLR code being invoked instead of this one.  To make the
            schedule robust and readable, we now use CosineAnnealingLR directly,
            which takes T_max (total steps) explicitly and needs no lambda.

        BUG 2 – Beta explosion with deep cascades (|β| → 10^5 for L=20):
            Each W matrix attenuates by ~(slayer·Acell/d²), so ‖G_T‖ and ‖G_R‖
            decay exponentially in L.  The cascade Z = G_R H G_T therefore has
            ‖Z‖_F ≈ ε_L → 0 for large L, causing β = <Z,A>/‖Z‖² → ∞.
            The gradient of ‖βZ − A‖² w.r.t. ξ is 2β·Re(∂Z/∂ξ)^H(βZ−A), which
            explodes with β even after clipping, because clip_grad_norm_ cannot
            know that the true signal is in the direction of ∂Z/∂ξ, not β.
            FIX: work with the *normalised* cascade Ẑ = Z/‖Z‖_F.  Then β̂ = <Ẑ,A>
            is bounded by ‖A‖_F, gradients stay well-scaled at all depths, and
            the loss ‖β̂Ẑ − A‖² is invariant to the overall SIM attenuation.
            The true β (needed for inference) is recovered as β = β̂/‖Z‖_F.

        Design rationale (unchanged)
        -----------------------------
        - Adam with per-parameter second moments as implicit Hessian preconditioner.
        - CosineAnnealingLR with linear warm-up for smooth LR decay.
        - Global gradient clipping (preserves inter-layer gradient ratios).
        - Asymmetric RX/TX update budget (RX landscape is harder).
        - Best-iterate tracking + patience-based early stopping.

        Parameters
        ----------
        A_target        : array-like, complex  – target semantic operator A
        H_mimo          : array-like, complex  – MIMO channel matrix H
        max_iters       : int                  – maximum outer iterations
        lr              : float                – peak learning rate
        lambda_reg      : float                – RX noise regularisation weight λ
        warmup_frac     : float                – fraction of iters for LR warm-up
        clip_value      : float                – global gradient clip norm
        rx_steps_per_tx : int                  – RX inner steps per TX outer step
        patience        : int                  – early-stop window (iterations)
        rel_tol         : float                – relative improvement threshold

        Returns
        -------
        loss_history : list[float]  – normalised emulation loss per outer step
        """

        print(f"\n⚡ ADAM ENGINE STARTING: L={len(self.xi_T)} | lr={lr}")
        import math

        A_torch = torch.as_tensor(A_target, dtype=torch.complex64)
        H_torch = torch.as_tensor(H_mimo,   dtype=torch.complex64)

        device  = next(self.parameters()).device
        A_torch = A_torch.to(device)
        H_torch = H_torch.to(device)

        # Pre-compute ‖A‖_F once — used for relative error reporting only
        A_norm = torch.norm(A_torch, p='fro').item()

        # ------------------------------------------------------------------ #
        #  Optimisers                                                         #
        # ------------------------------------------------------------------ #
        opt_T = optim.Adam(self.xi_T.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        opt_R = optim.Adam(self.xi_R.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

        # ------------------------------------------------------------------ #
        #  Schedulers — CosineAnnealingLR is explicit about T_max,           #
        #  no lambda arithmetic needed, step() called once per outer iter.   #
        #  Linear warm-up handled via a multiplicative warm-up scheduler      #
        #  chained with CosineAnnealingLR via SequentialLR.                  #
        # ------------------------------------------------------------------ #
        warmup_iters = max(1, int(max_iters * warmup_frac))
        cosine_iters = max(1, max_iters - warmup_iters)
        lr_min       = lr * 1e-3

        def _make_schedulers(opt):
            warmup = optim.lr_scheduler.LinearLR(
                opt,
                start_factor=1e-3,   # begin at lr * 1e-3
                end_factor=1.0,      # reach full lr at end of warm-up
                total_iters=warmup_iters,
            )
            cosine = optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=cosine_iters,
                eta_min=lr_min,
            )
            return optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[warmup, cosine],
                milestones=[warmup_iters],
            )

        sched_T = _make_schedulers(opt_T)
        sched_R = _make_schedulers(opt_R)

        # ------------------------------------------------------------------ #
        #  Helper: normalised beta (BUG 2 fix)                               #
        #  Returns (beta_hat, Z_norm) where beta_hat = <Ẑ, A> is O(‖A‖)    #
        #  and Z_norm = ‖Z‖_F.  The true beta = beta_hat / Z_norm.          #
        # ------------------------------------------------------------------ #
        def _normalised_beta(Z):
            Z_norm = torch.norm(Z, p='fro') + 1e-12
            Z_hat  = Z / Z_norm
            beta_hat = torch.sum(torch.conj(Z_hat) * A_torch).detach()
            return beta_hat, Z_norm.detach()

        # ------------------------------------------------------------------ #
        #  Helper: one gradient step on a SIM side using normalised loss     #
        # ------------------------------------------------------------------ #
        def _step_side(opt, params, extra_loss=None):
            opt.zero_grad()
            Z, G_R = self.get_effective_cascade(H_torch)
            # Normalise cascade to decouple β from phase gradients
            Z_norm   = torch.norm(Z, p='fro') + 1e-12
            Z_hat    = Z / Z_norm
            beta_hat = torch.sum(torch.conj(Z_hat) * A_torch)
            loss_J   = torch.norm(beta_hat * Z_hat - A_torch, p='fro') ** 2
            total    = loss_J + (extra_loss(G_R) if extra_loss else 0.0)
            total.backward()
            nn.utils.clip_grad_norm_(params, max_norm=clip_value)
            opt.step()
            with torch.no_grad():
                for p in params:
                    p.copy_(p % (2.0 * math.pi))
            return loss_J.item(), (beta_hat / Z_norm).detach(), Z_norm.detach()

        noise_reg = (lambda G_R: lambda_reg * torch.norm(G_R, p='fro') ** 2) \
                    if lambda_reg > 0 else None

        # ------------------------------------------------------------------ #
        #  Main loop                                                          #
        # ------------------------------------------------------------------ #
        loss_history = []
        best_loss    = float('inf')
        best_xi_T    = [p.data.clone() for p in self.xi_T]
        best_xi_R    = [p.data.clone() for p in self.xi_R]
        beta_k       = torch.tensor(1.0 + 0j, device=device)
        Z_norm_k     = torch.tensor(1.0,      device=device)

        for k in range(max_iters):

            # --- 1. TX step ---
            _, _, _ = _step_side(opt_T, list(self.xi_T.parameters()))

            # --- 2. RX step(s) ---
            current_loss = 0.0
            for _ in range(rx_steps_per_tx):
                current_loss, beta_k, Z_norm_k = _step_side(
                    opt_R, list(self.xi_R.parameters()), noise_reg
                )

            # --- 3. Scheduler step (once per outer iteration – BUG 1 fix) ---
            sched_T.step()
            sched_R.step()

            loss_history.append(current_loss)

            # --- 4. Best-iterate tracking ---
            if current_loss < best_loss:
                best_loss = current_loss
                best_xi_T = [p.data.clone() for p in self.xi_T]
                best_xi_R = [p.data.clone() for p in self.xi_R]

            # --- 5. Logging ---
            if k % 100 == 0:
                cur_lr     = opt_T.param_groups[0]['lr']
                true_beta  = torch.abs(beta_k).item()
                print(
                    f"  [Iter {k:4d}/{max_iters}] "
                    f"Loss: {current_loss:.6f} | "
                    f"LR: {cur_lr:.2e} | "
                    f"|β|: {true_beta:.2e} | "
                    f"‖Z‖: {Z_norm_k.item():.2e}"
                )

            # --- 6. Early stopping ---
            if k >= patience:
                window_start = loss_history[-patience]
                if window_start > 1e-12:
                    rel_improvement = (window_start - current_loss) / window_start
                    if rel_improvement < rel_tol:
                        print(
                            f"  [Early stop @ iter {k}]  "
                            f"Δloss/loss = {rel_improvement:.2e} < tol {rel_tol:.2e}"
                        )
                        break

        # ------------------------------------------------------------------ #
        #  Restore best parameters                                           #
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            for p, best in zip(self.xi_T, best_xi_T):
                p.copy_(best)
            for p, best in zip(self.xi_R, best_xi_R):
                p.copy_(best)

        # ------------------------------------------------------------------ #
        #  Final report                                                       #
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            Z_f, _   = self.get_effective_cascade(H_torch)
            Z_norm_f = torch.norm(Z_f, p='fro') + 1e-12
            Z_hat_f  = Z_f / Z_norm_f
            beta_f   = torch.sum(torch.conj(Z_hat_f) * A_torch) / Z_norm_f
            fro_err  = torch.norm(beta_f * Z_f - A_torch) / A_norm
            print(
                f"  ✓ Done. Best normalised loss: {best_loss:.6f} | "
                f"Relative Frobenius error: {fro_err.item():.4f} | "
                f"‖Z‖_F: {Z_norm_f.item():.2e}\n"
            )

        return loss_history  


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

        # 2. Trainable Parameters: phase shifts xi_T and xi_R (TRANSPARENT INIT)
        self.xi_T = nn.ParameterList([
            # Creiamo un tensore di ZERI con la stessa identica forma di 'p'
            nn.Parameter(torch.zeros_like(torch.tensor(p, dtype=torch.float32))) 
            for p in sim_cpu._phase_shifts_TX
        ])
        
        self.xi_R = nn.ParameterList([
            # Creiamo un tensore di ZERI con la stessa identica forma di 'p'
            nn.Parameter(torch.zeros_like(torch.tensor(p, dtype=torch.float32))) 
            for p in sim_cpu._phase_shifts_RX
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




    def optimize_alternating(
        self,
        A_target,
        H_mimo,
        max_iters: int = 3000,
        lr: float = 0.3,
        lambda_reg: float = 0.0,
        warmup_frac: float = 0.05,
        clip_value: float = 1.0,
        rx_steps_per_tx: int = 2,
        patience: int = 300,
        rel_tol: float = 1e-5,
    ):
        """
        Alternating Optimization for Dual-SIM Phase Configuration.

        Bug fixes vs previous version
        ------------------------------
        BUG 1 – Scheduler step count mismatch (LR appearing flat in logs):
            LambdaLR.step() was called once per outer iteration, but the lambda
            function was written as if `step` == outer iteration index.  With
            rx_steps_per_tx=2 the RX scheduler was called at the right pace, but
            the TX scheduler was called at the same pace — both received the same
            monotone `step` counter from PyTorch's internal epoch counter, so the
            lambda was evaluated correctly only if sched.step() is called exactly
            once per iteration.  The real issue was that the flat-LR seen in the
            logs (e.g. 0.049482 constant for 2000 iters) came from the previous
            SGD+MultiStepLR code being invoked instead of this one.  To make the
            schedule robust and readable, we now use CosineAnnealingLR directly,
            which takes T_max (total steps) explicitly and needs no lambda.

        BUG 2 – Beta explosion with deep cascades (|β| → 10^5 for L=20):
            Each W matrix attenuates by ~(slayer·Acell/d²), so ‖G_T‖ and ‖G_R‖
            decay exponentially in L.  The cascade Z = G_R H G_T therefore has
            ‖Z‖_F ≈ ε_L → 0 for large L, causing β = <Z,A>/‖Z‖² → ∞.
            The gradient of ‖βZ − A‖² w.r.t. ξ is 2β·Re(∂Z/∂ξ)^H(βZ−A), which
            explodes with β even after clipping, because clip_grad_norm_ cannot
            know that the true signal is in the direction of ∂Z/∂ξ, not β.
            FIX: work with the *normalised* cascade Ẑ = Z/‖Z‖_F.  Then β̂ = <Ẑ,A>
            is bounded by ‖A‖_F, gradients stay well-scaled at all depths, and
            the loss ‖β̂Ẑ − A‖² is invariant to the overall SIM attenuation.
            The true β (needed for inference) is recovered as β = β̂/‖Z‖_F.

        Design rationale (unchanged)
        -----------------------------
        - Adam with per-parameter second moments as implicit Hessian preconditioner.
        - CosineAnnealingLR with linear warm-up for smooth LR decay.
        - Global gradient clipping (preserves inter-layer gradient ratios).
        - Asymmetric RX/TX update budget (RX landscape is harder).
        - Best-iterate tracking + patience-based early stopping.

        Parameters
        ----------
        A_target        : array-like, complex  – target semantic operator A
        H_mimo          : array-like, complex  – MIMO channel matrix H
        max_iters       : int                  – maximum outer iterations
        lr              : float                – peak learning rate
        lambda_reg      : float                – RX noise regularisation weight λ
        warmup_frac     : float                – fraction of iters for LR warm-up
        clip_value      : float                – global gradient clip norm
        rx_steps_per_tx : int                  – RX inner steps per TX outer step
        patience        : int                  – early-stop window (iterations)
        rel_tol         : float                – relative improvement threshold

        Returns
        -------
        loss_history : list[float]  – normalised emulation loss per outer step
        """

        print(f"\n⚡ ADAM ENGINE STARTING: L={len(self.xi_T)} | lr={lr}")
        import math

        A_torch = torch.as_tensor(A_target, dtype=torch.complex64)
        H_torch = torch.as_tensor(H_mimo,   dtype=torch.complex64)

        device  = next(self.parameters()).device
        A_torch = A_torch.to(device)
        H_torch = H_torch.to(device)

        # Pre-compute ‖A‖_F once — used for relative error reporting only
        A_norm = torch.norm(A_torch, p='fro').item()

        # ------------------------------------------------------------------ #
        #  Optimisers                                                         #
        # ------------------------------------------------------------------ #
        opt_T = optim.Adam(self.xi_T.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        opt_R = optim.Adam(self.xi_R.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

        # ------------------------------------------------------------------ #
        #  Schedulers — CosineAnnealingLR is explicit about T_max,           #
        #  no lambda arithmetic needed, step() called once per outer iter.   #
        #  Linear warm-up handled via a multiplicative warm-up scheduler      #
        #  chained with CosineAnnealingLR via SequentialLR.                  #
        # ------------------------------------------------------------------ #
        warmup_iters = max(1, int(max_iters * warmup_frac))
        cosine_iters = max(1, max_iters - warmup_iters)
        lr_min       = lr * 1e-3

        def _make_schedulers(opt):
            warmup = optim.lr_scheduler.LinearLR(
                opt,
                start_factor=1e-3,   # begin at lr * 1e-3
                end_factor=1.0,      # reach full lr at end of warm-up
                total_iters=warmup_iters,
            )
            cosine = optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=cosine_iters,
                eta_min=lr_min,
            )
            return optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[warmup, cosine],
                milestones=[warmup_iters],
            )

        sched_T = _make_schedulers(opt_T)
        sched_R = _make_schedulers(opt_R)

        # ------------------------------------------------------------------ #
        #  Helper: normalised beta (BUG 2 fix)                               #
        #  Returns (beta_hat, Z_norm) where beta_hat = <Ẑ, A> is O(‖A‖)    #
        #  and Z_norm = ‖Z‖_F.  The true beta = beta_hat / Z_norm.          #
        # ------------------------------------------------------------------ #
        def _normalised_beta(Z):
            Z_norm = torch.norm(Z, p='fro') + 1e-12
            Z_hat  = Z / Z_norm
            beta_hat = torch.sum(torch.conj(Z_hat) * A_torch).detach()
            return beta_hat, Z_norm.detach()

        # ------------------------------------------------------------------ #
        #  Helper: one gradient step on a SIM side using normalised loss     #
        # ------------------------------------------------------------------ #
        def _step_side(opt, params, extra_loss=None):
            opt.zero_grad()
            Z, G_R = self.get_effective_cascade(H_torch)
            # Normalise cascade to decouple β from phase gradients
            Z_norm   = torch.norm(Z, p='fro') + 1e-12
            Z_hat    = Z / Z_norm
            beta_hat = torch.sum(torch.conj(Z_hat) * A_torch)
            loss_J   = torch.norm(beta_hat * Z_hat - A_torch, p='fro') ** 2
            total    = loss_J + (extra_loss(G_R) if extra_loss else 0.0)
            total.backward()
            nn.utils.clip_grad_norm_(params, max_norm=clip_value)
            opt.step()
            with torch.no_grad():
                for p in params:
                    p.copy_(p % (2.0 * math.pi))
            return loss_J.item(), (beta_hat / Z_norm).detach(), Z_norm.detach()

        noise_reg = (lambda G_R: lambda_reg * torch.norm(G_R, p='fro') ** 2) \
                    if lambda_reg > 0 else None

        # ------------------------------------------------------------------ #
        #  Main loop                                                          #
        # ------------------------------------------------------------------ #
        loss_history = []
        best_loss    = float('inf')
        best_xi_T    = [p.data.clone() for p in self.xi_T]
        best_xi_R    = [p.data.clone() for p in self.xi_R]
        beta_k       = torch.tensor(1.0 + 0j, device=device)
        Z_norm_k     = torch.tensor(1.0,      device=device)

        for k in range(max_iters):

            # --- 1. TX step ---
            _, _, _ = _step_side(opt_T, list(self.xi_T.parameters()))

            # --- 2. RX step(s) ---
            current_loss = 0.0
            for _ in range(rx_steps_per_tx):
                current_loss, beta_k, Z_norm_k = _step_side(
                    opt_R, list(self.xi_R.parameters()), noise_reg
                )

            # --- 3. Scheduler step (once per outer iteration – BUG 1 fix) ---
            sched_T.step()
            sched_R.step()

            loss_history.append(current_loss)

            # --- 4. Best-iterate tracking ---
            if current_loss < best_loss:
                best_loss = current_loss
                best_xi_T = [p.data.clone() for p in self.xi_T]
                best_xi_R = [p.data.clone() for p in self.xi_R]

            # --- 5. Logging ---
            if k % 100 == 0:
                cur_lr     = opt_T.param_groups[0]['lr']
                true_beta  = torch.abs(beta_k).item()
                print(
                    f"  [Iter {k:4d}/{max_iters}] "
                    f"Loss: {current_loss:.6f} | "
                    f"LR: {cur_lr:.2e} | "
                    f"|β|: {true_beta:.2e} | "
                    f"‖Z‖: {Z_norm_k.item():.2e}"
                )

            # --- 6. Early stopping ---
            if k >= patience:
                window_start = loss_history[-patience]
                if window_start > 1e-12:
                    rel_improvement = (window_start - current_loss) / window_start
                    if rel_improvement < rel_tol:
                        print(
                            f"  [Early stop @ iter {k}]  "
                            f"Δloss/loss = {rel_improvement:.2e} < tol {rel_tol:.2e}"
                        )
                        break

        # ------------------------------------------------------------------ #
        #  Restore best parameters                                           #
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            for p, best in zip(self.xi_T, best_xi_T):
                p.copy_(best)
            for p, best in zip(self.xi_R, best_xi_R):
                p.copy_(best)

        # ------------------------------------------------------------------ #
        #  Final report                                                       #
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            Z_f, _   = self.get_effective_cascade(H_torch)
            Z_norm_f = torch.norm(Z_f, p='fro') + 1e-12
            Z_hat_f  = Z_f / Z_norm_f
            beta_f   = torch.sum(torch.conj(Z_hat_f) * A_torch) / Z_norm_f
            fro_err  = torch.norm(beta_f * Z_f - A_torch) / A_norm
            print(
                f"  ✓ Done. Best normalised loss: {best_loss:.6f} | "
                f"Relative Frobenius error: {fro_err.item():.4f} | "
                f"‖Z‖_F: {Z_norm_f.item():.2e}\n"
            )

        return loss_history  
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

        # 2. Trainable Parameters: phase shifts xi_T and xi_R (TRANSPARENT INIT)
        self.xi_T = nn.ParameterList([
            # Creiamo un tensore di ZERI con la stessa identica forma di 'p'
            nn.Parameter(torch.zeros_like(torch.tensor(p, dtype=torch.float32))) 
            for p in sim_cpu._phase_shifts_TX
        ])
        
        self.xi_R = nn.ParameterList([
            # Creiamo un tensore di ZERI con la stessa identica forma di 'p'
            nn.Parameter(torch.zeros_like(torch.tensor(p, dtype=torch.float32))) 
            for p in sim_cpu._phase_shifts_RX
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




    def optimize_alternating(
        self,
        A_target,
        H_mimo,
        max_iters: int = 3000,
        lr: float = 0.1, # Adam richiede un LR più basso di SGD (0.05 invece di 0.5)
        lambda_reg: float = 0.0,
        warmup_frac: float = 0.05,
        clip_value: float = 1.0,
        rx_steps_per_tx: int = 2,
        patience: int = 500,
        rel_tol: float = 1e-7,
    ):
        
        import torch
        import torch.optim as optim
        import math

        print(f"\n⚡ ADAM ENGINE : L={len(self.xi_T)} | lr={lr}")

        A_torch = torch.as_tensor(A_target, dtype=torch.complex64)
        H_torch = torch.as_tensor(H_mimo,   dtype=torch.complex64)

        device  = next(self.parameters()).device
        A_torch = A_torch.to(device)
        H_torch = H_torch.to(device)

        A_norm = torch.norm(A_torch, p='fro').item()

        # 1. Optimizzatori Adam
        opt_T = optim.Adam(self.xi_T.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        opt_R = optim.Adam(self.xi_R.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

        # 2. Schedulers (il tuo setup originale con warm-up e coseno, ottimo per Adam)
        warmup_iters = max(1, int(max_iters * warmup_frac))
        cosine_iters = max(1, max_iters - warmup_iters)
        lr_min       = lr * 1e-3

        def _make_schedulers(opt):
            warmup = optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_iters)
            cosine = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_iters, eta_min=lr_min)
            return optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_iters])

        sched_T = _make_schedulers(opt_T)
        sched_R = _make_schedulers(opt_R)

        # Helper: Esegue uno step di gradiente sulla Loss Pura 
        def _step_side(opt, params, beta_fixed, is_rx=False):
            opt.zero_grad()
            Z, G_R = self.get_effective_cascade(H_torch)
            
            # NESSUNA NORMALIZZAZIONE QUI. Usiamo Z e beta_fixed puro.
            loss_J = torch.norm(beta_fixed * Z - A_torch, p='fro') ** 2
            
            total = loss_J
            if is_rx and lambda_reg > 0:
                total += lambda_reg * torch.norm(G_R, p='fro') ** 2
                
            total.backward()
            
            # Global Gradient Clipping (sostituisce p.grad /= norm di SGD)
            torch.nn.utils.clip_grad_norm_(params, max_norm=clip_value)
            opt.step()
            
            with torch.no_grad():
                for p in params:
                    p.copy_(p % (2.0 * math.pi))
                    
            return loss_J.item(), torch.norm(Z, p='fro').item()

        # 3. Main Loop
        loss_history = []
        best_loss    = float('inf')
        best_xi_T    = [p.data.clone() for p in self.xi_T]
        best_xi_R    = [p.data.clone() for p in self.xi_R]

        for k in range(max_iters):

            # --- a) Closed-form update for beta^k (Eq. 17 del paper) ---
            with torch.no_grad():
                Z_k, _ = self.get_effective_cascade(H_torch)
                num_beta = torch.sum(torch.conj(Z_k) * A_torch)
                den_beta = torch.sum(torch.conj(Z_k) * Z_k) + 1e-12
                beta_k = num_beta / den_beta

            # --- b) TX step ---
            _, _ = _step_side(opt_T, list(self.xi_T.parameters()), beta_k, is_rx=False)

            # --- c) RX step(s) ---
            current_loss = 0.0
            Z_norm_k = 0.0
            for _ in range(rx_steps_per_tx):
                current_loss, Z_norm_k = _step_side(opt_R, list(self.xi_R.parameters()), beta_k, is_rx=True)

            # --- d) Scheduler step ---
            sched_T.step()
            sched_R.step()

            loss_history.append(current_loss)

            # --- e) Best-iterate tracking ---
            if current_loss < best_loss:
                best_loss = current_loss
                best_xi_T = [p.data.clone() for p in self.xi_T]
                best_xi_R = [p.data.clone() for p in self.xi_R]

            # --- f) Logging ---
            if k % 100 == 0:
                cur_lr    = opt_T.param_groups[0]['lr']
                true_beta = torch.abs(beta_k).item()
                print(
                    f"  [Iter {k:4d}/{max_iters}] "
                    f"Loss: {current_loss:.6f} | "
                    f"LR: {cur_lr:.2e} | "
                    f"|β|: {true_beta:.2e} | "
                    f"‖Z‖: {Z_norm_k:.2e}"
                )

            # --- g) Early stopping ---
            if k >= patience:
                window_start = loss_history[-patience]
                if window_start > 1e-12:
                    rel_improvement = (window_start - current_loss) / window_start
                    if rel_improvement < rel_tol:
                        print(f"  [Early stop @ iter {k}] Δloss/loss = {rel_improvement:.2e} < tol {rel_tol:.2e}")
                        break

        # 4. Restore best parameters
        with torch.no_grad():
            for p, best in zip(self.xi_T, best_xi_T): p.copy_(best)
            for p, best in zip(self.xi_R, best_xi_R): p.copy_(best)

        # 5. Final report 
        with torch.no_grad():
            Z_f, _ = self.get_effective_cascade(H_torch)
            beta_f = torch.sum(torch.conj(Z_f) * A_torch) / (torch.sum(torch.conj(Z_f) * Z_f) + 1e-12)
            fro_err = torch.norm(beta_f * Z_f - A_torch) / A_norm
            print(
                f"  ✓ Done. Best loss: {best_loss:.6f} | "
                f"Relative Frobenius error: {fro_err.item():.4f} | "
                f"‖Z‖_F: {torch.norm(Z_f, p='fro').item():.2e}\n"
            )

        return loss_history







