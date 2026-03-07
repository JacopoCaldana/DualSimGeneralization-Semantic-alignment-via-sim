import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

import torch.optim as optim



class DualSIMoptimizer:
    """
    Ottimizzatore Dual Stacked Intelligent Metasurface (Dual-SIM).
    Implementa l'architettura con SIM sia al trasmettitore (TX) che al ricevitore (RX)
    per l'allineamento semantico over-the-air.
    """

    def __init__(
        self,
        # --- Parametri TX-SIM (L_T layers) ---
        num_layers_TX: int,                 # L_T 
        num_meta_atoms_TX_in_x: int,        # Dimensione semantica input (theta_bar) 
        num_meta_atoms_TX_in_y: int,
        num_meta_atoms_TX_out_x: int,       # Numero antenne TX (N_T) 
        num_meta_atoms_TX_out_y: int,
        num_meta_atoms_TX_int_x: int,       # Atomi nei layer intermedi TX
        num_meta_atoms_TX_int_y: int,
        thickness_TX: float,

        # --- Parametri RX-SIM (L_R layers) ---
        num_layers_RX: int,                 # L_R 
        num_meta_atoms_RX_in_x: int,        # Numero antenne RX (N_R) 
        num_meta_atoms_RX_in_y: int,
        num_meta_atoms_RX_out_x: int,       # Dimensione semantica output (omega_bar)
        num_meta_atoms_RX_out_y: int,
        num_meta_atoms_RX_int_x: int,       # Atomi nei layer intermedi RX
        num_meta_atoms_RX_int_y: int,
        thickness_RX: float,

        # --- Parametri Fisici comuni ---
        wavelength: float,                  # lambda 
        verbose: bool = False,
        # Spaziature (dx, dy, sx, sy) - possono essere passate come dizionario per brevità
        spacings: dict = None 
    ):
        # Validazione spessori e layer
        if num_layers_TX <= 0 or num_layers_RX <= 0:
            raise ValueError("Il numero di layer deve essere positivo per entrambe le SIM.")

        # 1. Configurazione TX-SIM
        self.L_T = num_layers_TX
        self.N_in_T = num_meta_atoms_TX_in_x * num_meta_atoms_TX_in_y    # theta_bar 
        self.N_out_T = num_meta_atoms_TX_out_x * num_meta_atoms_TX_out_y  # N_T 
        self.M_int_T = num_meta_atoms_TX_int_x * num_meta_atoms_TX_int_y
        self.slayer_TX = thickness_TX / num_layers_TX # Spaziatura tra layer TX 

        # 2. Configurazione RX-SIM
        self.L_R = num_layers_RX
        self.N_in_R = num_meta_atoms_RX_in_x * num_meta_atoms_RX_in_y    # N_R 
        self.N_out_R = num_meta_atoms_RX_out_x * num_meta_atoms_RX_out_y  # omega_bar 
        self.M_int_R = num_meta_atoms_RX_int_x * num_meta_atoms_RX_int_y
        self.slayer_RX = thickness_RX / num_layers_RX # Spaziatura tra layer RX 

        # 3. Parametri EM
        self.wavelength = wavelength
        self.kappa = 2 * np.pi / wavelength # Numero d'onda 
        self.verbose = verbose
        
        # Salvataggio spaziature (usando valori di default se non forniti)
        self.spacings = spacings or {
            'tx_in': 0.5 * wavelength, 'tx_out': 0.5 * wavelength, 'tx_int': 0.5 * wavelength,
            'rx_in': 0.5 * wavelength, 'rx_out': 0.5 * wavelength, 'rx_int': 0.5 * wavelength
        }

        # 4. Inizializzazione Fasi (xi_T e xi_R) 
        # Le fasi sono i parametri allenabili dell'architettura
        self._phase_shifts_TX = [
            np.random.uniform(0, 2 * np.pi, self.M_int_T) for _ in range(self.L_T)
        ]
        self._phase_shifts_RX = [
            np.random.uniform(0, 2 * np.pi, self.M_int_R) for _ in range(self.L_R)
        ]

        # 5. Pre-calcolo Matrici di Attenuazione Fisse (W) 
        # In una Dual-SIM, abbiamo due catene di propagazione indipendenti
    
        self.W0_TX = self._calculate_W('TX_input', 'TX_intermediate')
        self.WL_TX = self._calculate_W('TX_intermediate', 'TX_output')
        self.W_int_TX = [self._calculate_W('TX_intermediate', 'TX_intermediate') for _ in range(1, self.L_T)]

        
        self.W0_RX = self._calculate_W('RX_input', 'RX_intermediate')
        self.WL_RX = self._calculate_W('RX_intermediate', 'RX_output')
        self.W_int_RX = [self._calculate_W('RX_intermediate', 'RX_intermediate') for _ in range(1, self.L_R)]

   

    def _get_2d_coords(
        self, linear_idx: int, num_atoms_x: int
    ) -> tuple[int, int]:
        """
        Convert 1-indexed linear meta-atom index to 1-indexed (x, y) coordinates.

        Args:
            linear_idx: 1-indexed linear index
            num_atoms_x: Number of atoms in x-direction

        Returns:
            tuple of (nx, ny) as 1-indexed coordinates
        """
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
        slayer: float  # <--- AGGIUNTO: slayer specifico per TX o RX
    ) -> float:
       """
    Calcola la distanza di propagazione d_m,m' tra due meta-atomi.
    Adattata per supportare slayer_TX o slayer_RX.
       """
    # Ottieni le coordinate 2D per entrambi gli atomi
       from_x, from_y = self._get_2d_coords(from_linear_idx, from_layer_num_atoms_x)
       to_x, to_y = self._get_2d_coords(to_linear_idx, to_layer_num_atoms_x)

    # Calcola le differenze orizzontali usando la spaziatura della griglia
       dx_diff = (from_x - to_x) * spacing_x
       dy_diff = (from_y - to_y) * spacing_y

    # Calcola la distanza 3D includendo la separazione verticale specifica (slayer)
    # Questa formula implementa la componente d_m,m' nell'equazione di propagazione 
       distance = np.sqrt(dx_diff**2 + dy_diff**2 + slayer**2)
       return distance
    
    def _calculate_W(self, from_layer_type: str, to_layer_type: str) -> np.ndarray:
        """
        Calcola la matrice di attenuazione W per una coppia di layer.
        Implementa l'equazione (3) o (5) del paper.
        """
        # Mappatura dei parametri in base al lato (TX o RX) e al tipo di layer
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
        
        # Area della cella (A_cell nel paper) 
        A_cell = from_sx * from_sx 
        W = np.zeros((to_total, from_total), dtype=complex)

        for to_idx in range(1, to_total + 1):
            for from_idx in range(1, from_total + 1):
                # Calcolo distanza d_m,m' 
                d = self._calculate_propagation_distance(
                    from_nx, from_idx, to_nx, to_idx, from_sx, from_sx, slayer
                )
                
                if d > 0:
                    # Formula del paper: [W]_{m,m'} = (s_layer * A_cell / d^2) * (1/(2pi*d) - j/lambda) * e^(j*kappa*d)
                    attenuation = (slayer * A_cell / (d**2)) * \
                                  (1 / (2 * np.pi * d) - 1j / self.wavelength) * \
                                  np.exp(1j * self.kappa * d)
                    W[to_idx - 1, from_idx - 1] = attenuation
        return W
    
    def _calculate_G_T(self) -> np.ndarray:
        """Calcola G_T(xi_T) come prodotto a cascata (Eq. 4) """
        # Inizializziamo con W_0
        G_T = self.W0_TX.copy()
        
        # Moltiplicazione per ogni layer: Upsilon_l * W_l
        for l in range(self.L_T):
            # Upsilon_l è la matrice diagonale delle fasi exp(j*xi) 
            Y_l = np.diag(np.exp(1j * self._phase_shifts_TX[l]))
            G_T = Y_l @ G_T
            if l < len(self.W_int_TX):
                G_T = self.W_int_TX[l] @ G_T
        
        # Moltiplicazione finale per W_L
        return self.WL_TX @ G_T

    def _calculate_G_R(self) -> np.ndarray:
        """Calcola G_R(xi_R) come prodotto a cascata (Eq. 6) """
        G_R = self.W0_RX.copy()
        for l in range(self.L_R):
            Y_l = np.diag(np.exp(1j * self._phase_shifts_RX[l]))
            G_R = Y_l @ G_R
            if l < len(self.W_int_RX):
                G_R = self.W_int_RX[l] @ G_R
        return self.WL_RX @ G_R
    
    def calculate_effective_cascade(self, H_mimo: np.ndarray) -> np.ndarray:
        """
        Calcola l'operatore totale Z = G_R * H * G_T.
        Corrisponde all'equazione (7) del paper.
        """
        GT = self._calculate_G_T()
        GR = self._calculate_G_R()
        
        # Canale effettivo nel dominio delle onde
        Z = GR @ H_mimo @ GT
        return Z
    



    def calculate_optimal_beta(self, Z: np.ndarray, A: np.ndarray) -> complex:
        """Calcola lo scaling ottimale beta (Eq. 17) """
        z_vec = Z.flatten()
        a_vec = A.flatten()
        return np.vdot(z_vec, a_vec) / np.vdot(z_vec, z_vec)

    def get_supervised_target(self, X: np.ndarray, Y: np.ndarray, gamma: float = 1e-3) -> np.ndarray:
        """Calcola la matrice target A_L (Supervisionata, Eq. 10)"""
        # X: latent TX, Y: latent RX
        XT_H = X.conj().T
        reg = gamma * np.eye(X.shape[0])
        return Y @ XT_H @ np.linalg.inv(X @ XT_H + reg)

    def get_zeroshot_target(self, F_T: np.ndarray, F_R: np.ndarray) -> np.ndarray:
        """Calcola la matrice target A_F (Zero-shot, Eq. 12)"""
        return F_R.conj().T @ F_T





import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm

class DualSIMoptimizerTorch(nn.Module):
    def __init__(self, sim_cpu):
        """
        Eredita i parametri fisici dalla classe DualSIMoptimizer (NumPy).
        Convertiamo le matrici di propagazione pre-calcolate in Tensori Torch.
        """
        super().__init__()
        self.verbose = sim_cpu.verbose
        self.L_T = sim_cpu.L_T
        self.L_R = sim_cpu.L_R

        # 1. Convertiamo le matrici fisse W in complessi Torch 
        self.W0_T = torch.tensor(sim_cpu.W0_TX, dtype=torch.complex64)
        self.WL_T = torch.tensor(sim_cpu.WL_TX, dtype=torch.complex64)
        self.W_int_T = [torch.tensor(W, dtype=torch.complex64) for W in sim_cpu.W_int_TX]

        self.W0_R = torch.tensor(sim_cpu.W0_RX, dtype=torch.complex64)
        self.WL_R = torch.tensor(sim_cpu.WL_RX, dtype=torch.complex64)
        self.W_int_R = [torch.tensor(W, dtype=torch.complex64) for W in sim_cpu.W_int_RX]

        # 2. Parametri Ottimizzabili: Fasi xi_T e xi_R 
        # Usiamo ParameterList per rendere le fasi visibili all'ottimizzatore
        self.xi_T = nn.ParameterList([
            nn.Parameter(torch.tensor(p, dtype=torch.float32)) for p in sim_cpu._phase_shifts_TX
        ])
        self.xi_R = nn.ParameterList([
            nn.Parameter(torch.tensor(p, dtype=torch.float32)) for p in sim_cpu._phase_shifts_RX
        ])

    def _calculate_G(self, W0, WL, W_int, phases, L):
        """Calcola la matrice di trasferimento G per un lato (Eq. 4 o 6)."""
        G = W0
        for l in range(L):
            # Matrice diagonale delle fasi: exp(j * xi) 
            Y_l = torch.diag(torch.exp(1j * phases[l]))
            G = Y_l @ G
            if l < len(W_int):
                G = W_int[l] @ G
        return WL @ G

    def get_effective_cascade(self, H):
        """Calcola Z = G_R * H * G_T (Eq. 7)."""
        GT = self._calculate_G(self.W0_T, self.WL_T, self.W_int_T, self.xi_T, self.L_T)
        GR = self._calculate_G(self.W0_R, self.WL_R, self.W_int_R, self.xi_R, self.L_R)
        Z = GR @ H @ GT
        return Z, GR

    def optimize_alternating(self, A_target, H_mimo, max_iters, lr, lambda_reg=1e-4):
        """Implementazione dell'Algoritmo 2 (Ottimizzazione Alternata)."""
        A_torch = torch.tensor(A_target, dtype=torch.complex64)
        H_torch = torch.tensor(H_mimo, dtype=torch.complex64)
        
        # Ottimizzatori separati per permettere l'aggiornamento alternato 
        opt_T = optim.Adam(self.xi_T.parameters(), lr=lr)
        opt_R = optim.Adam(self.xi_R.parameters(), lr=lr)
        
        loss_history = []

        for k in tqdm(range(max_iters), disable=not self.verbose):
            # --- STEP 1: Update Beta (Forma chiusa)  ---
            with torch.no_grad():
                Z, _ = self.get_effective_cascade(H_torch)
                beta = torch.sum(torch.conj(Z) * A_torch) / torch.sum(torch.conj(Z) * Z)

            # --- STEP 2: Ottimizzazione TX (xi_T) ---
            opt_T.zero_grad()
            Z_t, _ = self.get_effective_cascade(H_torch)
            loss_J_T = torch.norm(beta * Z_t - A_torch, p='fro')**2 # Semantic Loss 
            loss_J_T.backward()
            opt_T.step()

            # --- STEP 3: Ottimizzazione RX (xi_R) + Regolarizzazione  ---
            opt_R.zero_grad()
            Z_r, GR = self.get_effective_cascade(H_torch)
            loss_J_R = torch.norm(beta * Z_r - A_torch, p='fro')**2
            loss_N = torch.norm(GR, p='fro')**2 # Noise Regularization (Eq. 14) 
            total_loss_R = loss_J_R + lambda_reg * loss_N
            total_loss_R.backward()
            opt_R.step()

            # --- STEP 4: Proiezione Fasi  ---
            with torch.no_grad():
                for p in self.xi_T: p.copy_(p % (2 * torch.pi))
                for p in self.xi_R: p.copy_(p % (2 * torch.pi))

            loss_history.append(loss_J_R.item())
            
        return loss_history