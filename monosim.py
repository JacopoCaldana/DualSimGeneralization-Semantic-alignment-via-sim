import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim

class MonoSIMPhysical:
    """
    Gestisce i calcoli fisici ed elettromagnetici per una singola SIM (su CPU).
    Nessuna dipendenza da PyTorch qui dentro.
    """
    def __init__(
        self,
        num_layers: int,
        num_atoms_in_x: int, num_atoms_in_y: int,
        num_atoms_out_x: int, num_atoms_out_y: int,
        num_atoms_int_x: int, num_atoms_int_y: int,
        thickness: float,
        wavelength: float,
        spacings: dict = None,
    ):
        self.L = num_layers
        self.N_in = num_atoms_in_x * num_atoms_in_y
        self.N_out = num_atoms_out_x * num_atoms_out_y
        self.M_int = num_atoms_int_x * num_atoms_int_y
        self.slayer = thickness / num_layers
        self.wavelength = wavelength
        self.kappa = 2 * np.pi / wavelength
        
        # Default spacings if not provided
        self.spacings = spacings or {'in': 0.5 * wavelength, 'out': 0.5 * wavelength, 'int': 0.5 * wavelength}
        
        # Salviamo la griglia (nx) per calcolare le coordinate 2D
        self.nx_in = num_atoms_in_x
        self.nx_out = num_atoms_out_x
        self.nx_int = num_atoms_int_x

        # Inizializzazione Fasi (Distribuzione Normale vicina a zero per stabilità)
        std_dev = 0.01
        # Ora TUTTI i layer (L) hanno la stessa dimensione intermedia M_int
        self.phase_shifts = [
            np.random.normal(0, std_dev, self.M_int) for _ in range(self.L)
        ]

        # Pre-calcolo delle matrici di attenuazione fisse W
        print(f"Pre-calculating W matrices for {self.L} layers...")
        self.W0, self.WL, self.W_int = self._build_W_system()

    def _get_2d_coords(self, linear_idx: int, num_atoms_x: int) -> tuple[int, int]:
        my = int(np.ceil(linear_idx / num_atoms_x))
        mx = linear_idx - (my - 1) * num_atoms_x
        return mx, my

    def _calculate_propagation_distance(self, from_nx, from_idx, to_nx, to_idx, spacing) -> float:
        # Assumiamo spacing quadrato (sx = sy) come nei tuoi dizionari
        from_x, from_y = self._get_2d_coords(from_idx, from_nx)
        to_x, to_y = self._get_2d_coords(to_idx, to_nx)
        dx_diff = (from_x - to_x) * spacing
        dy_diff = (from_y - to_y) * spacing
        return np.sqrt(dx_diff**2 + dy_diff**2 + self.slayer**2)

    def _calculate_W(self, from_total, from_nx, from_spacing, to_total, to_nx) -> np.ndarray:
        A_cell = from_spacing * from_spacing 
        W = np.zeros((to_total, from_total), dtype=complex)

        for to_idx in range(1, to_total + 1):
            for from_idx in range(1, from_total + 1):
                d = self._calculate_propagation_distance(from_nx, from_idx, to_nx, to_idx, from_spacing)
                
                if d > 0:
                    # Formula del collega per l'attenuazione (più rigorosa)
                    attenuation = (A_cell * self.slayer / (2 * np.pi * d**3)) * \
                                  (1 - 1j * self.kappa * d) * \
                                  np.exp(1j * self.kappa * d)
                    W[to_idx - 1, from_idx - 1] = attenuation
        return W

    def _build_W_system(self):
        # 1. Salto iniziale (da piano Input a primo strato SIM)
        W0 = self._calculate_W(
            self.N_in, self.nx_in, self.spacings['in'], 
            self.M_int, self.nx_int
        )
        
        # 2. Salto finale (dall'ultimo strato SIM al piano Output)
        WL = self._calculate_W(
            self.M_int, self.nx_int, self.spacings['int'], 
            self.N_out, self.nx_out
        )
        
        # 3. Salti intermedi (tra i layer della SIM)
        W_int = []
        for _ in range(1, self.L):
            W_l = self._calculate_W(
                self.M_int, self.nx_int, self.spacings['int'], 
                self.M_int, self.nx_int
            )
            W_int.append(W_l)
            
        return W0, WL, W_int

import torch
import torch.nn as nn

class MonoSIMTorch(nn.Module):
    """
    Modulo PyTorch puro. 
    Riceve l'oggetto fisico, ne estrae le costanti come Buffer e crea i gradienti per le Fasi.
    """
    def __init__(self, sim_cpu: MonoSIMPhysical):
        super().__init__()
        self.L = sim_cpu.L
        
        self.register_buffer('W0', torch.tensor(sim_cpu.W0, dtype=torch.complex64))
        self.register_buffer('WL', torch.tensor(sim_cpu.WL, dtype=torch.complex64))
        
        for i, W in enumerate(sim_cpu.W_int):
            self.register_buffer(f'W_int_{i}', torch.tensor(W, dtype=torch.complex64))

         
        # Usiamo randn per permettere alla sigmoide di coprire agevolmente da 0 a 2pi
        self.phase_params = nn.ParameterList([
            nn.Parameter(torch.randn(sim_cpu.M_int, dtype=torch.float32)) for _ in range(self.L)
        ])

    def get_cascade(self) -> torch.Tensor:
        G = self.W0.clone()
        
        #  Parametrizzazione Sigmoide per gradienti stabili 
        phases = [2 * torch.pi * torch.sigmoid(p) for p in self.phase_params]
        
        Y_0 = torch.diag(torch.exp(1j * phases[0]))
        G = Y_0 @ G
        
        for l in range(1, self.L):
            W_l = getattr(self, f'W_int_{l-1}')
            Y_l = torch.diag(torch.exp(1j * phases[l]))
            
            G = W_l @ G
            G = Y_l @ G
            
        G = self.WL @ G
        return G

    def forward(self, x: torch.Tensor, beta=None) -> torch.Tensor:
        G = self.get_cascade()
        out = G @ x
        if beta is not None:
            out = beta * out
        return out