# physics_engine.py
import torch

def get_sim_transfer_matrix(xi, W_list, layer_sizes):
    """
    Calcola la matrice di trasferimento G per un SIM rispettando la shape 
    variabile di ciascun layer.
    
    Args:
        xi (torch.Tensor): Vettore piatto 1D di tutte le fasi del SIM.
        W_list (list): Lista dei buffer delle matrici di propagazione geometrica W.
        layer_sizes (list di int): Lista contenente il numero di elementi per ogni strato.
    """
    device = xi.device
    input_dim = W_list[0].shape[1]
    
    # Inizializzazione con la matrice identità sulla dimensione d'ingresso
    G = torch.eye(input_dim, dtype=torch.complex64, device=device)
    
    start_idx = 0
    for l, size in enumerate(layer_sizes):
        # Estraiamo lo slice corretto per il layer l usando la sua dimensione reale
        end_idx = start_idx + size
        xi_l = xi[start_idx:end_idx]
        start_idx = end_idx
        
        # Creazione della matrice diagonale di sfasamento Y_l (ora di dimensione corretta!)
        Y_l = torch.diag(torch.exp(1j * xi_l))
        
        # Sequenza di moltiplicazione identica al tuo modello originario
        G = Y_l @ W_list[l] @ G
        
    return G