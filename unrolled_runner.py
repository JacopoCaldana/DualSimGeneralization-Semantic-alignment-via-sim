import torch
import gc
import os
import sys
from torch.utils.data import DataLoader

# Gestione dei path per importare i moduli core dalla cartella radice superiore
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dualsim import DualSIMoptimizer, DualSIMoptimizerTorch
from inference import run_evaluation  
from dataset import SemanticAlignmentDataset
from unrolled_model import DualSIMUnrolledTorch
from train_unrolled import train_dual_sim_unrolled

def make_forward_cascade_fn(sim_model_torch):
    """
    Mappa i vettori piatti alle liste di tensori richieste dal simulatore fisico.
    Versione blindata per forzare PyTorch a mantenere attivo il grafo dei gradienti.
    """
    shapes_T = [p.shape for p in sim_model_torch.xi_T]
    sizes_T = [p.numel() for p in sim_model_torch.xi_T]
    shapes_R = [p.shape for p in sim_model_torch.xi_R]
    sizes_R = [p.numel() for p in sim_model_torch.xi_R]

    def forward_cascade_fn(xi_T_flat, xi_R_flat, H_batch):
        # Protezione accoppiata: cloniamo e attiviamo esplicitamente i gradienti sui segmenti suddivisi
        xi_T_split = torch.split(xi_T_flat, sizes_T)
        xi_T_list = [t.clone().requires_grad_(True).reshape(s) for t, s in zip(xi_T_split, shapes_T)]
        
        xi_R_split = torch.split(xi_R_flat, sizes_R)
        xi_R_list = [t.clone().requires_grad_(True).reshape(s) for t, s in zip(xi_R_split, shapes_R)]
        
        # Esecuzione del calcolo fisico
        return sim_model_torch.get_effective_cascade_functional(H_batch, xi_T_list, xi_R_list)
    return forward_cascade_fn

def run_unrolled_experiment_layers(
    A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device, strategy_name, seed
):
    """Sweep su strati e atomi usando l'ottimizzazione Unrolled."""
    print("\n" + "="*50)
    print(f"🚀 [UNROLLED EXPERIMENT] Layers Sweep | Seed: {seed}")
    print("="*50)

    layer_list = [2, 5, 10]
    atoms_list = [16, 32]
    K_layers = 20          
    epochs = 30
    lr_unrolled = 5e-3

    # Generazione dataset sintetico locale alla Parte 2
    dataset = SemanticAlignmentDataset(num_samples=1500)
    train_loader = DataLoader(dataset, batch_size=36, shuffle=True)

    for M_int in atoms_list:
        for L in layer_list:
            print(f"\n🔄 Config Unrolled: {M_int}x{M_int} Atomi | L = {L} Strati")
            
            # Setup identico al caso analogico
            sim_cpu = DualSIMoptimizer(
                num_layers_TX=L, num_meta_atoms_TX_in_x=16, num_meta_atoms_TX_in_y=12,
                num_meta_atoms_TX_out_x=24, num_meta_atoms_TX_out_y=16,
                num_meta_atoms_TX_int_x=M_int, num_meta_atoms_TX_int_y=M_int,  
                thickness_TX=(5 * 0.005) * L,
                num_layers_RX=L, num_meta_atoms_RX_in_x=24, num_meta_atoms_RX_in_y=16,
                num_meta_atoms_RX_out_x=24, num_meta_atoms_RX_out_y=16,
                num_meta_atoms_RX_int_x=M_int, num_meta_atoms_RX_int_y=M_int,
                thickness_RX=(5 * 0.005) * L,
                wavelength=0.005, spacings=None, verbose=False 
            )
            physical_model = DualSIMoptimizerTorch(sim_cpu).to(device)
            physical_model.eval()

            d_T = sum(p.numel() for p in physical_model.xi_T)
            d_R = sum(p.numel() for p in physical_model.xi_R)
            forward_fn = make_forward_cascade_fn(physical_model)

            # 1. Istanziazione e Addestramento della Rete Unrolled per questa specifica geometria
            unrolled_model = DualSIMUnrolledTorch(d_T=d_T, d_R=d_R, K_layers=K_layers, init_gamma=1.0).to(device)
            trained_unrolled, _ = train_dual_sim_unrolled(
                model=unrolled_model, 
                dataloader=train_loader, 
                forward_cascade_fn=forward_fn, 
                epochs=epochs, 
                lr=lr_unrolled,
                clip_value=0.5
            )

            # 2. INFERENZA ZERO-SHOT SUI DATI REALI (Esecuzione differenziabile per VJP)
            trained_unrolled.eval()
            
            # I tensori fissi esterni non richiedono gradienti
            H_tensor = torch.as_tensor(H_mimo, dtype=torch.complex64).to(device)
            A_tensor = torch.as_tensor(A_target, dtype=torch.complex64).to(device)
                
            # Fuori da no_grad: le variabili di fase xi DEVONO calcolare i gradienti analitici interni per il VJP!
            std_dev = 0.01
            xi_T_init = (torch.randn(d_T, device=device) * std_dev).requires_grad_(True)
            xi_R_init = (torch.randn(d_R, device=device) * std_dev).requires_grad_(True)
                
            # Otteniamo le fasi ottime predette dalla rete
            xi_T_opt, xi_R_opt, _, beta_opt = trained_unrolled(
                xi_T_init, xi_R_init, H_tensor, A_tensor, forward_fn
            )
                
            # Congeliamo i gradienti ESCLUSIVAMENTE per l'operazione in-place di iniezione parametri
            with torch.no_grad():
                sizes_T = [p.numel() for p in physical_model.xi_T]
                sizes_R = [p.numel() for p in physical_model.xi_R]
                for p, v in zip(physical_model.xi_T, torch.split(xi_T_opt, sizes_T)): 
                    p.copy_(v.reshape(p.shape))
                for p, v in zip(physical_model.xi_R, torch.split(xi_R_opt, sizes_R)): 
                    p.copy_(v.reshape(p.shape))

            # 3. VALUTAZIONE SUL TASK DOWNSTREAM (Usa il codice nativo della Parte 1)
            acc_val = run_evaluation(
                model=physical_model, dataloader=dm_task.test_dataloader(), 
                H_mimo=H_tensor, snr_db=None, beta_opt=beta_opt, 
                L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, clf=clf, device=device
            )
            print(f"✅ [RISULTATO RETE UNROLLED] Accuracy: {acc_val * 100:.2f}%")

            # Pulizia per evitare colli di bottiglia e Out of Memory sulla GPU
            del sim_cpu, physical_model, unrolled_model, trained_unrolled
            gc.collect()
            torch.cuda.empty_cache()