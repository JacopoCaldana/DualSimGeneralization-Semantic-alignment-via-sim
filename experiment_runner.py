import torch
import numpy as np
import json
from pathlib import Path
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim

# --- Local Modules Import ---
from dualsim import DualSIMoptimizer, DualSIMoptimizerTorch
from inference import run_evaluation, run_evaluation_disjoint_fixed
from monosim import MonoSIMoptimizer, MonoSIMoptimizerTorch
import matplotlib.pyplot as plt
from pathlib import Path

from utils import complex_compressed_tensor, decompress_complex_tensor,get_rx_equalizer

# Set base directory for results
#BASE_DIR = Path('/Users/jacopocaldana/Desktop/Università/Tesi')
BASE_DIR = Path(__file__).resolve().parent

def plot_and_save_loss(loss_history, filename="loss_plot.png", title="Convergence"):
    """
    Plots the semantic loss history and saves it as a high-resolution image.
    Specifically designed to monitor SIM optimization stability.
    """
    # Safety check: prevent execution if the loss history is empty or None
    if not loss_history:
        print("❌ No loss data available to plot.")
        return

    # Set a clean, professional visual style for publication-ready plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Initialize the figure with standard dimensions for documentation
    plt.figure(figsize=(10, 6))
    
    # Plot the semantic loss trend over optimization iterations
    plt.plot(loss_history, color='tab:red', linewidth=1.5, label='Semantic Loss')
    
    # Set titles and axis labels with appropriate padding and font sizes
    plt.title(title, pad=15, fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Semantic Loss (log scale)', fontsize=12)
    
    # Apply a logarithmic scale to the Y-axis to clearly visualize 
    # the convergence behavior as the loss approaches zero.
    plt.yscale('log') 
    
    # Add a subtle grid for better data readability
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Save the plot with high resolution (300 DPI) and minimal white space
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # Explicitly close the figure to release memory and prevent 
    # graphical overlaps during multiple iterations in a loop.
    plt.close()
    print(f"✅ Convergence plot saved to: {filename}")

def run_sim_configuration(
    L, M_int, A_target, H_mimo, snr_list, 
    dm_task, clf, L_in, mu_in, L_out, mu_out, device, 
    max_iters=5000, lr=0.1
):
    """
    Trains and evaluates a specific Dual-SIM physical configuration.
    Returns a dictionary containing accuracy results for each requested SNR.
    """
    wavelength = 0.005  # 5mm (60 GHz)
    slayer = 5 * wavelength 
    dx = wavelength / 2 

    # 1. Physical Initialization (NumPy-based)
    sim_cpu = DualSIMoptimizer(
        num_layers_TX=L, 
        num_meta_atoms_TX_in_x=16, num_meta_atoms_TX_in_y=12,
        num_meta_atoms_TX_out_x=24, num_meta_atoms_TX_out_y=16,
        num_meta_atoms_TX_int_x=M_int, num_meta_atoms_TX_int_y=M_int,  
        thickness_TX=slayer * L,
        
        num_layers_RX=L,
        num_meta_atoms_RX_in_x=24, num_meta_atoms_RX_in_y=16,
        num_meta_atoms_RX_out_x=24, num_meta_atoms_RX_out_y=16,
        num_meta_atoms_RX_int_x=M_int, num_meta_atoms_RX_int_y=M_int,
        thickness_RX=slayer * L,
        
        wavelength=wavelength,
        spacings={'tx_in': dx, 'tx_out': dx, 'tx_int': dx, 'rx_in': dx, 'rx_out': dx, 'rx_int': dx},
        verbose=False 
    )

    # Convert to PyTorch Model
    model = DualSIMoptimizerTorch(sim_cpu).to(device)

    # 2. Alternating Optimization (Algorithm 2)
    loss_history = model.optimize_alternating(
        A_target=A_target, H_mimo=H_mimo, 
        max_iters=max_iters, lr=lr, lambda_reg=1e-4
    )

    # --- CHECK CONVERGENCE ---
    
    plot_filename = BASE_DIR / f"loss_plot_L{L}_M{M_int}.png"
    plot_title = f"Convergence: Layers={L}, Atoms={M_int}x{M_int} (iters={max_iters}, lr={lr})"
    
    plot_and_save_loss(loss_history, filename=plot_filename, title=plot_title)
    # ----------------------------------------

    # 3. Calculate Final Optimal Scaling Factor (Beta) for Inference
    with torch.no_grad():
        Z_final, _ = model.get_effective_cascade(H_mimo)
        beta_opt = torch.sum(torch.conj(Z_final) * A_target) / (torch.sum(torch.conj(Z_final) * Z_final) + 1e-12)

    # 4. Evaluation across all SNR levels 
    results = {}
    for snr in snr_list:
        acc = run_evaluation(
            model=model, dataloader=dm_task.test_dataloader(), 
            H_mimo=H_mimo, snr_db=snr, beta_opt=beta_opt, 
            L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
            clf=clf, device=device
        )
        # Use "Inf" string for JSON compatibility if SNR is None (infinite)
        snr_key = "Inf" if snr is None else str(snr)
        results[snr_key] = acc * 100

    return results, loss_history

def run_experiment_layers(A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device, strategy_name="Linear"):
    """
    EXPERIMENT 1: Accuracy vs Number of SIM Layers (L).
    Salvataggio differenziato per strategia (Linear/PPFE).
    """
    print("\n" + "="*50)
    print(f"🚀 STARTING EXPERIMENT 1: ACCURACY vs LAYERS (L) | Strategy: {strategy_name}")
    print("="*50)
    
    layer_list = [2, 5, 10, 15, 20, 25]
    atoms_list = [16, 32] 
    snr_eval = [None]
    
    results_layers = {}

    for M_int in atoms_list:
        results_layers[f"{M_int}x{M_int}"] = {}
        for L in layer_list:
            print(f"\n🔄 Testing Config: {M_int}x{M_int} Meta-atoms | L = {L} Layers")
            
            acc_dict, _ = run_sim_configuration(
                L=L, M_int=M_int, A_target=A_target, H_mimo=H_mimo, 
                snr_list=snr_eval, dm_task=dm_task, clf=clf, 
                L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
                device=device, max_iters=5000
            )
            
            acc_val = acc_dict["Inf"]
            results_layers[f"{M_int}x{M_int}"][str(L)] = acc_val
            print(f"✅ Result: Accuracy = {acc_val:.2f}%")
            
            # --- SALVATAGGIO DINAMICO ---
            filename = BASE_DIR / f"results_layers_{strategy_name}.json"
            with open(filename, "w") as f:
                json.dump(results_layers, f, indent=4)
                
    print(f"\n🎯 EXPERIMENT 1 COMPLETED! Data saved to {filename.name}")
    return results_layers

def run_experiment_snr(A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device, strategy_name="Linear"):
    """
    EXPERIMENT 2: Accuracy vs Signal-to-Noise Ratio (SNR).
    Salvataggio differenziato per strategia (Linear/PPFE).
    """
    print("\n" + "="*50)
    print(f"🚀 STARTING EXPERIMENT 2: ACCURACY vs SNR | Strategy: {strategy_name}")
    print("="*50)
    
    L_fixed = 10
    atoms_list = [16, 32]
    snr_list = [-30, -20, -10, 0, 10, 20, 30]
    
    results_snr = {}

    for M_int in atoms_list:
        print(f"\n🔄 Testing Config: {M_int}x{M_int} Meta-atoms | L = {L_fixed} (Fixed)")
        
        acc_dict, _ = run_sim_configuration(
            L=L_fixed, M_int=M_int, A_target=A_target, H_mimo=H_mimo, 
            snr_list=snr_list, dm_task=dm_task, clf=clf, 
            L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
            device=device, max_iters=5000
        )
        
        results_snr[f"{M_int}x{M_int}"] = acc_dict
        
        for snr_val, acc_val in acc_dict.items():
            print(f"  - SNR {snr_val:>3} dB : {acc_val:.2f}%")
            
        # --- SALVATAGGIO DINAMICO ---
        filename = BASE_DIR / f"results_snr_{strategy_name}.json"
        with open(filename, "w") as f:
            json.dump(results_snr, f, indent=4)
            
    print(f"\n🎯 EXPERIMENT 2 COMPLETED! Data saved to {filename.name}")
    return results_snr

def run_experiment_1_mono_sim(A_target, dm_task, clf, L_in, mu_in, L_out, mu_out, device):
    """
    EXPERIMENT 1: Ablation Study (Mono-SIM at TX only).
    H_mimo = Identity, G_R = Identity, No Noise.
    Evaluates pure Semantic Alignment capabilities of the TX Metasurface.
    """
    print("\n" + "="*50)
    print("🚀 STARTING EXPERIMENT 1: MONO-SIM ABLATION STUDY")
    print("="*50)
    
    layer_list = [1, 2, 5, 10, 15, 20]
    # For this test, we can fix the meta-atoms to 32x32 to see the layer effect clearly
    M_int = 32 
    
    wavelength = 0.005 
    slayer = 5 * wavelength 
    dx = wavelength / 2 
    
    A_torch = torch.as_tensor(A_target, dtype=torch.complex64, device=device)
    results_mono = {}

    for L in layer_list:
        print(f"\n🔄 Testing TX-Only Config: {M_int}x{M_int} Meta-atoms | L = {L} Layers")
        
        # 1. Initialize Physics (Only TX parameters actually matter here)
        sim_cpu = DualSIMoptimizer(
            num_layers_TX=L, num_meta_atoms_TX_in_x=16, num_meta_atoms_TX_in_y=12,
            num_meta_atoms_TX_out_x=24, num_meta_atoms_TX_out_y=16,
            num_meta_atoms_TX_int_x=M_int, num_meta_atoms_TX_int_y=M_int, thickness_TX=slayer * L,
            num_layers_RX=1, num_meta_atoms_RX_in_x=24, num_meta_atoms_RX_in_y=16,
            num_meta_atoms_RX_out_x=24, num_meta_atoms_RX_out_y=16,
            num_meta_atoms_RX_int_x=1, num_meta_atoms_RX_int_y=1, thickness_RX=slayer,
            wavelength=wavelength, spacings={'tx_in': dx, 'tx_out': dx, 'tx_int': dx, 'rx_in': dx, 'rx_out': dx, 'rx_int': dx}
        )
        
        model = DualSIMoptimizerTorch(sim_cpu).to(device)
        
        # 2. Custom TX-Only Optimization Loop
        opt_T = torch.optim.Adam(model.xi_T.parameters(), lr=0.1)
        
        for k in range(500):
            opt_T.zero_grad()
            
            # Pure Mono-SIM Cascade: Z = G_T (Since H=I and G_R=I)
            G_T = model._calculate_G_T()
            
            # Beta calculation
            with torch.no_grad():
                beta = torch.sum(torch.conj(G_T) * A_torch) / (torch.sum(torch.conj(G_T) * G_T) + 1e-12)
            
            # Semantic Loss (Frobenius norm)
            loss_T = torch.norm(beta * G_T - A_torch, p='fro')**2
            loss_T.backward()
            opt_T.step()
            
            # Phase projection
            with torch.no_grad():
                for p in model.xi_T: p.copy_(p % (2 * torch.pi))
                
        # 3. Custom Evaluation (No Noise, No Channel, No RX)
        clf.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            G_T_final = model._calculate_G_T()
            beta_opt = torch.sum(torch.conj(G_T_final) * A_torch) / (torch.sum(torch.conj(G_T_final) * G_T_final) + 1e-12)
            
        for x_real, labels in dm_task.test_dataloader():
            x_real, labels = x_real.to(device), labels.to(device)
            
            x_c = complex_compressed_tensor(x_real.T, device=device)
            x_w = a_inv_times_b(L_in, x_c - mu_in)
            
            # Propagation
            y_signal = G_T_final @ x_w
            
            # De-whitening & Classification
            z_hat = (L_out @ (beta_opt * y_signal)) + mu_out
            y_hat = decompress_complex_tensor(z_hat, device=device).T
            
            preds = torch.argmax(clf(y_hat), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        acc_val = (np.array(all_preds) == np.array(all_labels)).mean() * 100
        results_mono[str(L)] = acc_val
        print(f"✅ Mono-SIM Result: Accuracy = {acc_val:.2f}%")
        
        # Save results
        with open(BASE_DIR / "results_exp_1_mono.json", "w") as f:
            json.dump(results_mono, f, indent=4)
            
    return results_mono




  ########################################################################################################
  ############# ASYMMETRIC EXPERIMENTS###############
  ##################################################################

PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def run_sim_configuration_asymmetric(
    L_TX, L_RX, M_int, A_target, H_mimo, snr_list, 
    dm_task, clf, L_in, mu_in, L_out, mu_out, device, 
    max_iters=500, lr=0.1
):
    """
    Funzione universale per valutare Dual-SIM con qualsiasi numero di layer.
    """
    # Parametri fisici fissi
    wavelength = 0.005 
    slayer = 5 * wavelength 
    dx = wavelength / 2 

    # Inizializzazione Fisica Asimmetrica
    sim_cpu = DualSIMoptimizer(
        num_layers_TX=L_TX, 
        num_meta_atoms_TX_int_x=M_int, num_meta_atoms_TX_int_y=M_int,
        num_meta_atoms_TX_in_x=16, num_meta_atoms_TX_in_y=12,
        num_meta_atoms_TX_out_x=24, num_meta_atoms_TX_out_y=16,
        
        num_layers_RX=L_RX,
        num_meta_atoms_RX_int_x=M_int, num_meta_atoms_RX_int_y=M_int,
        num_meta_atoms_RX_in_x=24, num_meta_atoms_RX_in_y=16,
        num_meta_atoms_RX_out_x=24, num_meta_atoms_RX_out_y=16,
        
        wavelength=wavelength,
        spacings={'tx_in': dx, 'tx_out': dx, 'tx_int': dx, 'rx_in': dx, 'rx_out': dx, 'rx_int': dx},
        verbose=False 
    )

    model = DualSIMoptimizerTorch(sim_cpu).to(device)

    # Ottimizzazione con LR variabile
    # Assicurati che optimize_alternating accetti 'lr' come parametro
    loss_history = model.optimize_alternating(
        A_target=A_target, H_mimo=H_mimo, 
        max_iters=max_iters, lr=lr
    )

    # Salvataggio Plot di Convergenza (Nome file Universale)
    plot_filename = PLOT_DIR / f"loss_LTX{L_TX}_LRX{L_RX}_M{M_int}.png"
    plot_and_save_loss(loss_history, filename=plot_filename, title=f"TX={L_TX}, RX={L_RX} (LR={lr})")

    # Valutazione finale
    with torch.no_grad():
        Z_final, _ = model.get_effective_cascade(H_mimo)
        # Calcolo del fattore di scala ottimale beta
        beta_opt = torch.sum(torch.conj(Z_final) * A_target) / (torch.sum(torch.conj(Z_final) * Z_final) + 1e-12)

    results = {}
    for snr in snr_list:
        acc = run_evaluation(model, dm_task.test_dataloader(), H_mimo, snr, beta_opt, L_in, mu_in, L_out, mu_out, clf, device)
        results["Inf" if snr is None else str(snr)] = acc * 100

    return results, loss_history


def run_experiment_rx_depth(A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device):
    """
    ESPERIMENTO: TX Fisso (10) vs RX Variabile.
    Salva in: results_rx_depth_study.json
    """
    print("\n" + "="*50)
    print("🚀 STARTING RX-DEPTH STUDY (Fixed TX=10)")
    print("="*50)
    
    L_TX_fixed = 10
    rx_layers_list = [1, 2, 5, 10, 15, 20]
    M_int = 32
    snr_eval = [None]
    lr_custom = 0.05 # LR ridotto per convergenza profonda

    results_rx = {}

    for L_RX in rx_layers_list:
        print(f"\n🔄 Config: L_TX={L_TX_fixed} | L_RX={L_RX} | LR={lr_custom}")
        iters = 500 if L_RX <= 10 else 1000
        
        acc_dict, _ = run_sim_configuration_asymmetric(
            L_TX=L_TX_fixed, L_RX=L_RX, M_int=M_int, 
            A_target=A_target, H_mimo=H_mimo, 
            snr_list=snr_eval, dm_task=dm_task, clf=clf, 
            L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
            device=device, max_iters=iters, lr=lr_custom
        )
        
        results_rx[str(L_RX)] = acc_dict["Inf"]
        with open(BASE_DIR / "results_rx_depth_study.json", "w") as f:
            json.dump(results_rx, f, indent=4)
            
    return results_rx

def run_experiment_tx_depth(A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device):
    """
    ESPERIMENTO: RX Fisso (10) vs TX Variabile.
    Salva in: results_asymmetric_tx_study.json
    """
    print("\n" + "!"*60)
    print("💎 STARTING TX-DEPTH STUDY (Fixed RX=10)")
    print("!"*60)

    L_RX_fixed = 10
    tx_layers_list = [2, 5, 10, 15, 20]
    M_int = 32
    snr_eval = [None]
    lr_custom = 0.05

    results_tx = {}

    for L_TX in tx_layers_list:
        print(f"\n🧪 Config: L_TX={L_TX} | L_RX={L_RX_fixed} | LR={lr_custom}")
        iters = 500 if L_TX <= 10 else 1000
        
        acc_dict, _ = run_sim_configuration_asymmetric(
            L_TX=L_TX, L_RX=L_RX_fixed, M_int=M_int,
            A_target=A_target, H_mimo=H_mimo,
            snr_list=snr_eval, dm_task=dm_task, clf=clf,
            L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out,
            device=device, max_iters=iters, lr=lr_custom
        )
        
        results_tx[str(L_TX)] = acc_dict["Inf"]
        with open(BASE_DIR / "results_asymmetric_tx_study.json", "w") as f:
            json.dump(results_tx, f, indent=4)

    return results_tx


###################################################################################
######### DISJOINT EXPERIMENTS ########################
####################################################

def plot_disjoint_convergence(loss_T, loss_R, L, M_int, strategy_name):
    """Genera il grafico di convergenza per le loss TX e RX nel caso Disjoint."""
    if not loss_T or not loss_R:
        return

    plt.figure(figsize=(10, 6))
    
    # Plottiamo le due curve
    plt.plot(loss_T, color='tab:blue', linewidth=2, label='TX Semantic Loss (Target: A)')
    plt.plot(loss_R, color='tab:orange', linewidth=2, label=r'RX Zero-Forcing Loss (Target: $H^\dagger$)')
    
    plt.title(f'Disjoint Convergence (L={L}, M={M_int}x{M_int}) - {strategy_name}', pad=15, fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Frobenius Loss (log scale)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    
    # Salvataggio dinamico
    save_path = BASE_DIR / f"loss_disjoint_{strategy_name}_L{L}_M{M_int}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Grafico convergenza Disjoint salvato: {save_path.name}")



import torch
import torch.optim as optim

import torch
import torch.optim as optim


def run_sim_configuration_disjoint(
    L, M_int, A_target, H_mimo, snr_list, 
    dm_task, clf, L_in, mu_in, L_out, mu_out, device, 
    max_iters=5000, lr=0.1
):
    """
    Trains and evaluates a Disjoint Dual-SIM setup.
    TX performs Semantic Alignment, RX performs Channel Equalization.
    Includes Global Beta Recalibration to fix scale drift and ensure fair comparison.
    """
    wavelength = 0.005  # 5mm (60 GHz)
    slayer = 5 * wavelength 
    dx = wavelength / 2 

    # --- 1. Inizializzazione Fisica INDIPENDENTE ---
    # TX-SIM: Input semantico -> Antenne TX
    tx_sim_cpu = MonoSIMoptimizer(
        num_layers=L, 
        num_meta_atoms_in_x=16, num_meta_atoms_in_y=12,
        num_meta_atoms_out_x=24, num_meta_atoms_out_y=16, # N_T = 384
        num_meta_atoms_int_x=M_int, num_meta_atoms_int_y=M_int,  
        thickness=slayer * L,
        wavelength=wavelength, spacings={'in': dx, 'out': dx, 'int': dx}, verbose=False 
    )

    # RX-SIM: Antenne RX -> Output semantico
    rx_sim_cpu = MonoSIMoptimizer(
        num_layers=L,
        num_meta_atoms_in_x=24, num_meta_atoms_in_y=16,   # N_R = 384
        num_meta_atoms_out_x=24, num_meta_atoms_out_y=16, 
        num_meta_atoms_int_x=M_int, num_meta_atoms_int_y=M_int,
        thickness=slayer * L,
        wavelength=wavelength, spacings={'in': dx, 'out': dx, 'int': dx}, verbose=False 
    )

    # Convert to PyTorch Models
    tx_model = MonoSIMoptimizerTorch(tx_sim_cpu).to(device)
    rx_model = MonoSIMoptimizerTorch(rx_sim_cpu).to(device)

    # --- 2. Ottimizzazione TX (Semantic Alignment) ---
    print("   -> Optimizing TX-SIM (Semantic Target)...")
    loss_history_tx, _ = tx_model.optimize_sim(
        target_matrix=A_target, max_iters=max_iters, lr=lr
    )

    # --- 3. Ottimizzazione RX (Channel Equalization) ---
    print("   -> Optimizing RX-SIM (Zero-Forcing Target)...")
    # Generiamo il target ZF (snr_db=None) usando la funzione basata su SVD
    Q_target = get_rx_equalizer(H_mimo, snr_db=None).to(device)
    
    loss_history_rx, _ = rx_model.optimize_sim(
        target_matrix=Q_target, max_iters=max_iters, lr=lr, lambda_reg=1e-4
    )

    # --- CHECK CONVERGENCE (Plottiamo entrambe le loss) ---
    plot_filename_tx = BASE_DIR / f"loss_plot_TX_L{L}_M{M_int}.png"
    plot_filename_rx = BASE_DIR / f"loss_plot_RX_L{L}_M{M_int}.png"
    
    plot_and_save_loss(loss_history_tx, filename=plot_filename_tx, title=f"TX Convergence (L={L}, M={M_int})")
    plot_and_save_loss(loss_history_rx, filename=plot_filename_rx, title=f"RX Convergence (L={L}, M={M_int})")

    # --- 3.5 RICALIBRAZIONE DEL BETA GLOBALE (La vera fix) ---
    # Calcoliamo il beta ottimale sull'intera cascata esatta, 
    # garantendo che la scala sia perfetta per il de-whitening.
    with torch.no_grad():
        G_T_opt = tx_model.get_cascade()
        G_R_opt = rx_model.get_cascade()
        Z_final = G_R_opt @ H_mimo @ G_T_opt
        
        beta_global = torch.sum(torch.conj(Z_final) * A_target) / (torch.sum(torch.conj(Z_final) * Z_final) + 1e-12)

    # --- 4. Evaluation across all SNR levels ---
    results = {}
    for snr in snr_list:
        acc = run_evaluation_disjoint_fixed(
            tx_model=tx_model, rx_model=rx_model, dataloader=dm_task.test_dataloader(), 
            H_mimo=H_mimo, snr_db=snr, beta_global=beta_global, 
            L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
            clf=clf, device=device
        )
        snr_key = "Inf" if snr is None else str(snr)
        results[snr_key] = acc * 100

    # Ritorniamo i risultati e le loss per debug
    return results, loss_history_tx, loss_history_rx


def run_experiment_layers_disjoint(A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device, strategy_name="Linear"):
    """
    EXPERIMENT 1 (DISJOINT): Accuracy vs Number of SIM Layers (L).
    Salvataggio differenziato per strategia (Linear/PPFE) e approccio (Disjoint).
    """
    print("\n" + "="*50)
    print(f"🚀 STARTING DISJOINT EXP 1: ACCURACY vs LAYERS (L) | Strategy: {strategy_name}")
    print("="*50)
    
    layer_list = [2, 5, 10, 15, 20, 25]
    atoms_list = [16, 32] 
    snr_eval = [None]  # Testiamo senza rumore per valutare la pura capacità di emulazione
    
    results_layers = {}

    for M_int in atoms_list:
        results_layers[f"{M_int}x{M_int}"] = {}
        for L in layer_list:
            print(f"\n🔄 Testing Config: {M_int}x{M_int} Meta-atoms | L = {L} Layers")
            
            # Nota i due '_' per raccogliere entrambe le loss history (TX e RX)
            acc_dict, _, _ = run_sim_configuration_disjoint(
                L=L, M_int=M_int, A_target=A_target, H_mimo=H_mimo, 
                snr_list=snr_eval, dm_task=dm_task, clf=clf, 
                L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
                device=device, max_iters=5000, lr=0.1
            )
            
            acc_val = acc_dict["Inf"]
            results_layers[f"{M_int}x{M_int}"][str(L)] = acc_val
            print(f"✅ Result: Accuracy = {acc_val:.2f}%")
            
            # --- SALVATAGGIO DINAMICO ---
            filename = BASE_DIR / f"results_layers_disjoint_{strategy_name}.json"
            with open(filename, "w") as f:
                json.dump(results_layers, f, indent=4)
                
    print(f"\n🎯 EXPERIMENT 1 COMPLETED! Data saved to {filename.name}")
    return results_layers


def run_experiment_snr_disjoint(A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device, strategy_name="Linear"):
    """
    EXPERIMENT 2 (DISJOINT): Accuracy vs Signal-to-Noise Ratio (SNR).
    Salvataggio differenziato per strategia (Linear/PPFE) e approccio (Disjoint).
    """
    print("\n" + "="*50)
    print(f"🚀 STARTING DISJOINT EXP 2: ACCURACY vs SNR | Strategy: {strategy_name}")
    print("="*50)
    
    L_fixed = 10
    atoms_list = [16, 32]
    snr_list = [-30, -20, -10, 0, 10, 20, 30]
    
    results_snr = {}

    for M_int in atoms_list:
        print(f"\n🔄 Testing Config: {M_int}x{M_int} Meta-atoms | L = {L_fixed} (Fixed)")
        
        # Testiamo la configurazione fissa su tutto il range di SNR
        acc_dict, _, _ = run_sim_configuration_disjoint(
            L=L_fixed, M_int=M_int, A_target=A_target, H_mimo=H_mimo, 
            snr_list=snr_list, dm_task=dm_task, clf=clf, 
            L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
            device=device, max_iters=5000, lr=0.1
        )
        
        results_snr[f"{M_int}x{M_int}"] = acc_dict
        
        for snr_val, acc_val in acc_dict.items():
            print(f"  - SNR {snr_val:>3} dB : {acc_val:.2f}%")
            
        # --- SALVATAGGIO DINAMICO ---
        filename = BASE_DIR / f"results_snr_disjoint_{strategy_name}.json"
        with open(filename, "w") as f:
            json.dump(results_snr, f, indent=4)
            
    print(f"\n🎯 EXPERIMENT 2 COMPLETED! Data saved to {filename.name}")
    return results_snr