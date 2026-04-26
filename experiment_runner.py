import torch
import numpy as np
import json
import gc
import random
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



import csv
import os
from pathlib import Path

# Headers clonati esattamente dal file di Mario
CSV_HEADERS = [
    "Dataset", "Training Label Size", "Grouping", "Method", "Classes", "Seed", "Alignment Type",
    "Number Proto", "Number Clusters", "Weighted", "Accuracy No Mismatch", "Classifier Loss No Mismatch",
    "MSE Original No Mimo", "Accuracy Original No Mimo", "Classifier Loss Original No Mimo",
    "MSE Original Mimo", "Accuracy Original Mimo", "Classifier Loss Original Mimo",
    "MSE SIM No Mimo", "Accuracy SIM No Mimo", "Classifier Loss SIM No Mimo",
    "MSE SIM Mimo", "Accuracy SIM Mimo", "Classifier Loss SIM Mimo",
    "SIM Training Loss", "Receiver Model", "Transmitter Model", "Latent Real Dim",
    "Latent Complex Dim", "Lambda", "SNR [dB]", "SIM Layers", "SIM Wavelength",
    "SIM Thickness", "SIM Meta Atoms Intermediate X", "SIM Meta Atoms Intermediate Y",
    "SIM Spacing Divisor Input", "SIM Spacing Divisor Output", "SIM Spacing Divisor Intermediate",
    "SIM Meta Atoms Spacing Input X", "SIM Meta Atoms Spacing Input Y",
    "SIM Meta Atoms Spacing Output X", "SIM Meta Atoms Spacing Output Y",
    "SIM Meta Atoms Spacing Intermediate X", "SIM Meta Atoms Spacing Intermediate Y",
    "SIM Learning Rate", "Iterations", "Simulation"
]

def append_result_to_csv(csv_path, result_data):
    """Crea il CSV (se non esiste) e fa l'append sicuro di una nuova riga."""
    file_exists = os.path.isfile(csv_path)
    # Assicura che tutte le chiavi esistano, altrimenti stringa vuota
    row_to_write = {header: result_data.get(header, "") for header in CSV_HEADERS}
    
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_to_write)
        f.flush()

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
    max_iters=3000, lr=0.1, seed=42,
    lambda_base=1e-3,          #  Cambiato default per allinearsi all'optimizer
    adaptive_training=False  
):
    """
    Versione aggiornata: supporta sia il training statico (default) 
    che quello adattivo per SNR.
    """
    wavelength = 0.005  
    slayer = 5 * wavelength 
    dx = wavelength / 2  

    # 1. Inizializzazione fisica
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

    results = {}
    last_loss_history = []
    
    #  Assicuriamoci che A_target sia sul device corretto per il calcolo di beta
    A_torch = torch.as_tensor(A_target, dtype=torch.complex64).to(device)

    # --- CASO A: TRAINING STATICO ---
    if not adaptive_training:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = DualSIMoptimizerTorch(sim_cpu).to(device)

        last_loss_history = model.optimize_alternating(
            A_target=A_target, H_mimo=H_mimo, max_iters=max_iters, 
            lr=lr, lambda_base=lambda_base, snr_db=None
        )

        with torch.no_grad():
            Z_f, _ = model.get_effective_cascade(torch.as_tensor(H_mimo, dtype=torch.complex64).to(device))
            #  Uso A_torch che si trova sicuramente sul device corretto
            beta_opt = torch.sum(torch.conj(Z_f) * A_torch) / (torch.sum(torch.conj(Z_f) * Z_f) + 1e-12)

        for snr in snr_list:
            acc = run_evaluation(
                model=model, dataloader=dm_task.test_dataloader(), 
                H_mimo=H_mimo, snr_db=snr, beta_opt=beta_opt, 
                L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
                clf=clf, device=device
            )
            results[str(snr) if snr is not None else "Inf"] = acc * 100

    # --- CASO B: TRAINING ADATTIVO ---
    else:
        for snr in snr_list:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            model = DualSIMoptimizerTorch(sim_cpu).to(device)

            last_loss_history = model.optimize_alternating(
                A_target=A_target, H_mimo=H_mimo, max_iters=max_iters, 
                lr=lr, lambda_base=lambda_base, snr_db=snr
            )

            with torch.no_grad():
                Z_f, _ = model.get_effective_cascade(torch.as_tensor(H_mimo, dtype=torch.complex64).to(device))
                #  Uso A_torch che si trova sicuramente sul device corretto
                beta_opt = torch.sum(torch.conj(Z_f) * A_torch) / (torch.sum(torch.conj(Z_f) * Z_f) + 1e-12)

            acc = run_evaluation(
                model=model, dataloader=dm_task.test_dataloader(), 
                H_mimo=H_mimo, snr_db=snr, beta_opt=beta_opt, 
                L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
                clf=clf, device=device
            )
            results[str(snr) if snr is not None else "Inf"] = acc * 100

    return results, last_loss_history


def run_experiment_layers(A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device, strategy_name="Linear", seed=42):
    
    print("\n" + "="*50)
    print(f"🚀 STARTING EXPERIMENT 1: ACCURACY vs LAYERS (L) | Strategy: {strategy_name} | Seed: {seed}")
    print("="*50)
    
    layer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    #layer_list= [20]
    atoms_list = [16, 32, 64] 
    snr_eval = [None]
    
    max_iters_run = 3000
    base_lr = 0.1
    
    wavelength = 0.005
    slayer = 5 * wavelength
    dx = wavelength / 2
    
    csv_filename = f"final_results_{strategy_name}.csv"
    json_filename = f"results_layers_{strategy_name}_seed{seed}.json" # Separato per seed

    if os.path.exists(json_filename):
        with open(json_filename, "r") as f:
            results_layers = json.load(f)
    else:
        results_layers = {}

    base_run_data = {
        "Dataset": "CIFAR-10",
        "Classes": 10,
        "Seed": seed,
        "Alignment Type": strategy_name,
        "Iterations": max_iters_run,
        "Simulation": "DualSIM_Superfast_L1.3",
        "SIM Wavelength": wavelength,
        "SIM Thickness": slayer,
        "SIM Meta Atoms Spacing Intermediate X": dx,
        "SIM Meta Atoms Spacing Intermediate Y": dx,
        "SIM Meta Atoms Spacing Input X": dx,
        "SIM Meta Atoms Spacing Input Y": dx,
        "SIM Meta Atoms Spacing Output X": dx,
        "SIM Meta Atoms Spacing Output Y": dx,
    }

    for M_int in atoms_list:
        if f"{M_int}x{M_int}" not in results_layers:
            results_layers[f"{M_int}x{M_int}"] = {}
            
        for L in layer_list:
            print(f"\n🔄 Testing Config: {M_int}x{M_int} Meta-atoms | L = {L} Layers")
            
            actual_lr = base_lr / (L ** 1.3)
            
            # --- INIEZIONE DEL SEED E DEL LR ---
            acc_dict, loss_history = run_sim_configuration(
                L=L, M_int=M_int, A_target=A_target, H_mimo=H_mimo, 
                snr_list=snr_eval, dm_task=dm_task, clf=clf, 
                L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
                device=device, max_iters=max_iters_run,
                lr=actual_lr, seed=seed 
            )
            
            acc_val = acc_dict["Inf"]
            final_loss = loss_history[-1] if loss_history else 0.0
            results_layers[f"{M_int}x{M_int}"][str(L)] = acc_val
            
            print(f"✅ Result: Accuracy = {acc_val:.2f}% | Final Loss = {final_loss:.4f}")
            
            current_run_data = base_run_data.copy()
            current_run_data.update({
                "SIM Layers": L,
                "SIM Meta Atoms Intermediate X": M_int,
                "SIM Meta Atoms Intermediate Y": M_int,
                "SIM Learning Rate": actual_lr,
                "SIM Training Loss": final_loss,
                "Accuracy SIM Mimo": acc_val
            })
            
            append_result_to_csv(csv_filename, current_run_data)
            
            with open(json_filename, "w") as f:
                json.dump(results_layers, f, indent=4)

            # 🧹 PULIZIA MEMORIA
            del acc_dict, loss_history
            gc.collect()
            torch.cuda.empty_cache()
                
    print(f"\n🎯 EXPERIMENT 1 COMPLETED! Data saved to {csv_filename} and {json_filename}")
    return results_layers


def run_experiment_snr(A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device, strategy_name="Linear", seed=42):
    
    print("\n" + "="*50)
    print(f"🚀 STARTING EXPERIMENT 2: ACCURACY vs SNR | Strategy: {strategy_name} | Seed: {seed}")
    print("="*50)
    
    L_fixed = 10
    atoms_list = [32, 64]
    snr_list = [-30, -20, -10, 0, 10, 20, 30]
    
    max_iters_run = 3000
    base_lr = 0.1
    
    wavelength = 0.005
    slayer = 5 * wavelength
    dx = wavelength / 2
    
    csv_filename = f"final_results_snr_{strategy_name}.csv"
    json_filename = f"results_snr_{strategy_name}_seed{seed}.json"
    
    if os.path.exists(json_filename):
        with open(json_filename, "r") as f:
            results_snr = json.load(f)
    else:
        results_snr = {}

    base_run_data = {
        "Dataset": "CIFAR-10",
        "Classes": 10,
        "Seed": seed,
        "Alignment Type": strategy_name,
        "Iterations": max_iters_run,
        "Simulation": "DualSIM_Superfast_L1.3_SNR",
        "SIM Layers": L_fixed,
        "SIM Learning Rate": base_lr,
        "SIM Wavelength": wavelength,
        "SIM Thickness": slayer,
        "SIM Meta Atoms Spacing Intermediate X": dx,
        "SIM Meta Atoms Spacing Intermediate Y": dx,
    }

    for M_int in atoms_list:
        print(f"\n🔄 Testing Config: {M_int}x{M_int} Meta-atoms | L = {L_fixed} (Fixed)")
        
        # --- INIEZIONE DEL SEED E DEL LR ---
        acc_dict, loss_history = run_sim_configuration(
            L=L_fixed, M_int=M_int, A_target=A_target, H_mimo=H_mimo, 
            snr_list=snr_list, dm_task=dm_task, clf=clf, 
            L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
            device=device, max_iters=max_iters_run,
            lr=base_lr, seed=seed,
            adaptive_training=True # <--- QUESTO DEVE ESSERE TRUE
        )
        
        final_loss = loss_history[-1] if loss_history else 0.0
        results_snr[f"{M_int}x{M_int}"] = acc_dict
        
        # Salviamo sul CSV una riga per ogni livello di SNR testato
        for snr_val, acc_val in acc_dict.items():
            print(f"  - SNR {snr_val:>3} dB : {acc_val:.2f}%")
            
            current_run_data = base_run_data.copy()
            current_run_data.update({
                "SIM Meta Atoms Intermediate X": M_int,
                "SIM Meta Atoms Intermediate Y": M_int,
                "SIM Training Loss": final_loss,
                "SNR [dB]": snr_val if snr_val != "Inf" else "",
                "Accuracy SIM Mimo": acc_val
            })
            append_result_to_csv(csv_filename, current_run_data)
            
        with open(json_filename, "w") as f:
            json.dump(results_snr, f, indent=4)
            
        # 🧹 PULIZIA MEMORIA
        del acc_dict, loss_history
        gc.collect()
        torch.cuda.empty_cache()
            
    print(f"\n🎯 EXPERIMENT 2 COMPLETED! Data saved to {csv_filename} and {json_filename}")
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
    max_iters=5000, lr=0.1, seed=42,
    adaptive_training=False # MODIFICA 1: Flag per attivare l'MMSE specifico per SNR
):
    wavelength = 0.005  
    slayer = 5 * wavelength 
    dx = wavelength / 2 

    # Inizializzazione fisica
    tx_sim_cpu = MonoSIMoptimizer(...)
    rx_sim_cpu = MonoSIMoptimizer(...)

    tx_model = MonoSIMoptimizerTorch(tx_sim_cpu).to(device)
    
    # --- 1. OTTIMIZZAZIONE TX (Sempre fuori dal loop, è noise-blind) ---
    print("   -> Optimizing TX-SIM (Semantic Target)...")
    loss_history_tx, _ = tx_model.optimize_sim(target_matrix=A_target, max_iters=max_iters, lr=lr)

    with torch.no_grad():
        G_T_opt = tx_model.get_cascade()
        H_eff = H_mimo @ G_T_opt

    results = {}
    loss_history_rx_final = []

    # --- LOGICA DI TRAINING ---

    if adaptive_training:
        # --- MODIFICA 2: CICLO ADATTIVO (Una RX-SIM ottimizzata per ogni SNR) ---
        for snr in snr_list:
            print(f"   -> [Adaptive] Optimizing RX-SIM for SNR: {snr} dB...")
            # Reset RX-SIM per ogni SNR per garantire la massima specificità
            rx_model = MonoSIMoptimizerTorch(rx_sim_cpu).to(device)
            
            # Chiamata alla nuova funzione MMSE (usiamo snr_db invece di lambda_reg)
            loss_history_rx, beta_snr = rx_model.optimize_rx_sequential(
                H_eff=H_eff, A_target=A_target, max_iters=max_iters, lr=lr, snr_db=snr
            )
            
            # Valutazione immediata per questo SNR
            acc = run_evaluation_disjoint_fixed(
                tx_model=tx_model, rx_model=rx_model, dataloader=dm_task.test_dataloader(), 
                H_mimo=H_mimo, snr_db=snr, beta_global=beta_snr, 
                L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
                clf=clf, device=device
            )
            results[str(snr) if snr is not None else "Inf"] = acc * 100
            loss_history_rx_final = loss_history_rx # Salviamo l'ultima per il log
            
    else:
        # --- COMPORTAMENTO STANDARD (Training RX unico, non adattivo) ---
        print("   -> [Static] Optimizing RX-SIM (Sequential Recovery)...")
        rx_model = MonoSIMoptimizerTorch(rx_sim_cpu).to(device)
        # Training unico con SNR "infinito" (o None) per retrocompatibilità
        loss_history_rx_final, beta_global = rx_model.optimize_rx_sequential(
            H_eff=H_eff, A_target=A_target, max_iters=max_iters, lr=lr, snr_db=None
        )

        for snr in snr_list:
            acc = run_evaluation_disjoint_fixed(
                tx_model=tx_model, rx_model=rx_model, dataloader=dm_task.test_dataloader(), 
                H_mimo=H_mimo, snr_db=snr, beta_global=beta_global, 
                L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
                clf=clf, device=device
            )
            results[str(snr) if snr is not None else "Inf"] = acc * 100

    return results, loss_history_tx, loss_history_rx_final



def run_experiment_layers_disjoint(A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device, strategy_name="Linear", seed=42):
    import os
    import json
    import gc

    print("\n" + "="*50)
    print(f"🚀 STARTING DISJOINT EXP 1: ACCURACY vs LAYERS (L) | Strategy: {strategy_name} | Seed: {seed}")
    print("="*50)
    
    layer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    atoms_list = [16, 32] 
    snr_eval = [None]
    
    # Prepara nomi file con seed
    csv_filename = f"final_results_disjoint_{strategy_name}.csv"
    json_filename = f"results_layers_disjoint_{strategy_name}_seed{seed}.json"

    if os.path.exists(json_filename):
        with open(json_filename, "r") as f:
            results_layers = json.load(f)
    else:
        results_layers = {}

    # Dati base per CSV (stessa struttura del caso Joint)
    base_run_data = {
        "Dataset": "CIFAR-10",
        "Classes": 10,
        "Seed": seed,
        "Alignment Type": strategy_name,
        "Method": "Disjoint",
        "Iterations": 5000,
        "Simulation": "DualSIM_Disjoint",
        "SIM Wavelength": 0.005,
        "SIM Thickness": 0.025,
    }

    for M_int in atoms_list:
        if f"{M_int}x{M_int}" not in results_layers:
            results_layers[f"{M_int}x{M_int}"] = {}
            
        for L in layer_list:
            print(f"\n🔄 Testing Config: {M_int}x{M_int} Meta-atoms | L = {L} Layers")
            
            acc_dict, _, loss_rx = run_sim_configuration_disjoint(
                L=L, M_int=M_int, A_target=A_target, H_mimo=H_mimo, 
                snr_list=snr_eval, dm_task=dm_task, clf=clf, 
                L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
                device=device, max_iters=5000, lr=0.1, seed=seed
            )
            
            acc_val = acc_dict["Inf"]
            results_layers[f"{M_int}x{M_int}"][str(L)] = acc_val
            final_loss = loss_rx[-1] if loss_rx else 0.0
            
            print(f"✅ Result: Accuracy = {acc_val:.2f}%")
            
            # --- SALVATAGGIO CSV ---
            current_run_data = base_run_data.copy()
            current_run_data.update({
                "SIM Layers": L,
                "SIM Meta Atoms Intermediate X": M_int,
                "SIM Meta Atoms Intermediate Y": M_int,
                "SIM Learning Rate": 0.1,
                "SIM Training Loss": final_loss,
                "Accuracy SIM Mimo": acc_val
            })
            append_result_to_csv(csv_filename, current_run_data)
            
            # --- SALVATAGGIO JSON ---
            with open(json_filename, "w") as f:
                json.dump(results_layers, f, indent=4)

            gc.collect()
            torch.cuda.empty_cache()
                
    print(f"\n🎯 EXPERIMENT 1 COMPLETED! Data saved to {csv_filename} and {json_filename}")
    return results_layers

def run_experiment_snr_disjoint(A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device, strategy_name="Linear", seed=42):
    """
    EXPERIMENT 2 (DISJOINT): Accuracy vs Signal-to-Noise Ratio (SNR).
    Utilizza la regolarizzazione MMSE adattiva per la RX-SIM.
    """
    import os
    import json
    import gc

    print("\n" + "="*50)
    print(f"🚀 STARTING DISJOINT EXP 2: ACCURACY vs SNR | Strategy: {strategy_name} | Seed: {seed}")
    print("="*50)
    
    L_fixed = 10
    atoms_list = [32, 64] # Come definito nel tuo snippet originale
    snr_list = [-30, -20, -10, 0, 10, 20, 30]
    
    # Prepara nomi file con seed
    csv_filename = f"final_results_snr_disjoint_{strategy_name}.csv"
    json_filename = f"results_snr_disjoint_{strategy_name}_seed{seed}.json"
    
    if os.path.exists(json_filename):
        with open(json_filename, "r") as f:
            results_snr = json.load(f)
    else:
        results_snr = {}

    # Dati base per CSV
    base_run_data = {
        "Dataset": "CIFAR-10",
        "Classes": 10,
        "Seed": seed,
        "Alignment Type": strategy_name,
        "Method": "Disjoint",
        "Iterations": 5000,
        "Simulation": "DualSIM_Disjoint_MMSE_Adaptive",
        "SIM Wavelength": 0.005,
        "SIM Thickness": 0.025, # slayer (0.0025) * L (10)
    }

    for M_int in atoms_list:
        print(f"\n🔄 Testing Config: {M_int}x{M_int} Meta-atoms | L = {L_fixed} (Fixed)")
        
        # Chiamata alla configurazione con adaptive_training=True
        # Restituisce acc_dict con un valore per ogni SNR
        acc_dict, _, loss_rx = run_sim_configuration_disjoint(
            L=L_fixed, M_int=M_int, A_target=A_target, H_mimo=H_mimo, 
            snr_list=snr_list, dm_task=dm_task, clf=clf, 
            L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
            device=device, max_iters=3000, lr=0.1, seed=seed,
            adaptive_training=True # <--- ATTIVA LA LOGICA MMSE SPECIFICA PER OGNI SNR
        )
        
        results_snr[f"{M_int}x{M_int}"] = acc_dict
        final_loss = loss_rx[-1] if loss_rx else 0.0
        
        # Salviamo sul CSV una riga per ogni livello di SNR testato
        for snr_val, acc_val in acc_dict.items():
            print(f"  - SNR {snr_val:>3} dB : {acc_val:.2f}%")
            
            current_run_data = base_run_data.copy()
            current_run_data.update({
                "SIM Layers": L_fixed,
                "SIM Meta Atoms Intermediate X": M_int,
                "SIM Meta Atoms Intermediate Y": M_int,
                "SIM Learning Rate": 0.1,
                "SIM Training Loss": final_loss,
                "SNR [dB]": snr_val if snr_val != "Inf" else "",
                "Accuracy SIM Mimo": acc_val
            })
            append_result_to_csv(csv_filename, current_run_data)
            
        # --- SALVATAGGIO JSON (Aggiornato dopo ogni M_int) ---
        with open(json_filename, "w") as f:
            json.dump(results_snr, f, indent=4)
            
        # 🧹 PULIZIA MEMORIA
        gc.collect()
        torch.cuda.empty_cache()
            
    print(f"\n🎯 EXPERIMENT 2 COMPLETED! Data saved to {csv_filename} and {json_filename}")
    return results_snr


###################################


def run_grid_search_lambda(
    A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device, 
    lambda_candidates=[0.5,1],
    strategy_name="Linear", seed=42, max_iters_gs=3000
):
    """
    Esegue una Grid Search per trovare il lambda_base ottimale includendo il regime critico a -30 dB.
    
    Allineamento:
    - max_iters_gs è settato a 3000 per replicare esattamente le condizioni 
      dell'esperimento finale (run_experiment_snr), evitando bias nella scelta di lambda.
    """
    import json
    import gc
    import torch
    import numpy as np

    print("\n" + "="*80)
    print(f"🔬 GRID SEARCH: ADAPTIVE REGULARIZATION (Extended Range) | Strategy: {strategy_name}")
    print("="*80)
    
    # Parametri fissi per la ricerca
    L_fixed = 10
    M_int_fixed = 32 
    # Includiamo i regimi rumorosi per testare la massima robustezza
    snr_critici = [-30, -20, -10, 0] 
    base_lr = 0.1
    
    gs_results = {}
    best_lambda = None
    best_mean_acc = -1.0
    
    json_gs_file = f"gs_lambda_{strategy_name}_seed{seed}.json"

    for l_base in lambda_candidates:
        print(f"\n▶️ Analisi Candidato λ_base: {l_base} (Iterazioni: {max_iters_gs})")
        
        # Chiamata con adaptive_training=True per attivare lo scaling 1/SNR nell'optimizer
        acc_dict, _ = run_sim_configuration(
            L=L_fixed, M_int=M_int_fixed, A_target=A_target, H_mimo=H_mimo, 
            snr_list=snr_critici, dm_task=dm_task, clf=clf, 
            L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
            device=device, max_iters=max_iters_gs, # <-- Allineato a 3000
            lr=base_lr, seed=seed,
            lambda_base=l_base,
            adaptive_training=True 
        )
        
        # Log istantaneo delle performance sui 4 livelli
        log_str = " | ".join([f"{snr}dB: {acc:.2f}%" for snr, acc in acc_dict.items()])
        print(f"   📊 Risultati: {log_str}")
        
        current_mean_acc = np.mean(list(acc_dict.values()))
        gs_results[str(l_base)] = {
            "detailed_acc": acc_dict,
            "mean_acc": current_mean_acc
        }
        
        if current_mean_acc > best_mean_acc:
            best_mean_acc = current_mean_acc
            best_lambda = l_base

        # Salvataggio incrementale
        with open(json_gs_file, "w") as f:
            json.dump(gs_results, f, indent=4)
            
        # Pulizia memoria aggressiva dopo ogni test
        del acc_dict
        gc.collect()
        torch.cuda.empty_cache()

    # --- TABELLA RIASSUNTIVA FINALE ---
    print("\n" + "="*85)
    header = f"{'Candidato λ':<12} | {'-30 dB':<9} | {'-20 dB':<9} | {'-10 dB':<9} | {'0 dB':<9} | {'Media':<9}"
    print(header)
    print("-" * 85)
    
    for l_val, data in gs_results.items():
        det = data["detailed_acc"]
        # Helper robusto per estrarre il valore
        def gv(k): return det.get(str(k), det.get(k, 0.0))
        
        row = (f"{l_val:<12} | {gv(-30):6.2f}%  | {gv(-20):6.2f}%  | "
               f"{gv(-10):6.2f}%  | {gv(0):6.2f}%  | {data['mean_acc']:6.2f}%")
        print(row)
    
    print("="*85)
    print(f"🏆 VINCITORE: lambda_base = {best_lambda} con Accuratezza Media = {best_mean_acc:.2f}%")
    print("="*85 + "\n")
    
    return best_lambda, gs_results