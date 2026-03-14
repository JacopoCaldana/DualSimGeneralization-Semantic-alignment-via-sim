import torch
import numpy as np
import json
from pathlib import Path

# --- Local Modules Import ---
from dualsim import DualSIMoptimizer, DualSIMoptimizerTorch
from inference import run_evaluation

# Set base directory for results
BASE_DIR = Path('/Users/jacopocaldana/Desktop/Università/Tesi')

def run_sim_configuration(
    L, M_int, A_target, H_mimo, snr_list, 
    dm_task, clf, L_in, mu_in, L_out, mu_out, device, 
    max_iters=500, lr=0.1
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

def run_experiment_layers(A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device):
    """
    EXPERIMENT 1: Accuracy vs Number of SIM Layers (L).
    Evaluates how the depth of the metasurface affects semantic alignment.
    """
    print("\n" + "="*50)
    print("🚀 STARTING EXPERIMENT 1: ACCURACY vs LAYERS (L)")
    print("="*50)
    
    # Experiment parameters
    layer_list = [2, 5, 10, 15, 20, 25]
    atoms_list = [16, 32, 64] 
    snr_eval = [None]  # None represents infinite SNR (noise-free)
    
    # Results dictionary
    results_layers = {}

    for M_int in atoms_list:
        results_layers[f"{M_int}x{M_int}"] = {}
        
        for L in layer_list:
            print(f"\n🔄 Testing Config: {M_int}x{M_int} Meta-atoms | L = {L} Layers")
            
            acc_dict, _ = run_sim_configuration(
                L=L, M_int=M_int, A_target=A_target, H_mimo=H_mimo, 
                snr_list=snr_eval, dm_task=dm_task, clf=clf, 
                L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
                device=device, max_iters=500
            )
            
            # Save the result for this specific combination
            acc_val = acc_dict["Inf"]
            results_layers[f"{M_int}x{M_int}"][str(L)] = acc_val
            print(f"✅ Result: Accuracy = {acc_val:.2f}%")
            
            # Incremental save to prevent data loss in case of interruption
            with open(BASE_DIR / "results_exp_layers.json", "w") as f:
                json.dump(results_layers, f, indent=4)
                
    print("\n🎯 EXPERIMENT 1 COMPLETED! Data saved to results_exp_layers.json")
    return results_layers

def run_experiment_snr(A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device):
    """
    EXPERIMENT 2: Accuracy vs Signal-to-Noise Ratio (SNR).
    Evaluates the robustness of the Dual-SIM system under different noise conditions.
    """
    print("\n" + "="*50)
    print("🚀 STARTING EXPERIMENT 2: ACCURACY vs SNR")
    print("="*50)
    
    # Experiment parameters
    L_fixed = 10
    atoms_list = [16, 32, 64]
    snr_list = [-30, -20, -10, 0, 10, 20, 30]
    
    # Results dictionary
    results_snr = {}

    for M_int in atoms_list:
        print(f"\n🔄 Testing Config: {M_int}x{M_int} Meta-atoms | L = {L_fixed} (Fixed)")
        
        # Train once per configuration and evaluate across the entire SNR range
        acc_dict, _ = run_sim_configuration(
            L=L_fixed, M_int=M_int, A_target=A_target, H_mimo=H_mimo, 
            snr_list=snr_list, dm_task=dm_task, clf=clf, 
            L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
            device=device, max_iters=500
        )
        
        results_snr[f"{M_int}x{M_int}"] = acc_dict
        
        for snr_val, acc_val in acc_dict.items():
            print(f"  - SNR {snr_val:>3} dB : {acc_val:.2f}%")
            
        # Incremental save
        with open(BASE_DIR / "results_exp_snr.json", "w") as f:
            json.dump(results_snr, f, indent=4)
            
    print("\n🎯 EXPERIMENT 2 COMPLETED! Data saved to results_exp_snr.json")
    return results_snr