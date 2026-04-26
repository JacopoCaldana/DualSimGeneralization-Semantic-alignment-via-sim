 #HELPING STORE FUNCTIONING FUNCTIONS WHILE I TRY NEW METHODS
 
 
 
 
 def optimize_alternating(self, A_target, H_mimo, max_iters, lr=0.5, lambda_reg=0):
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




def run_sim_configuration(
    L, M_int, A_target, H_mimo, snr_list, 
    dm_task, clf, L_in, mu_in, L_out, mu_out, device, 
    max_iters=800, lr=0.5
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




def optimize_alternating(self, A_target, H_mimo, max_iters=1000, lr=0.8, momentum=0.9, lambda_reg=0):
        """
        Alternating Optimization Algorithm per SUPER-CONVERGENZA.
        Adatta AUTMATICAMENTE il Learning Rate alla profondità dei layer.
        """
        import torch.optim as optim

        A_torch = torch.as_tensor(A_target, dtype=torch.complex64)
        H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64)
        
        # 1. Trova automaticamente il numero di layer (L)
        L_layers = len(self.xi_T)
        
        # 2. SCALING AGGRESSIVO PER LAYER PROFONDI
        # Usiamo l'esponente 1.5 per frenare maggiormente i sistemi molto attenuati
        dynamic_lr = lr / (L_layers ** 1.1)
        
        # 3. Inizializza SGD con il learning rate dinamico scalato
        opt_T = optim.SGD(self.xi_T.parameters(), lr=dynamic_lr/10, momentum=momentum, nesterov=True)
        opt_R = optim.SGD(self.xi_R.parameters(), lr=dynamic_lr/10, momentum=momentum, nesterov=True)
        
        # 4. Passa il dynamic_lr come picco massimo (max_lr) allo scheduler
        scheduler_T = optim.lr_scheduler.OneCycleLR(opt_T, max_lr=dynamic_lr, total_steps=max_iters, pct_start=0.15)
        scheduler_R = optim.lr_scheduler.OneCycleLR(opt_R, max_lr=dynamic_lr, total_steps=max_iters, pct_start=0.15)
        
        loss_history = []

        for k in range(max_iters):
            # --- a) Closed-form update for beta^k ---
            with torch.no_grad():
                Z_k, _ = self.get_effective_cascade(H_torch)
                num_beta = torch.sum(torch.conj(Z_k) * A_torch)
                den_beta = torch.sum(torch.conj(Z_k) * Z_k) + 1e-12
                beta_k = num_beta / den_beta

            # --- b) TX Optimization ---
            opt_T.zero_grad()
            Z_T, _ = self.get_effective_cascade(H_torch)
            loss_J_T = torch.norm(beta_k * Z_T - A_torch, p='fro')**2
            loss_J_T.backward()
            
            with torch.no_grad():
                for p in self.xi_T:
                    if p.grad is not None:
                        grad_norm = torch.norm(p.grad) + 1e-12
                        p.grad /= grad_norm 
            opt_T.step()
            scheduler_T.step() # Aggiorna il LR ad ogni singolo step

            with torch.no_grad():
                for p in self.xi_T: p.copy_(p % (2 * torch.pi))

            # --- c) RX Optimization ---
            opt_R.zero_grad()
            Z_R, G_R = self.get_effective_cascade(H_torch)
            loss_J_R = torch.norm(beta_k * Z_R - A_torch, p='fro')**2
            total_loss_R = loss_J_R + lambda_reg * torch.norm(G_R, p='fro')**2
            total_loss_R.backward()
            
            with torch.no_grad():
                for p in self.xi_R:
                    if p.grad is not None:
                        grad_norm = torch.norm(p.grad) + 1e-12
                        p.grad /= grad_norm
            opt_R.step()
            scheduler_R.step() # Aggiorna il LR ad ogni singolo step

            with torch.no_grad():
                for p in self.xi_R: p.copy_(p % (2 * torch.pi))

            current_loss = loss_J_R.item()
            loss_history.append(current_loss)
            
            # Stampiamo più spesso visto che le iterazioni sono poche
            if k % 100 == 0:
                current_lr = opt_T.param_groups[0]['lr']
                print(f"      [Iter {k:4d}/{max_iters}] Loss: {current_loss:.4f} | LR: {current_lr:.4f} | Beta Mag: {torch.abs(beta_k).item():.2e}")
            
        with torch.no_grad():
            Z_final, _ = self.get_effective_cascade(H_torch)
            beta_f = torch.sum(torch.conj(Z_final) * A_torch) / (torch.sum(torch.conj(Z_final) * Z_final) + 1e-12)
            fro_err = torch.norm(beta_f * Z_final - A_torch) / torch.norm(A_torch)
            print(f"   🏁 Superfast Optimization Done. Final Relative Error: {fro_err.item():.4f}\n")
            
        return loss_history








# 2. Trainable Parameters: phase shifts xi_T and xi_R
        self.xi_T = nn.ParameterList([
            nn.Parameter(torch.tensor(p, dtype=torch.float32)) for p in sim_cpu._phase_shifts_TX
        ])
        self.xi_R = nn.ParameterList([
            nn.Parameter(torch.tensor(p, dtype=torch.float32)) for p in sim_cpu._phase_shifts_RX
        ])                    




def run_experiment_layers(A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device, strategy_name="Linear"):
    """
    EXPERIMENT 1: Accuracy vs Number of SIM Layers (L).
    Salvataggio differenziato per strategia (Linear/PPFE).
    """
    print("\n" + "="*50)
    print(f"🚀 STARTING EXPERIMENT 1: ACCURACY vs LAYERS (L) | Strategy: {strategy_name}")
    print("="*50)
    
    #layer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    layer_list = [ 2, 5,10, 15, 20]
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
                device=device, max_iters=2000
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




################################


# ============================================================
# 1. ENVIRONMENT SETUP AND PATHS
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Absolute paths for project structure
#BASE_DIR = Path('/Users/jacopocaldana/Desktop/Università/Tesi')
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models'

# Encoder names used to build the checkpoint path
TX_NAME = "vit_small_patch16_224" 
RX_NAME = "vit_base_patch16_224" 
CLF_PATH = MODEL_PATH / "data" / "classifiers" / "cifar10" / RX_NAME / f"seed_{SEED}.ckpt"

print(f"🔍 Preliminary classifier check...")
if not CLF_PATH.exists():
    raise FileNotFoundError(f"Critical Error: Checkpoint does not exist at {CLF_PATH}. "
                            "Please verify the model folder before starting.")
print("✅ Checkpoint found. Proceeding...")


# ============================================================
# 2. DATA AND CLASSIFIER LOADING
# ============================================================
print("⏳ Loading Datamodules...")

TRAIN_SIZE_PER_CLASS = 4200
# Datamodule for Semantic Alignment (using a limited set of pilots)
dm_align = DataModuleAlignmentClassification(
    dataset="cifar10", tx_enc=TX_NAME, rx_enc=RX_NAME,      
    train_label_size=TRAIN_SIZE_PER_CLASS, method='centroid', batch_size=128, seed=SEED
)
dm_align.setup()

# Datamodule for the classification task evaluation
dm_task = DataModuleClassifier(dataset="cifar10", rx_enc=TX_NAME, batch_size=128)
dm_task.setup()

# Load pre-trained classifier to the compute device
clf = Classifier.load_from_checkpoint(CLF_PATH).to(device).eval()


# ============================================================
# 3. CALCULATION OF TARGET MATRIX A AND CHANNEL H
# ============================================================
# Complex Compression: Mapping real features to the complex domain
input_c = complex_compressed_tensor(dm_align.train_data.z_tx.T, device=device)
output_c = complex_compressed_tensor(dm_align.train_data.z_rx.T, device=device)

# Pre-whitening: Normalizing the latent spaces
input_w, L_in, mu_in = prewhiten(input_c, device=device) 
output_w, L_out, mu_out = prewhiten(output_c, device=device)

# --- ALIGNMENT STRATEGY SELECTION ---
ALIGNMENT_TYPE = 'PPFE'  # Set to 'PPFE' for Zero-Shot alignment

if ALIGNMENT_TYPE == 'Linear':
    # Supervised Ridge Regression to find the mapping between spaces
    A_target = ridge_regression(input_w, output_w, lmb=1e-3)
    print(f"✅ Target matrix A calculated (Supervised - Linear)")
else:
    # Prototype-based Parseval Frame Equalization (Zero-Shot)
    A_target = ppfe(
        input_w, output_w, 
        output_real=dm_align.train_data.z_rx, 
        n_clusters=20, n_proto=1000, seed=SEED
    )
    print(f"✅ Target matrix A calculated (Zero-Shot - PPFE)")

# Real Rayleigh Fading MIMO Channel Generation (as described in Eq. 8)
H_mimo = complex_gaussian_matrix(0, 1, size=(384, 384)).to(device)
# ----No Channel---
#H_mimo = torch.eye(384, dtype=torch.complex64, device=device)


# ============================================================
# 4. BASELINE CALCULATION (ORACLE)
# ============================================================
print("\n🔮 Calculating Baseline Accuracy (Oracle)...")
# The Oracle represents the ideal software-only performance without the SIM hardware constraints
oracle_acc = run_oracle_test(dm_task, A_target, L_in, mu_in, L_out, mu_out, clf, device)
print(f"🎯 Baseline (Ideal A_target without channel): {oracle_acc * 100:.2f}%")


# ============================================================
# 5. EXPERIMENT EXECUTION
# ============================================================
# Decomment the experiment you wish to run

# --- EXPERIMENT 1: Impact of Meta-surface Layers (L) ---
print(f"🚀 Starting Experiment 1: Layers Variation ({ALIGNMENT_TYPE})...")
data_layers = run_experiment_layers(
      A_target=A_target, H_mimo=H_mimo, dm_task=dm_task, clf=clf, 
      L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, device=device,
      strategy_name=ALIGNMENT_TYPE
)

# --- EXPERIMENT 2: Impact of Signal-to-Noise Ratio (SNR) ---
#print(f"\n 🚀 Starting Experiment 2: SNR Sweep ({ALIGNMENT_TYPE})...")
#data_snr = run_experiment_snr(
 #    A_target=A_target, H_mimo=H_mimo, dm_task=dm_task, clf=clf, 
  #   L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, device=device,
   #  strategy_name=ALIGNMENT_TYPE
#)

# --- MONO-SIM ABLATION STUDY ---
# print(f"\n 🚀 Starting Mono-SIM Ablation Study ({ALIGNMENT_TYPE})...")
# data_mono = run_experiment_1_mono_sim(
#     A_target=A_target, dm_task=dm_task, clf=clf, 
#     L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, device=device,
#     strategy_name=ALIGNMENT_TYPE
# )

# --- ASYMMETRIC RX DEPTH ---
# print(f"🚀 Starting Asymmetric RX experiment ({ALIGNMENT_TYPE})..")
# data_asym = run_experiment_rx_depth(
#     A_target=A_target, H_mimo=H_mimo, dm_task=dm_task, clf=clf, 
#     L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, device=device,
#     strategy_name=ALIGNMENT_TYPE
# )

# --- ASYMMETRIC TX DEPTH ---
# print(f"🚀 Starting Asymmetric TX experiment ({ALIGNMENT_TYPE})..")
# data_tx_var = run_experiment_tx_depth(
#     A_target=A_target, H_mimo=H_mimo, dm_task=dm_task, clf=clf, 
#     L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, device=device,
#     strategy_name=ALIGNMENT_TYPE
# )

# ============================================================
#  DISJOINT OPTIMIZATION EXPERIMENTS 
# ============================================================

# --- EXPERIMENT 1 (DISJOINT): Impact of Meta-surface Layers (L) ---
#print(f"🚀 Starting Disjoint Experiment 1: Layers Variation ({ALIGNMENT_TYPE})...")
#data_layers_disjoint = run_experiment_layers_disjoint(
 #   A_target=A_target, 
  #  H_mimo=H_mimo, 
   # dm_task=dm_task, 
   # clf=clf, 
  #  L_in=L_in, 
   # mu_in=mu_in, 
    #L_out=L_out, 
  #  mu_out=mu_out, 
   # device=device,
    #strategy_name=ALIGNMENT_TYPE
#)

# --- EXPERIMENT 2 (DISJOINT): Impact of Signal-to-Noise Ratio (SNR) ---
#print(f"\n🚀 Starting Disjoint Experiment 2: SNR Sweep ({ALIGNMENT_TYPE})...")
#data_snr_disjoint = run_experiment_snr_disjoint(
#    A_target=A_target, 
 #   H_mimo=H_mimo, 
  #  dm_task=dm_task, 
   # clf=clf, 
   # L_in=L_in, 
  #  mu_in=mu_in, 
   # L_out=L_out, 
   # mu_out=mu_out, 
    #device=device,
    #strategy_name=ALIGNMENT_TYPE
#)


# ============================================================
# 6. HYPERPARAMETER OPTIMIZATION: LR vs DEPTH (Grid Search)
# ============================================================

# Questo esperimento serve a trovare il Learning Rate ottimale per diverse
# profondità (L=2, 10, 25), evitando stalli o oscillazioni numeriche.

#print(f"\n 🔍 Starting Cross-Grid Search: Layers vs Learning Rate (Strategy: {ALIGNMENT_TYPE})...")

#grid_results = run_lr_depth_grid_search(
 #   A_target=A_target, 
  #  H_mimo=H_mimo, 
   # dm_task=dm_task, 
    #clf=clf, 
  #  L_in=L_in, mu_in=mu_in, 
   # L_out=L_out, mu_out=mu_out, 
    #device=device,
    #strategy_name=ALIGNMENT_TYPE
#)

#print(f"\n ✅ Grid Search Completed. Results saved to: grid_search_L_vs_LR_{ALIGNMENT_TYPE}.json")


####################################


def run_sim_configuration(
    L, M_int, A_target, H_mimo, snr_list, 
    dm_task, clf, L_in, mu_in, L_out, mu_out, device, 
    max_iters=3000, lr=0.3
):
    """
    Trains and evaluates a specific Dual-SIM physical configuration.
    Returns a dictionary containing accuracy results for each requested SNR.
    """
    wavelength = 0.005  # 5mm (60 GHz)
    slayer = 5 * wavelength 
    dx = wavelength / 2  #con 2=no amplification 

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

    loss_history = model.optimize_alternating(
        A_target=A_target,
        H_mimo=H_mimo,
        max_iters=3000,
        momentum=0.9,
        lr=0.3,           
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

def run_experiment_layers(A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device, strategy_name="Linear", seed=current_seed):
    import json
    
    print("\n" + "="*50)
    print(f"🚀 STARTING EXPERIMENT 1: ACCURACY vs LAYERS (L) | Strategy: {strategy_name} | Seed: {seed}")
    print("="*50)
    
    
    layer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    atoms_list = [16,32, 64] 
    snr_eval = [None]
    
    max_iters_run = 2000
    base_lr = 0.5 
    
    # --- PARAMETRI FISICI FISSI ---
    wavelength = 0.005        # 5mm (60 GHz)
    slayer = 5 * wavelength   # Distanza tra i layer (0.025)
    dx = wavelength / 2       # Spaziatura dei meta-atomi (0.0025)
    
    csv_filename = f"final_results_{strategy_name}.csv"
    json_filename = f"results_layers_{strategy_name}.json"

    # 2. RECUPERA IL VECCHIO JSON (per non sovrascriverlo)
    if os.path.exists(json_filename):
        with open(json_filename, "r") as f:
            results_layers = json.load(f)
        print("📥 JSON precedente caricato correttamente. Aggiungo il nuovo dato...")
    else:
        results_layers = {}

    # --- DATI STATICI DELLA RUN ---
    base_run_data = {
        "Dataset": "CIFAR-10",
        "Classes": 10,
        "Seed": seed,
        "Alignment Type": strategy_name,
        "Iterations": max_iters_run,
        "Simulation": "DualSIM_Superfast_L1.3",
        
        # Inserimento dei parametri fisici
        "SIM Wavelength": wavelength,
        "SIM Thickness": slayer,
        
        # Assegnazione di dx alla spaziatura della griglia
        "SIM Meta Atoms Spacing Intermediate X": dx,
        "SIM Meta Atoms Spacing Intermediate Y": dx,
        
        # Opzionale: se i layer di Input/Output hanno la stessa spaziatura, lo mappiamo anche qui
        "SIM Meta Atoms Spacing Input X": dx,
        "SIM Meta Atoms Spacing Input Y": dx,
        "SIM Meta Atoms Spacing Output X": dx,
        "SIM Meta Atoms Spacing Output Y": dx,
    }

    for M_int in atoms_list:
        results_layers[f"{M_int}x{M_int}"] = {}
        for L in layer_list:
            print(f"\n🔄 Testing Config: {M_int}x{M_int} Meta-atoms | L = {L} Layers")
            
            # Lancio della simulazione
            acc_dict, loss_history = run_sim_configuration(
                L=L, M_int=M_int, A_target=A_target, H_mimo=H_mimo, 
                snr_list=snr_eval, dm_task=dm_task, clf=clf, 
                L_in=L_in, mu_in=mu_in, L_out=L_out, mu_out=mu_out, 
                device=device, max_iters=max_iters_run
            )
            
            # Estrazione risultati
            acc_val = acc_dict["Inf"]
            final_loss = loss_history[-1] if loss_history else 0.0
            results_layers[f"{M_int}x{M_int}"][str(L)] = acc_val
            
            print(f"✅ Result: Accuracy = {acc_val:.2f}% | Final Loss = {final_loss:.4f}")
            
            # --- DATI DINAMICI DELLA RUN ---
            actual_lr = base_lr / (L ** 1.3)
            
            current_run_data = base_run_data.copy()
            current_run_data.update({
                "SIM Layers": L,
                "SIM Meta Atoms Intermediate X": M_int,
                "SIM Meta Atoms Intermediate Y": M_int,
                "SIM Learning Rate": actual_lr,
                "SIM Training Loss": final_loss,
                "Accuracy SIM Mimo": acc_val
            })
            
            # Salva la riga nel CSV
            append_result_to_csv(csv_filename, current_run_data)
            
            # Salva il backup in JSON
            with open(json_filename, "w") as f:
                json.dump(results_layers, f, indent=4)

            # ==========================================
            # 🧹 PULIZIA DELLA MEMORIA
            # ==========================================
            del acc_dict
            del loss_history
            gc.collect()
            torch.cuda.empty_cache()
            print(f"🧹 VRAM Flushed.")    
                
    print(f"\n🎯 EXPERIMENT 1 COMPLETED! Data saved to {csv_filename} and {json_filename}")
    return results_layers

def run_experiment_snr(A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device, strategy_name="Linear",seed=current_seed):
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



#####################



def optimize_alternating(self, A_target, H_mimo, max_iters=3000, lr=0.3, momentum=0.85, lambda_reg=0):
        """
        Alternating Optimization Algorithm.
        Adatta AUTOMATICAMENTE il Learning Rate alla profondità dei layer.
        """
        import torch.optim as optim
        import math

        A_torch = torch.as_tensor(A_target, dtype=torch.complex64)
        H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64)
        
        # 1. Trova automaticamente il numero di layer (L)
        L_layers = len(self.xi_T)
        dynamic_lr = lr / math.pow(L_layers, 0.25)

        #dynamic_lr=lr
        
        # 4. MULTI-STEP SCHEDULER DINAMICO
        # Se L è piccolo (es. L=4), frena presto (es. al 60%).
        # Se L è grande (es. L=16), frena molto tardi (es. al 85%).
        base_fraction = 0.5 + 0.1 * math.sqrt(L_layers) 
        
        # Limitiamo il punto di frenata tra il 60% e il 90% delle iterazioni
        m1_frac = min(max(base_fraction, 0.7), 0.9)
        m2_frac = m1_frac + 0.1 # Il secondo gradino 10% dopo il primo
        
        m1 = int(max_iters * m1_frac)
        m2 = int(max_iters * m2_frac)

        # 3. Inizializza SGD 
        opt_T = optim.SGD(self.xi_T.parameters(), lr=dynamic_lr, momentum=0.85, nesterov=True)
        opt_R = optim.SGD(self.xi_R.parameters(), lr=dynamic_lr, momentum=0.85, nesterov=True)

        scheduler_T = optim.lr_scheduler.MultiStepLR(opt_T, milestones=[m1, m2], gamma=0.1)
        scheduler_R = optim.lr_scheduler.MultiStepLR(opt_R, milestones=[m1, m2], gamma=0.1)

        loss_history = []

        for k in range(max_iters):
            # --- a) Closed-form update for beta^k ---
            with torch.no_grad():
                Z_k, _ = self.get_effective_cascade(H_torch)
                num_beta = torch.sum(torch.conj(Z_k) * A_torch)
                den_beta = torch.sum(torch.conj(Z_k) * Z_k) + 1e-12
                beta_k = num_beta / den_beta

            # --- b) TX Optimization ---
            opt_T.zero_grad()
            Z_T, _ = self.get_effective_cascade(H_torch)
            loss_J_T = torch.norm(beta_k * Z_T - A_torch, p='fro')**2
            loss_J_T.backward()
            
            with torch.no_grad():
                for p in self.xi_T:
                    if p.grad is not None:
                        # La normalizzazione impone un passo rigido.
                        # Per non saltare il minimo, lo scheduler deve portare il LR a 0.
                        grad_norm = torch.norm(p.grad) + 1e-12
                        p.grad /= grad_norm 
            opt_T.step()
            scheduler_T.step() # Aggiorna il LR (curva del coseno verso lo zero)

            with torch.no_grad():
                for p in self.xi_T: p.copy_(p % (2 * torch.pi))

            # --- c) RX Optimization ---
            opt_R.zero_grad()
            Z_R, G_R = self.get_effective_cascade(H_torch)
            loss_J_R = torch.norm(beta_k * Z_R - A_torch, p='fro')**2
            total_loss_R = loss_J_R + lambda_reg * torch.norm(G_R, p='fro')**2
            total_loss_R.backward()
            
            with torch.no_grad():
                for p in self.xi_R:
                    if p.grad is not None:
                        grad_norm = torch.norm(p.grad) + 1e-12
                        p.grad /= grad_norm
            opt_R.step()
            scheduler_R.step() 

            with torch.no_grad():
                for p in self.xi_R: p.copy_(p % (2 * torch.pi))

            current_loss = loss_J_R.item()
            loss_history.append(current_loss)
            
            if k % 100 == 0:
                current_lr = opt_T.param_groups[0]['lr']
                print(f"      [Iter {k:4d}/{max_iters}] Loss: {current_loss:.4f} | LR: {current_lr:.6f} | Beta Mag: {torch.abs(beta_k).item():.2e}")
            
        with torch.no_grad():
            Z_final, _ = self.get_effective_cascade(H_torch)
            beta_f = torch.sum(torch.conj(Z_final) * A_torch) / (torch.sum(torch.conj(Z_final) * Z_final) + 1e-12)
            fro_err = torch.norm(beta_f * Z_final - A_torch) / torch.norm(A_torch)
            print(f"   🏁 Optimization Done. Final Relative Error: {fro_err.item():.4f}\n")
            
        return loss_history





def optimize_alternating(self, A_target, H_mimo, max_iters=3000, lr=0.3, momentum=0.85, lambda_reg=0):
        """
        Alternating Optimization Algorithm.
        Adatta AUTOMATICAMENTE il Learning Rate alla profondità dei layer.
        """
        import torch.optim as optim
        import math

        A_torch = torch.as_tensor(A_target, dtype=torch.complex64)
        H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64)
        
        # 1. Trova automaticamente il numero di layer (L)
        L_layers = len(self.xi_T)
        #dynamic_lr = lr / math.pow(L_layers, 0.25)

        dynamic_lr=lr
        
        # 4. MULTI-STEP SCHEDULER DINAMICO
        # Se L è piccolo (es. L=4), frena presto (es. al 60%).
        # Se L è grande (es. L=16), frena molto tardi (es. al 85%).
        base_fraction = 0.5 + 0.1 * math.sqrt(L_layers) 
        
        # Limitiamo il punto di frenata tra il 60% e il 90% delle iterazioni
        m1_frac = min(max(base_fraction, 0.7), 0.9)
        m2_frac = m1_frac + 0.1 # Il secondo gradino 10% dopo il primo
        
        m1 = int(max_iters * m1_frac)
        m2 = int(max_iters * m2_frac)

        # 3. Inizializza SGD 
        opt_T = optim.SGD(self.xi_T.parameters(), lr=dynamic_lr, momentum=0.85, nesterov=True)
        opt_R = optim.SGD(self.xi_R.parameters(), lr=dynamic_lr, momentum=0.85, nesterov=True)

        scheduler_T = optim.lr_scheduler.MultiStepLR(opt_T, milestones=[m1, m2], gamma=0.1)
        scheduler_R = optim.lr_scheduler.MultiStepLR(opt_R, milestones=[m1, m2], gamma=0.1)

        loss_history = []

        for k in range(max_iters):
            # --- a) Closed-form update for beta^k ---
            with torch.no_grad():
                Z_k, _ = self.get_effective_cascade(H_torch)
                num_beta = torch.sum(torch.conj(Z_k) * A_torch)
                den_beta = torch.sum(torch.conj(Z_k) * Z_k) + 1e-12
                beta_k = num_beta / den_beta

            # --- b) TX Optimization ---
            opt_T.zero_grad()
            Z_T, _ = self.get_effective_cascade(H_torch)
            loss_J_T = torch.norm(beta_k * Z_T - A_torch, p='fro')**2
            loss_J_T.backward()
            
            with torch.no_grad():
                for p in self.xi_T:
                    if p.grad is not None:
                        # La normalizzazione impone un passo rigido.
                        # Per non saltare il minimo, lo scheduler deve portare il LR a 0.
                        grad_norm = torch.norm(p.grad) + 1e-12
                        p.grad /= grad_norm 
            opt_T.step()
            scheduler_T.step() # Aggiorna il LR (curva del coseno verso lo zero)

            with torch.no_grad():
                for p in self.xi_T: p.copy_(p % (2 * torch.pi))

            # --- c) RX Optimization ---
            opt_R.zero_grad()
            Z_R, G_R = self.get_effective_cascade(H_torch)
            loss_J_R = torch.norm(beta_k * Z_R - A_torch, p='fro')**2
            total_loss_R = loss_J_R + lambda_reg * torch.norm(G_R, p='fro')**2
            total_loss_R.backward()
            
            with torch.no_grad():
                for p in self.xi_R:
                    if p.grad is not None:
                        grad_norm = torch.norm(p.grad) + 1e-12
                        p.grad /= grad_norm
            opt_R.step()
            scheduler_R.step() 

            with torch.no_grad():
                for p in self.xi_R: p.copy_(p % (2 * torch.pi))

            current_loss = loss_J_R.item()
            loss_history.append(current_loss)
            
            if k % 100 == 0:
                current_lr = opt_T.param_groups[0]['lr']
                print(f"      [Iter {k:4d}/{max_iters}] Loss: {current_loss:.4f} | LR: {current_lr:.6f} | Beta Mag: {torch.abs(beta_k).item():.2e}")
            
        with torch.no_grad():
            Z_final, _ = self.get_effective_cascade(H_torch)
            beta_f = torch.sum(torch.conj(Z_final) * A_torch) / (torch.sum(torch.conj(Z_final) * Z_final) + 1e-12)
            fro_err = torch.norm(beta_f * Z_final - A_torch) / torch.norm(A_torch)
            print(f"   🏁 Optimization Done. Final Relative Error: {fro_err.item():.4f}\n")
            
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
#########################################################################################

# --- a) Closed-form update for beta^k  ---
            with torch.no_grad():
                Z_k, _ = self.get_effective_cascade(H_torch)
                num_beta = torch.sum(torch.conj(Z_k) * A_torch)
                den_beta = torch.sum(torch.conj(Z_k) * Z_k) + 1e-12
                beta_k = num_beta / den_beta

##################
### OPTIMIZER X DISJOINT SENZA MODIFICHE X LAMDA ADATTIVO °°°° 



def optimize_rx_sequential(self, H_eff, A_target, max_iters=5000, lr=0.01, lambda_reg=1e-4):
        """
        Ottimizzazione Sequenziale RX usando Adam. .
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




def run_sim_configuration_disjoint(
    L, M_int, A_target, H_mimo, snr_list, 
    dm_task, clf, L_in, mu_in, L_out, mu_out, device, 
    max_iters=5000, lr=0.1,seed=42
):
    wavelength = 0.005  
    slayer = 5 * wavelength 
    dx = wavelength / 2 

    tx_sim_cpu = MonoSIMoptimizer(
        num_layers=L, num_meta_atoms_in_x=16, num_meta_atoms_in_y=12,
        num_meta_atoms_out_x=24, num_meta_atoms_out_y=16, 
        num_meta_atoms_int_x=M_int, num_meta_atoms_int_y=M_int,  
        thickness=slayer * L, wavelength=wavelength, spacings={'in': dx, 'out': dx, 'int': dx}, verbose=False 
    )

    rx_sim_cpu = MonoSIMoptimizer(
        num_layers=L, num_meta_atoms_in_x=24, num_meta_atoms_in_y=16,   
        num_meta_atoms_out_x=24, num_meta_atoms_out_y=16, 
        num_meta_atoms_int_x=M_int, num_meta_atoms_int_y=M_int,
        thickness=slayer * L, wavelength=wavelength, spacings={'in': dx, 'out': dx, 'int': dx}, verbose=False 
    )

    tx_model = MonoSIMoptimizerTorch(tx_sim_cpu).to(device)
    rx_model = MonoSIMoptimizerTorch(rx_sim_cpu).to(device)

    # --- 1. OTTIMIZZAZIONE TX ---
    print("   -> Optimizing TX-SIM (Semantic Target)...")
    loss_history_tx, _ = tx_model.optimize_sim(target_matrix=A_target, max_iters=max_iters, lr=lr)

    # --- 2. OTTIMIZZAZIONE RX (SEQUENZIALE) ---
    print("   -> Optimizing RX-SIM (Sequential Recovery)...")
    # Il TX viene congelato e propaga il segnale nel canale
    with torch.no_grad():
        G_T_opt = tx_model.get_cascade().cpu().numpy()
        H_eff = H_mimo.cpu().numpy() @ G_T_opt

    # La RX-SIM viene addestrata per recuperare 'A' guardando il canale distorto
    loss_history_rx, _ = rx_model.optimize_rx_sequential(
        H_eff=H_eff, A_target=A_target, max_iters=max_iters, lr=lr, lambda_reg=1e-4
    )

    # --- 3. RICALIBRAZIONE GLOBALE DEL BETA ---
    with torch.no_grad():
        G_R_opt = rx_model.get_cascade().cpu().numpy()
        Z_final = G_R_opt @ H_mimo.cpu().numpy() @ G_T_opt
        
        Z_torch = torch.tensor(Z_final, dtype=torch.complex64, device=device).clone().detach()
        A_torch = torch.tensor(A_target, dtype=torch.complex64, device=device).clone().detach()
        beta_global = torch.sum(torch.conj(Z_torch) * A_torch) / (torch.sum(torch.conj(Z_torch) * Z_torch) + 1e-12)

    # --- 4. EVALUATION ---
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

    return results, loss_history_tx, loss_history_rx



def run_experiment_snr_disjoint(A_target, H_mimo, dm_task, clf, L_in, mu_in, L_out, mu_out, device, strategy_name="Linear",seed=42):
    """
    EXPERIMENT 2 (DISJOINT): Accuracy vs Signal-to-Noise Ratio (SNR).
    Salvataggio differenziato per strategia (Linear/PPFE) e approccio (Disjoint).
    """
    print("\n" + "="*50)
    print(f"🚀 STARTING DISJOINT EXP 2: ACCURACY vs SNR | Strategy: {strategy_name}")
    print("="*50)
    
    L_fixed = 10
    atoms_list = [32, 64]
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