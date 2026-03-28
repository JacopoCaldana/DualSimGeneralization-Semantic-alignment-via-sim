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
