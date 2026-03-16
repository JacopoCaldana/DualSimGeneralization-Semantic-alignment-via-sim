def optimize_alternating(self, A_target, H_mimo, max_iters, lr=0.5, lambda_reg=1e-4):
        """
        Implementazione 1:1 del Projected Gradient Descent Alternato dal paper.
        """
        import torch.optim as optim
        
        A_torch = torch.as_tensor(A_target, dtype=torch.complex64)
        H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64)
        
        # Gradient Descent PURO (senza momentum) come da equazione (b)
        
        opt_T = optim.SGD(self.xi_T.parameters(), lr=lr)
        opt_R = optim.SGD(self.xi_R.parameters(), lr=lr)
        
        loss_history = []

        for k in range(max_iters):
            # ==========================================
            # a) Closed-form update for \beta
            # ==========================================
            with torch.no_grad():
                Z_k, _ = self.get_effective_cascade(H_torch)
                num_beta = torch.sum(torch.conj(Z_k) * A_torch)
                den_beta = torch.sum(torch.conj(Z_k) * Z_k) + 1e-12
                beta_k = num_beta / den_beta  # Questo è \beta^k

            # ==========================================
            # b) Gradient-based update for \xi_T
            # ==========================================
            opt_T.zero_grad()
            # Forward pass per avere il grafo calcolabile su \xi_T^k
            Z_T, _ = self.get_effective_cascade(H_torch)
            
            # Loss L_J calcolata con \beta^k (costante per questo step)
            loss_J_T = torch.norm(beta_k * Z_T - A_torch, p='fro')**2
            loss_J_T.backward()
            opt_T.step() # Applica: \xi_T^k - \eta_T \nabla L_J
            
            # Proiezione \Pi_{[0, 2\pi)} per TX
            with torch.no_grad():
                for p in self.xi_T:
                    p.copy_(p % (2 * torch.pi))

            # ==========================================
            # c) Gradient-based update for \xi_R
            # ==========================================
            opt_R.zero_grad()
            # Forward pass. Essendo opt_T.step() già avvenuto, 
            # questa chiamata usa implicitamente il NUOVO \xi_T^{k+1}
            Z_R, G_R = self.get_effective_cascade(H_torch)
            
            # L_J(\xi_T^{k+1}, \xi_R^k, \beta^k) + \lambda L_N(\xi_R^k)
            # Nota: usiamo lo STESSO beta_k calcolato allo step (a) come da formula
            loss_J_R = torch.norm(beta_k * Z_R - A_torch, p='fro')**2
            loss_N = torch.norm(G_R, p='fro')**2
            total_loss_R = loss_J_R + lambda_reg * loss_N
            
            total_loss_R.backward()
            opt_R.step() # Applica: \xi_R^k - \eta_R \nabla (L_J + \lambda L_N)

            # Proiezione \Pi_{[0, 2\pi)} per RX
            with torch.no_grad():
                for p in self.xi_R:
                    p.copy_(p % (2 * torch.pi))

                for p in self.xi_R: 
                    p.copy_(p % (2 * torch.pi))    

            # ==========================================
            # Monitoraggio
            # ==========================================
            current_loss = loss_J_R.item()
            loss_history.append(current_loss)
            
            if k % 50 == 0:
                print(f"      [Iter {k:4d}/{max_iters}] Semantic Loss: {current_loss:.4f} | Beta Mag: {torch.abs(beta_k).item():.2e}")

        # --- Calcolo Errore Frobenius Finale ---
        with torch.no_grad():
            Z_final, _ = self.get_effective_cascade(H_torch)
            beta_f = torch.sum(torch.conj(Z_final) * A_torch) / (torch.sum(torch.conj(Z_final) * Z_final) + 1e-12)
            fro_err = torch.norm(beta_f * Z_final - A_torch) / torch.norm(A_torch)
            print(f"   🏁 Optimization Done. Final Relative Frobenius Error: {fro_err.item():.4f}\n")

        return loss_history  



##### ADAM+SCHEDULER ###################

def optimize_alternating(self, A_target, H_mimo, max_iters, lr=0.01, lambda_reg=1e-4):
        """
        Alternating Optimization Algorithm (Gradient-Safe Version).
        Minimizes Frobenius norm error using the expanded analytical form to prevent 
        Zero-Gradient trapping when beta -> 0.
        """
        import torch.optim as optim # Assicurati che sia importato in cima al file
        
        A_torch = torch.as_tensor(A_target, dtype=torch.complex64)
        H_torch = torch.as_tensor(H_mimo, dtype=torch.complex64)
        
        opt_T = optim.Adam(self.xi_T.parameters(), lr=lr)
        opt_R = optim.Adam(self.xi_R.parameters(), lr=lr)

        # AGGIUNTA SCHEDULER: Riduce il Learning Rate del 50% ogni 250 iterazioni
        import torch.optim.lr_scheduler as lr_scheduler
        sched_T = lr_scheduler.StepLR(opt_T, step_size=250, gamma=0.5)
        sched_R = lr_scheduler.StepLR(opt_R, step_size=250, gamma=0.5)
        
        loss_history = []

        # Pre-calcoliamo la norma quadrata di A (costante) per alleggerire il calcolo
        norm_A_sq = torch.sum(torch.conj(A_torch) * A_torch).real

        # Rimuoviamo tqdm se stampiamo a mano ogni 50 iterazioni per non sporcare i log
        for k in range(max_iters):
            
            # --- STEP 1: TX Optimization ---
            opt_T.zero_grad()
            Z_t, _ = self.get_effective_cascade(H_torch)
            
            # Calcolo sicuro: Massimizziamo l'allineamento di forma (indipendente dalla scala)
            num_T = torch.abs(torch.sum(torch.conj(Z_t) * A_torch))**2
            den_T = torch.sum(torch.conj(Z_t) * Z_t).real + 1e-12
            
            loss_J_T = norm_A_sq - (num_T / den_T)
            loss_J_T.backward()
            opt_T.step()

            # --- STEP 2: RX Optimization ---
            opt_R.zero_grad()
            Z_r, GR = self.get_effective_cascade(H_torch)
            
            num_R = torch.abs(torch.sum(torch.conj(Z_r) * A_torch))**2
            den_R = torch.sum(torch.conj(Z_r) * Z_r).real + 1e-12
            
            loss_J_R = norm_A_sq - (num_R / den_R)
            
            # Regularization per controllare l'esaltazione del rumore (non influisce sul print)
            loss_N = torch.norm(GR, p='fro')**2
            total_loss_R = loss_J_R + lambda_reg * loss_N
            total_loss_R.backward()
            opt_R.step()

            #-Aggiunta scheduler-
            sched_T.step()
            sched_R.step()
            
            # --- STEP 3: MONITORAGGIO (Il tuo codice) ---
            current_loss = loss_J_R.item()
            loss_history.append(current_loss)
            
            if k % 50 == 0:
                print(f"      [Iter {k:4d}/{max_iters}] Semantic Loss: {current_loss:.6f}")

        return loss_history  