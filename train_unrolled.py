import torch
import torch.optim as optim

def train_dual_sim_unrolled(model, dataloader, forward_cascade_fn, epochs=10, lr=1e-2, clip_value=1.0):
    device = next(model.parameters()).device
    
    # Adam gestirà da solo l'adattamento dei passi senza tagli drastici esterni
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    
    model.train()
    loss_history = []

    print(f"⚡ Avvio Training Unrolled | Epoche: {epochs} | LR Costante: {lr:.2e}")

    for epoch in range(epochs):
        total_epoch_loss = 0.0
        
        for batch_idx, (A_batch, H_batch) in enumerate(dataloader):
            A_batch = A_batch.to(device)
            H_batch = H_batch.to(device)
            batch_size = A_batch.size(0)
            
            optimizer.zero_grad()
            batch_loss = 0.0
            
            std_dev = 0.01
            xi_T_init = (torch.randn(batch_size, model.d_T, device=device) * std_dev).requires_grad_(True)
            xi_R_init = (torch.randn(batch_size, model.d_R, device=device) * std_dev).requires_grad_(True)
            
            for i in range(batch_size):
                xi_T_in = xi_T_init[i].clone()
                xi_R_in = xi_R_init[i].clone()
                
                _, _, Z_final, beta_final = model(
                    xi_T_in, xi_R_in, H_batch[i], A_batch[i], forward_cascade_fn
                )
                
                loss_i = torch.norm(beta_final * Z_final - A_batch[i], p='fro')**2
                batch_loss += loss_i
            
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            
            # --- VERIFICA DEI GRADIENTI SUI PESI NEURALI ---
            if batch_idx == 0:
                if model.W_T[0].grad is not None:
                    w_grad_norm = model.W_T[0].grad.norm().item()
                    s_grad_norm = model.S_T[0].grad.norm().item()
                    if w_grad_norm == 0.0 and s_grad_norm == 0.0:
                        print(f"🚨 [ALLERTA GRAVE Epoca {epoch+1}]: I gradienti di W e S sono ESATTAMENTE ZERO. I pesi non si aggiorneranno mai!")
                    else:
                    # Puoi commentare o lasciare attiva la stampa dei gradienti se vuoi continuare a monitorarli
                     print(f"📈 [INFO GRADIENTI Epoca {epoch+1}]: Grad Norm W: {w_grad_norm:.6f} | Grad Norm S: {s_grad_norm:.6f}")
                else:
                    print(f"🚨 [ALLERTA CRITICA Epoca {epoch+1}]: Il gradiente di W_T è None! Il grafo si è spezzato completamente.")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            optimizer.step()
            total_epoch_loss += batch_loss.item()
            
        # NESSUN SCHEDULER STEP QUI. Manteniamo il LR costante per tutta la corsa.
        avg_loss = total_epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        print(f"  [Epoca {epoch+1:2d}/{epochs}] Loss Semantica: {avg_loss:.6f}")

    return model, loss_history