import torch
import numpy as np
from utils import (
    complex_compressed_tensor, 
    decompress_complex_tensor, 
    a_inv_times_b
)

def run_oracle_test(dm_task, A_target, L_in, mu_in, L_out, mu_out, clf, device):
    """
    Esegue il test 'Oracle': Allineamento semantico 100% Software.
    Bypassa SIM e Canale per trovare il limite superiore teorico.
    """
    print("\n🔍 --- AVVIO ORACLE CHECK (Pure Software) ---")
    clf.eval()
    all_preds, all_labels = [], [] 
    # Usiamo il test_dataloader del task (ViT-Small)
    dataloader = dm_task.test_dataloader()
    
    with torch.no_grad():
        for x_real_batch, labels_batch in dataloader:
            x_real_batch = x_real_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # 1. TRASFORMAZIONE TX (Software)
            # Portiamo i descrittori Small nello spazio complesso e sbianchiamo
            x_complex = complex_compressed_tensor(x_real_batch.T, device=device)
            x_white = a_inv_times_b(L_in, x_complex - mu_in)

            # 2. ALLINEAMENTO SEMANTICO IDEALE
            # Applichiamo la matrice A_target direttamente sui descrittori
            # Qui NON c'è rumore, NON c'è canale, NON c'è limite di fase della SIM
            z_hat_complex = A_target @ x_white 

            # 3. TRASFORMAZIONE RX (Software)
            # De-whitening e ritorno nel dominio reale per il classificatore Base
            z_hat = (L_out @ z_hat_complex) + mu_out
            y_hat_real = decompress_complex_tensor(z_hat, device=device).T

            # 4. CLASSIFICAZIONE
            logits = clf(y_hat_real)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    print(f"✅ Risultato Oracle: {accuracy*100:.2f}%")
    print("------------------------------------------")
    return accuracy

