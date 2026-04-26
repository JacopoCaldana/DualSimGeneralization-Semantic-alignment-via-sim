import torch
import numpy as np
from utils import (
    complex_compressed_tensor, 
    decompress_complex_tensor, 
    a_inv_times_b, 
    sigma_given_snr, 
    awgn,
    mmse_svd_equalizer,
    get_rx_equalizer 
)

def run_evaluation(model, dataloader, H_mimo, snr_db, beta_opt, L_in, mu_in, L_out, mu_out, clf, device):
    """
    Evaluates semantic accuracy on a given test set using the Dual-SIM architecture.
    
    Args:
        model: The DualSIMoptimizerTorch (or GPU) model.
        dataloader: Test set DataLoader.
        H_mimo: The MIMO channel matrix [384, 384].
        snr_db: Signal-to-Noise Ratio in dB.
        beta_opt: Optimal scaling factor calculated during training.
        L_in, mu_in: TX whitening parameters.
        L_out, mu_out: RX whitening parameters.
        clf: The pre-trained downstream classifier.
        device: Compute device (cpu, cuda, or mps).
    """
    model.eval()
    clf.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        # Z_cascade represents the effective end-to-end operator [384, 192]
        # G_R is the receiving SIM operator [384, 384]
        Z_cascade, G_R = model.get_effective_cascade(H_mimo)
        G_T = model._calculate_G_T() 

    for x_real_batch, labels_batch in dataloader:
        # 0. Batch Preparation
        # x_real_batch for ViT-Small is typically [Batch, 384]
        x_real_batch = x_real_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        # 1. TX Side: Complex Compression (384 real -> 192 complex) + Pre-whitening
        # Transpose to [Features, Batch] for matrix multiplication
        x_complex = complex_compressed_tensor(x_real_batch.T, device=device) # Result: [192, Batch]
        
        # Mean subtraction and Cholesky inversion (Whitening)
        x_white = a_inv_times_b(L_in, x_complex - mu_in) # [192, Batch]

        # 2. Over-The-Air (OTA) Propagation
        # y_signal = [384, 192] @ [192, Batch] = [384, Batch]
        y_signal = Z_cascade @ x_white 
        
        if snr_db is not None:
           sig_at_rx = H_mimo @ G_T @ x_white
           sigma_v = sigma_given_snr(snr_db, sig_at_rx)
           noise = awgn(sigma_v, y_signal.shape, device=device)
    
        # Applichiamo beta_opt SOLO al segnale, poi aggiungiamo il rumore filtrato
        # y_received ora contiene già il segnale scalato
           y_received = (beta_opt * y_signal) + (G_R @ noise) 
        else:
           y_received = beta_opt * y_signal

        # 3. RX Side: Scaling + De-whitening (Dimension 384)
        # Reconstruction in the complex latent space [384, Batch]
        z_hat = (L_out @ (y_received)) + mu_out
        
        # Decompression: [384 complex, Batch] -> [768 real, Batch]
        # Final transpose to return to [Batch, 768] for the classifier
        y_hat_real = decompress_complex_tensor(z_hat, device=device).T

        # 4. Downstream Task: Classification
        logits = clf(y_hat_real)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

    p_signal = torch.mean(torch.abs(y_signal)**2)
    p_noise = torch.mean(torch.abs(G_R @ noise)**2)
    print(f"SNR REALE: {10 * torch.log10(p_signal/p_noise):.2f} dB")    

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    return accuracy


def run_evaluation_mmse(dataloader, H_mimo, snr_db, A_target, L_in, mu_in, L_out, mu_out, clf, device):
    """
    MMSE Benchmark: Corrects the dimensional mismatch between descriptors (192) and antennas (384)
    using traditional digital signal processing.
    """
    clf.eval()
    all_preds, all_labels = [], []
    
    # 0. MMSE Equalizer Calculation
    G_mmse, _ = mmse_svd_equalizer(H_mimo, snr_db) 

    for x_real_batch, labels_batch in dataloader:
        x_real_batch, labels_batch = x_real_batch.to(device), labels_batch.to(device)
        
        # 1. TX Side: Compression + Pre-whitening
        x_complex = complex_compressed_tensor(x_real_batch.T, device=device) # [192, Batch]
        x_white = a_inv_times_b(L_in, x_complex - mu_in)

        # 2. Dimensional Adaptation (Zero-Padding)
        # Channel H is 384x384. We extend x_white from 192 to 384 dimensions.
        padding = torch.zeros((384 - 192, x_white.shape[1]), device=device, dtype=x_white.dtype)
        x_padded = torch.cat([x_white, padding], dim=0) # Now [384, Batch]

        # 3. MIMO Channel + AWGN Noise
        y_signal = H_mimo @ x_padded 
        sigma_v = sigma_given_snr(snr_db, y_signal)
        y_received = y_signal + awgn(sigma_v, y_signal.shape, device=device)

        # 4. Digital Equalization + Cropping
        x_recovered_full = G_mmse @ y_received # Recover all 384 dimensions
        x_recovered = x_recovered_full[:192, :] # Retain only the original 192

        # 5. Digital Semantic Alignment 
        z_hat_complex = A_target @ x_recovered # [384, 192] @ [192, Batch] = [384, Batch]

        # 6. RX Side: De-whitening + Classification
        z_hat = (L_out @ z_hat_complex) + mu_out
        y_hat_real = decompress_complex_tensor(z_hat, device=device).T

        preds = torch.argmax(clf(y_hat_real), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

    return (np.array(all_preds) == np.array(all_labels)).mean()



    


import torch
import numpy as np

def run_evaluation_disjoint_fixed(tx_model, rx_model, dataloader, H_mimo, snr_db, beta_global, L_in, mu_in, L_out, mu_out, clf, device):
    """
    Evaluates semantic accuracy for the Disjoint strategy, 
    matching the exact physical layer math of the joint optimization baseline.
    """
    tx_model.eval()
    rx_model.eval()
    clf.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        G_T = tx_model.get_cascade()
        G_R = rx_model.get_cascade()
        # Formiamo la cascata effettiva
        Z_cascade = G_R @ H_mimo @ G_T

    for x_real_batch, labels_batch in dataloader:
        x_real_batch = x_real_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        # 1. TX Side: Compression + Pre-whitening
        x_complex = complex_compressed_tensor(x_real_batch.T, device=device)
        x_white = a_inv_times_b(L_in, x_complex - mu_in)

        # 2. Over-The-Air (OTA) Propagation
        # Nessun beta applicato qui! Rispettiamo la fisica originale
        y_signal = Z_cascade @ x_white 
        
        if snr_db is not None:
            # Calcolo rumore basato ESATTAMENTE sul segnale che colpisce la RX-SIM
            sig_at_rx = H_mimo @ G_T @ x_white
            sigma_v = sigma_given_snr(snr_db, sig_at_rx)
            
            noise = awgn(sigma_v, y_signal.shape, device=device)
            # Il rumore viene filtrato fisicamente dalla RX-SIM
            y_received = y_signal + (G_R @ noise)
        else:
            y_received = y_signal

        # 3. RX Side: Digital Scaling + De-whitening
        # Applichiamo il beta_global digitale in ricezione
        z_hat = (L_out @ (beta_global * y_received)) + mu_out
        
        # Decompression
        y_hat_real = decompress_complex_tensor(z_hat, device=device).T

        # 4. Downstream Task
        logits = clf(y_hat_real)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    return accuracy


