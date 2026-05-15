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
    Evaluates semantic accuracy.
    Model for test phase: y_received = (beta_opt * y_signal) + noise
    """
    model.eval()
    clf.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        # Z_cascade rappresenta la propagazione fisica G2 @ H @ G1
        Z_cascade, _ = model.get_effective_cascade(H_mimo)

    for x_real_batch, labels_batch in dataloader:
        x_real_batch = x_real_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        # 1. TX Side: Complex Compression + Pre-whitening
        x_complex = complex_compressed_tensor(x_real_batch.T, device=device)
        x_white = a_inv_times_b(L_in, x_complex - mu_in)

        # 2. Over-The-Air Propagation
        # Segnale base che attraversa le metasuperfici e il canale
        y_signal = Z_cascade @ x_white 
        
        # 3. Allineamento Semantico (senza rumore per ora)
        # Il beta agisce SOLO sul segnale utile
        y_scaled = beta_opt * y_signal
        
        if snr_db is not None:
           # Calcoliamo il rumore in base alla potenza del segnale GIA' SCALATO
           sigma_v = sigma_given_snr(snr_db, y_scaled)
           
           # Generiamo il rumore bianco puro da aggiungere in fase di test
           noise = awgn(sigma_v, y_scaled.shape, device=device)
    
           # Il segnale finale ricevuto per il test è: (beta * signal) + noise
           y_received = y_scaled + noise 
        else:
           y_received = y_scaled

        # 4. RX Side: De-whitening + Decompression
        z_hat = (L_out @ y_received) + mu_out
        y_hat_real = decompress_complex_tensor(z_hat, device=device).T

        # 5. Downstream Task: Classification
        logits = clf(y_hat_real)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

    # Calcolo SNR Reale per controllo a terminale
    if snr_db is not None:
        p_signal = torch.mean(torch.abs(y_scaled)**2)
        p_noise = torch.mean(torch.abs(noise)**2)
        print(f"Target SNR: {snr_db} dB | Test SNR: {10 * torch.log10(p_signal/p_noise):.2f} dB")    

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
    Evaluates semantic accuracy for the Disjoint strategy.
    Adattato alla logica Air-SIM: y_received = (beta_global * G_R * H * G_T * x) + noise
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
        # Segnale base che attraversa le metasuperfici e il canale
        y_signal = Z_cascade @ x_white 
        
        # 3. Allineamento Semantico (senza rumore per ora)
        # Il beta_global agisce SOLO sul segnale utile
        y_scaled = beta_global * y_signal
        
        if snr_db is not None:
            # Calcoliamo il rumore in base alla potenza del segnale GIA' SCALATO
            sigma_v = sigma_given_snr(snr_db, y_scaled)
            
            # Generiamo il rumore bianco puro (NON colorato da G_R)
            noise = awgn(sigma_v, y_scaled.shape, device=device)
            
            # Il segnale finale ricevuto per il test è: (beta * signal) + noise
            y_received = y_scaled + noise
        else:
            y_received = y_scaled

        # 4. RX Side: De-whitening + Decompression
        # Nota: beta_global è già stato assorbito in y_received, quindi qui non lo moltiplichiamo di nuovo
        z_hat = (L_out @ y_received) + mu_out
        
        y_hat_real = decompress_complex_tensor(z_hat, device=device).T

        # 5. Downstream Task
        logits = clf(y_hat_real)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

    # Calcolo SNR Reale per controllo (opzionale, ma utile)
    # if snr_db is not None:
    #     p_signal = torch.mean(torch.abs(y_scaled)**2)
    #     p_noise = torch.mean(torch.abs(noise)**2)
    #     print(f"Target SNR: {snr_db} dB | Test SNR: {10 * torch.log10(p_signal/p_noise):.2f} dB")

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    return accuracy


