import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def extract_monosim_specific_configs(csv_path="final_results_monosim.csv"):
    # Carichiamo il CSV
    df = pd.read_csv(csv_path)
    
    # --- APPLICAZIONE DEI FILTRI RICHIESTI ---
    # 1. Solo Seed = 42
    df = df[df['Seed'] == 42]
    
    # 2. Solo Meta-Atomi 16 e 32
    valid_atoms = [16, 32]
    df = df[df['SIM Meta Atoms Intermediate X'].isin(valid_atoms)]
    
    # 3. Solo Layer specifici
    valid_layers = [2, 5, 10, 15, 20, 25]
    df = df[df['SIM Layers'].isin(valid_layers)]
    # -----------------------------------------
    
    for strategy in ['PPFE', 'Linear']:
        # Filtriamo per la strategia corrente
        df_strat = df[df['Alignment Type'] == strategy]
        
        # Estraiamo i dati. Dato che il seed è fisso a 42, ogni riga dovrebbe essere unica,
        # ma usiamo groupby.mean() per garantire che non ci siano duplicati imprevisti nel CSV
        agg_df = df_strat.groupby(['SIM Meta Atoms Intermediate X', 'SIM Layers'])['Accuracy SIM Mimo'].mean().reset_index()
        
        results_layers = {}
        
        for M in valid_atoms:
            chiave_atomi = f"{int(M)}x{int(M)}"
            results_layers[chiave_atomi] = {}
            
            # Filtriamo i dati per i meta-atomi correnti
            df_M = agg_df[agg_df['SIM Meta Atoms Intermediate X'] == M]
            
            for index, row in df_M.iterrows():
                layer_val = str(int(row['SIM Layers']))
                acc_val = row['Accuracy SIM Mimo'] * 100 # Convertiamo in percentuale
                results_layers[chiave_atomi][layer_val] = acc_val
                
        # Salvataggio nel file JSON
        out_file = BASE_DIR / f"results_layers_monosim_{strategy}.json"
        with open(out_file, "w") as f:
            json.dump(results_layers, f, indent=4)
            
        print(f"✅ Estrazione completata: {out_file.name} (Seed 42, {valid_atoms}, L={valid_layers})")

# Lancia la funzione
extract_monosim_specific_configs()


def extract_monosim_snr_configs(csv_path="final_results_snr.csv"):
    # Carichiamo il CSV
    df = pd.read_csv(csv_path)
    
    # --- APPLICAZIONE DEI FILTRI RICHIESTI ---
    # 1. Solo Seed = 42
    df = df[df['Seed'] == 42]
    
    # 2. Solo Layer = 10 (l'esperimento SNR è fissato a L=10)
    df = df[df['SIM Layers'] == 10]
    
    # 3. Solo Meta-Atomi 16 e 32
    valid_atoms = [16, 32]
    df = df[df['SIM Meta Atoms Intermediate X'].isin(valid_atoms)]
    # -----------------------------------------
    
    for strategy in ['PPFE', 'Linear']:
        # Filtriamo per la strategia corrente
        df_strat = df[df['Alignment Type'] == strategy]
        
        # Raggruppiamo per Meta-Atomi e livello di SNR
        agg_df = df_strat.groupby(['SIM Meta Atoms Intermediate X', 'SNR [dB]'])['Accuracy SIM Mimo'].mean().reset_index()
        
        results_snr = {}
        
        for M in valid_atoms:
            chiave_atomi = f"{int(M)}x{int(M)}"
            results_snr[chiave_atomi] = {}
            
            # Filtriamo i dati per i meta-atomi correnti
            df_M = agg_df[agg_df['SIM Meta Atoms Intermediate X'] == M]
            
            # Ordiniamo per SNR (dal rumore peggiore a quello migliore)
            df_M = df_M.sort_values(by='SNR [dB]')
            
            for index, row in df_M.iterrows():
                snr_val = str(int(row['SNR [dB]']))
                acc_val = row['Accuracy SIM Mimo'] * 100 # Convertiamo in percentuale (es. 0.85 -> 85.0)
                results_snr[chiave_atomi][snr_val] = acc_val
                
        # Salvataggio nel file JSON
        out_file = BASE_DIR / f"results_snr_monosim_{strategy}.json"
        with open(out_file, "w") as f:
            json.dump(results_snr, f, indent=4)
            
        print(f"✅ Estrazione SNR completata: {out_file.name} (Seed 42, {valid_atoms}, L=10)")

# Lancia la funzione
extract_monosim_snr_configs()