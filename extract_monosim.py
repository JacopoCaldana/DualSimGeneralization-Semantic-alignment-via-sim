import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SEEDS = [27, 42, 123]
STRATEGIES = ['PPFE', 'Linear']

def extract_layers_monosim_all_seeds(csv_path_pattern="final_results_{strategy}.csv"):
    """Estrae i dati Accuracy vs Layers con tag 'monosim' nel nome del file."""
    for strategy in STRATEGIES:
        # Adattiamo il percorso in base a come sono nominati i tuoi CSV
        csv_path = csv_path_pattern.format(strategy=strategy)
        if not Path(csv_path).exists():
            print(f"⚠️ Salto: {csv_path} non trovato.")
            continue

        df_all = pd.read_csv(csv_path)
        valid_atoms = [16, 32,64]

        for seed in SEEDS:
            df = df_all[(df_all['Seed'] == seed) & (df_all['Alignment Type'] == strategy)]
            if df.empty: continue

            agg_df = df.groupby(['SIM Meta Atoms Intermediate X', 'SIM Layers'])['Accuracy SIM Mimo'].mean().reset_index()
            
            results_layers = {}
            for M in valid_atoms:
                chiave_atomi = f"{int(M)}x{int(M)}"
                results_layers[chiave_atomi] = {}
                df_M = agg_df[agg_df['SIM Meta Atoms Intermediate X'] == M]
                for _, row in df_M.iterrows():
                    L = str(int(row['SIM Layers']))
                    acc = row['Accuracy SIM Mimo']
                    results_layers[chiave_atomi][L] = acc * 100 if acc <= 1.0 else acc

            # NOME FILE CON TAG MONOSIM
            out_file = BASE_DIR / f"results_layers_monosim_{strategy}_seed{seed}.json"
            with open(out_file, "w") as f:
                json.dump(results_layers, f, indent=4)
            print(f"✅ Creato: {out_file.name}")

def extract_snr_monosim_all_seeds(csv_path_pattern="final_results_snr_{strategy}.csv"):
    """Estrae i dati Accuracy vs SNR con tag 'monosim' nel nome del file."""
    for strategy in STRATEGIES:
        csv_path = csv_path_pattern.format(strategy=strategy)
        if not Path(csv_path).exists():
            print(f"⚠️ Salto: {csv_path} non trovato.")
            continue

        df_all = pd.read_csv(csv_path)
        valid_atoms = [16, 32,64]

        for seed in SEEDS:
            # L=10 è lo standard per i tuoi test SNR
            df = df_all[(df_all['Seed'] == seed) & (df_all['Alignment Type'] == strategy) & (df_all['SIM Layers'] == 10)]
            if df.empty: continue

            agg_df = df.groupby(['SIM Meta Atoms Intermediate X', 'SNR [dB]'])['Accuracy SIM Mimo'].mean().reset_index()
            
            results_snr = {}
            for M in valid_atoms:
                chiave_atomi = f"{int(M)}x{int(M)}"
                results_snr[chiave_atomi] = {}
                df_M = agg_df[agg_df['SIM Meta Atoms Intermediate X'] == M].sort_values(by='SNR [dB]')
                for _, row in df_M.iterrows():
                    snr_val = str(row['SNR [dB]'])
                    acc = row['Accuracy SIM Mimo']
                    results_snr[chiave_atomi][snr_val] = acc * 100 if acc <= 1.0 else acc

            # NOME FILE CON TAG MONOSIM
            out_file = BASE_DIR / f"results_snr_monosim_{strategy}_seed{seed}.json"
            with open(out_file, "w") as f:
                json.dump(results_snr, f, indent=4)
            print(f"✅ Creato: {out_file.name}")

if __name__ == "__main__":
    # Esegui l'estrazione dai tuoi CSV
    extract_layers_monosim_all_seeds()
    extract_snr_monosim_all_seeds()