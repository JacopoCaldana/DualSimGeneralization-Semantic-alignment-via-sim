import json
import matplotlib.pyplot as plt
from pathlib import Path

# Configurazione per rendere i grafici professionali
plt.style.use('seaborn-v0_8-whitegrid') # Stile pulito
plt.rcParams.update({'font.size': 12})

BASE_DIR = Path(__file__).resolve().parent
PLOT_DIR = BASE_DIR / "plots"

def plot_ablation_mono(oracle_acc=None):
    """Plotta i risultati dell'esperimento Mono-SIM (Accuracy vs Layers)."""
    json_path = BASE_DIR / "results_exp_1_mono.json"
    
    if not json_path.exists():
        print(f"⚠️ File {json_path.name} non trovato.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    # Convertiamo le chiavi in numeri e ordiniamo
    layers = sorted([int(k) for k in data.keys()])
    accuracies = [data[str(l)] for l in layers]

    plt.figure(figsize=(10, 6))
    plt.plot(layers, accuracies, marker='o', linestyle='-', color='tab:orange', linewidth=2, label='Mono-SIM (TX Only)')
    
    # Se passiamo l'Oracle, disegniamo una linea tratteggiata
    if oracle_acc:
        plt.axhline(y=oracle_acc, color='red', linestyle='--', label=f'Oracle Baseline ({oracle_acc:.1f}%)')

    plt.title('Ablation Study: Accuracy vs Number of Layers (L)', pad=20)
    plt.xlabel('Number of Layers (L)')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.xticks(layers) # Mostra solo i layer effettivamente testati
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    save_path = BASE_DIR / "plot_ablation_mono.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Grafico salvato in: {save_path}")

def plot_exp_layers():
    """Plotta il confronto tra diverse configurazioni di atomi (Fig 2 del paper)."""
    json_path = BASE_DIR / "results_exp_layers.json"
    
    if not json_path.exists():
        return

    with open(json_path, "r") as f:
        results = json.load(f)

    plt.figure(figsize=(10, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for (config, data), color in zip(results.items(), colors):
        layers = sorted([int(l) for l in data.keys()])
        accs = [data[str(l)] for l in layers]
        plt.plot(layers, accs, marker='s', label=f'Atoms {config}', color=color)

    plt.title('Accuracy vs Layers for different Meta-Atom configurations')
    plt.xlabel('Layers (L)')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.savefig(BASE_DIR / "plot_accuracy_layers.png", dpi=300)
    print("✅ Grafico Layers salvato.")


def plot_accuracy_vs_layers():
    """Genera il grafico Accuratezza vs Numero di Layer (Simile a Fig. 2 del paper)."""
    json_path = BASE_DIR / "results_exp_layers.json"
    if not json_path.exists():
        print("❓ File 'results_exp_layers.json' non trovato. Salto questo grafico.")
        return

    with open(json_path, "r") as f:
        results = json.load(f)

    plt.figure(figsize=(10, 6))
    # Definiamo colori e marker per distinguere le configurazioni di atomi
    configs = {"16x16": ('tab:blue', 'o'), "32x32": ('tab:orange', 's'), "64x64": ('tab:green', '^')}

    for label, (color, marker) in configs.items():
        if label in results:
            data = results[label]
            # Ordiniamo i layer (le chiavi JSON sono stringhe)
            layers = sorted([int(l) for l in data.keys()])
            accs = [data[str(l)] for l in layers]
            plt.plot(layers, accs, label=f'Meta-atoms {label}', color=color, marker=marker, linestyle='-')

    plt.title('Accuracy vs Number of SIM Layers ($L$)', pad=15)
    plt.xlabel('SIM Layers ($L$)')
    plt.ylabel('Downstream Classification Accuracy (%)')
    plt.ylim(0, 100)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(BASE_DIR / "plot_exp_layers.png", dpi=300, bbox_inches='tight')
    print("✅ Grafico 'Accuracy vs Layers' salvato.")

def plot_accuracy_vs_snr():
    """Genera il grafico Accuratezza vs SNR (Simile a Fig. 3 del paper)."""
    json_path = BASE_DIR / "results_exp_snr.json"
    if not json_path.exists():
        print("❓ File 'results_exp_snr.json' non trovato. Salto questo grafico.")
        return

    with open(json_path, "r") as f:
        results = json.load(f)

    plt.figure(figsize=(10, 6))
    configs = {"16x16": ('tab:blue', 'o'), "32x32": ('tab:orange', 's'), "64x64": ('tab:green', '^')}

    for label, (color, marker) in configs.items():
        if label in results:
            data = results[label]
            # Gestione SNR: filtriamo "Inf" per il plot dell'asse X e lo teniamo per una linea
            snrs_numeric = sorted([int(s) for s in data.keys() if s != "Inf"])
            accs = [data[str(s)] for s in snrs_numeric]
            
            line = plt.plot(snrs_numeric, accs, label=f'Meta-atoms {label}', color=color, marker=marker)
            
            # Se esiste il dato per SNR infinito, disegniamo un punto orizzontale tratteggiato alla fine
            if "Inf" in data:
                plt.axhline(y=data["Inf"], color=color, linestyle='--', alpha=0.5)

    plt.title('Accuracy vs Signal-to-Noise Ratio (SNR)', pad=15)
    plt.xlabel('SNR [dB]')
    plt.ylabel('Downstream Classification Accuracy (%)')
    plt.ylim(0, 100)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(BASE_DIR / "plot_exp_snr.png", dpi=300, bbox_inches='tight')
    print("✅ Grafico 'Accuracy vs SNR' salvato.")    

######################################################################################################################################

def plot_asymmetric_rx_study():
    """
    Plots the results of the Asymmetric RX-Depth experiment.
    """
    asym_json = BASE_DIR / "results_rx_depth_study.json"
    sym_json = BASE_DIR / "results_exp_layers.json" # 

    if not asym_json.exists():
        print(f"❌ File not found: {asym_json}")
        return

    # 1. Load Asymmetric Data
    with open(asym_json, "r") as f:
        data_asym = json.load(f)
    
    # Extract L_RX and Accuracy
    l_rx_values = sorted([int(k) for k in data_asym.keys()])
    acc_asym = [data_asym[str(k)] for k in l_rx_values]

    # 2. Setup Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    # Plot Asymmetric Curve
    plt.plot(l_rx_values, acc_asym, marker='s', markersize=8, linewidth=2.5, 
             color='#2ca02c', label=r'Asymmetric (Fixed $L_{TX}=10$, Var $L_{RX}$)')

    # 3. Optional: Overlay Symmetric Data for comparison
    if sym_json.exists():
        with open(sym_json, "r") as f:
            data_sym = json.load(f)
        
        # Prendiamo i dati per 32x32 meta-atomi (quelli del tuo 70%)
        if "32x32" in data_sym:
            l_sym_values = sorted([int(k) for k in data_sym["32x32"].keys()])
            acc_sym = [data_sym["32x32"][str(k)] for k in l_sym_values]
            plt.plot(l_sym_values, acc_sym, marker='o', linestyle='--', alpha=0.7,
                     color='#ff7f0e', label=r'Symmetric ($L_{TX} = L_{RX}$)')

    # 4. Formatting
    plt.title("Impact of Receiver Depth on Semantic Accuracy", fontsize=14, pad=15)
    plt.xlabel(r"Number of RX SIM Layers ($L_{RX}$)", fontsize=12)
    plt.ylabel("Downstream Classification Accuracy (%)", fontsize=12)
    
    plt.xticks(l_rx_values)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, loc='lower right', fontsize=11)

    # 5. Save
    save_path = PLOT_DIR / "asymmetric_rx_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Comparison plot saved to: {save_path}")



def plot_asymmetric_tx_study():
    """
    Plots the results of the Asymmetric TX-Depth experiment (Fixed RX=10).
    """
    tx_json = BASE_DIR / "results_asymmetric_tx_study.json"

    if not tx_json.exists():
        print(f"❌ Errore: Il file {tx_json.name} non esiste ancora.")
        print("Assicurati di aver completato l'esperimento TX prima di lanciare il plot.")
        return

    # 1. Caricamento dati
    with open(tx_json, "r") as f:
        data_tx = json.load(f)
    
    # Estrazione e ordinamento dei valori (L_TX e Accuracy)
    l_tx_values = sorted([int(k) for k in data_tx.keys()])
    acc_tx = [data_tx[str(k)] for k in l_tx_values]

    # 2. Configurazione Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(9, 6))
    
    # Plot della curva TX (usiamo il blu e il triangolo come marker)
    plt.plot(l_tx_values, acc_tx, marker='^', markersize=9, linewidth=2.5, 
             color='#1f77b4', label=r'Variable $L_{TX}$ (Fixed $L_{RX}=10$)')

    # 3. Formattazione estetica
    plt.title("Semantic Accuracy vs. Transmitter Depth", fontsize=14, pad=15)
    plt.xlabel(r"Number of TX SIM Layers ($L_{TX}$)", fontsize=12)
    plt.ylabel("Downstream Accuracy (%)", fontsize=12)
    
    # Impostazioni assi e griglia
    plt.xticks(l_tx_values)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Annotazione per chiarezza
    plt.text(min(l_tx_values), 5, f" Setup: Identity Channel\n SNR: Infinite\n RX Layers: 10 (Fixed)", 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.legend(frameon=True, loc='lower right', fontsize=11)

    # 4. Salvataggio
    save_path = PLOT_DIR / "tx_depth_impact.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plot salvato con successo in: {save_path}")



def plot_full_comparison():
    """
    Confronto Universale basato sui tuoi file:
    1. Var RX -> results_rx_depth_study.json
    2. Var TX -> results_asymmetric_tx_study.json
    3. Symmetric -> results_exp_layers.json
    """
    # MAPPA DEI FILE ESATTA
    rx_json = BASE_DIR / "results_rx_depth_study.json"
    tx_json = BASE_DIR / "results_asymmetric_tx_study.json"
    sym_json = BASE_DIR / "results_exp_layers.json"

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(11, 7))

    # --- 1. PLOT ASYMMETRIC RX (Verde) ---
    if rx_json.exists():
        with open(rx_json, "r") as f:
            data_rx = json.load(f)
        layers = sorted([int(k) for k in data_rx.keys()])
        acc = [data_rx[str(k)] for k in layers]
        plt.plot(layers, acc, marker='s', markersize=8, linewidth=2.5, 
                 color='#2ca02c', label=r'Var $L_{RX}$ (Fixed $L_{TX}=10$)')
    else:
        print(f"⚠️ Salto RX: {rx_json.name} non trovato.")

    # --- 2. PLOT ASYMMETRIC TX (Blu) ---
    if tx_json.exists():
        with open(tx_json, "r") as f:
            data_tx = json.load(f)
        layers = sorted([int(k) for k in data_tx.keys()])
        acc = [data_tx[str(k)] for k in layers]
        plt.plot(layers, acc, marker='^', markersize=8, linewidth=2.5, 
                 color='#1f77b4', label=r'Var $L_{TX}$ (Fixed $L_{RX}=10$)')
    else:
        print(f"⚠️ Salto TX: {tx_json.name} non trovato.")

    # --- 3. PLOT SYMMETRIC (Arancione) ---
    if sym_json.exists():
        with open(sym_json, "r") as f:
            data_sym = json.load(f)
        if "32x32" in data_sym:
            d = data_sym["32x32"]
            layers = sorted([int(k) for k in d.keys()])
            acc = [d[str(k)] for k in layers]
            plt.plot(layers, acc, marker='o', linestyle='--', alpha=0.6,
                     color='#ff7f0e', label=r'Symmetric ($L_{TX} = L_{RX}$)')

    # --- 4. BASELINE ORACLE ---
    plt.axhline(y=95.58, color='r', linestyle=':', alpha=0.6, label='Oracle Baseline (95.58%)')

    # --- FORMATTAZIONE E DINAMICHE ---
    plt.title("Dual-SIM Depth Analysis: TX vs RX Contribution", fontsize=15, pad=20)
    plt.xlabel("Number of Variable Layers ($L$)", fontsize=12)
    plt.ylabel("Classification Accuracy (%)", fontsize=12)
    
    # Box informativo sulle dinamiche (500/1000 iterazioni)
    plt.text(1, 5, 
             "Experiment Dynamics:\n"
             "• $L \leq 10$: 500 iters\n"
             "• $L > 10$: 1000 iters\n"
             "• Channel: Identity ($H=I$)\n"
             "• SNR: $\infty$", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.ylim(0, 100)
    plt.xticks([1, 2, 5, 10, 15, 20, 25])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, loc='lower right', fontsize=10, shadow=True)

    save_path = PLOT_DIR / "full_comparison_tx_rx_sym.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Plot salvato correttamente in: {save_path}")




if __name__ == "__main__":
    # Puoi inserire qui il valore dell'Oracle che ti ha stampato il terminale
    # Esempio: plot_ablation_mono(oracle_acc=95.4)
    plot_ablation_mono(oracle_acc=90.0) 
    plot_exp_layers()
    plot_accuracy_vs_layers()
    plot_accuracy_vs_snr()
    plot_asymmetric_rx_study()
    plot_asymmetric_tx_study()
    plot_full_comparison()

