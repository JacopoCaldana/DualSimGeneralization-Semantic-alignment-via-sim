import json
import matplotlib.pyplot as plt
from pathlib import Path

# Configurazione per rendere i grafici professionali
plt.style.use('seaborn-v0_8-whitegrid') # Stile pulito
plt.rcParams.update({'font.size': 12})

BASE_DIR = Path(__file__).resolve().parent

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

if __name__ == "__main__":
    # Puoi inserire qui il valore dell'Oracle che ti ha stampato il terminale
    # Esempio: plot_ablation_mono(oracle_acc=95.4)
    plot_ablation_mono(oracle_acc=90.0) 
    plot_exp_layers()
    plot_accuracy_vs_layers()
    plot_accuracy_vs_snr()