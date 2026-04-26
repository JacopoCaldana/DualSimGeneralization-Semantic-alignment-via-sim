import json
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import pandas as pd


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


def plot_comparison_layers():
    """Confronta Linear vs PPFE nello stesso grafico (Accuracy vs Layers)."""
    strategies = ["Linear", "PPFE"]
    configs = {"16x16": ('tab:blue', 'o'), "32x32": ('tab:orange', 's'),"64x64": ('tab:green','^')}
    
    plt.figure(figsize=(11, 7))

    for strategy in strategies:
        json_path = BASE_DIR / f"results_layers_{strategy}.json"
        if not json_path.exists():
            print(f"⚠️ File {json_path.name} non trovato. Salto...")
            continue

        with open(json_path, "r") as f:
            results = json.load(f)

        # Stile linea: continuo per Linear, tratteggiato per PPFE
        line_style = '-' if strategy == "Linear" else '--'
        alpha_val = 1.0 if strategy == "Linear" else 0.7

        for label, (color, marker) in configs.items():
            if label in results:
                data = results[label]
                layers = sorted([int(l) for l in data.keys()])
                accs = [data[str(l)] for l in layers]
                
                plt.plot(layers, accs, 
                         label=f'{label} ({strategy})', 
                         color=color, 
                         marker=marker, 
                         ls=line_style, 
                         alpha=alpha_val,
                         linewidth=2.5)

    plt.axhline(y=95.58, color='black', linestyle=':', alpha=0.5, label='No Missmatch (95.58%)')
    
    plt.title('Strategy Comparison: Accuracy vs SIM Layers ($L$)', fontsize=15, pad=20)
    plt.xlabel('Number of SIM Layers ($L$)', fontsize=12)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Legenda su due colonne per non coprire il grafico
    plt.legend(loc='lower right', ncol=2, frameon=True, shadow=True, fontsize=10)
    
    save_path = BASE_DIR / "plot_comparison_layers_final.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot confronto Layers salvato: {save_path.name}")


def plot_comparison_snr():
    """Confronta Linear vs PPFE nello stesso grafico (Accuracy vs SNR)."""
    strategies = ["Linear", "PPFE"]
    configs = {"16x16": ('tab:blue', 'o'), "32x32": ('tab:orange', 's')}
    
    plt.figure(figsize=(11, 7))

    for strategy in strategies:
        json_path = BASE_DIR / f"results_snr_{strategy}.json"
        if not json_path.exists():
            print(f"⚠️ File {json_path.name} non trovato. Salto...")
            continue

        with open(json_path, "r") as f:
            results = json.load(f)

        line_style = '-' if strategy == "Linear" else '--'
        alpha_val = 1.0 if strategy == "Linear" else 0.7

        for label, (color, marker) in configs.items():
            if label in results:
                data = results[label]
                snrs_numeric = sorted([int(s) for s in data.keys() if s != "Inf"])
                accs = [data[str(s)] for s in snrs_numeric]
                
                plt.plot(snrs_numeric, accs, 
                         label=f'{label} ({strategy})', 
                         color=color, 
                         marker=marker, 
                         ls=line_style, 
                         alpha=alpha_val,
                         linewidth=2.5)
                
                # Plot asintoto SNR Infinito (solo per Linear per non sporcare troppo)
                if "Inf" in data and strategy == "Linear":
                    plt.axhline(y=data["Inf"], color=color, linestyle=':', alpha=0.3)

    plt.title('Strategy Comparison: Accuracy vs SNR [dB]', fontsize=15, pad=20)
    plt.xlabel('SNR [dB]', fontsize=12)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.legend(loc='lower right', ncol=2, frameon=True, shadow=True, fontsize=10)
    
    save_path = BASE_DIR / "plot_comparison_snr_final.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot confronto SNR salvato: {save_path.name}")   

######################################################################################################################################
######## ASYMETRIC STUDY##############################
########################################################################

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

####################################################################################################################
######### DISJOINT OPT STUDY  ########################
###################################################################
def plot_accuracy_vs_layers_disjoint(strategy_name="PPFE"):
    """
    Genera il grafico Accuratezza vs Numero di Layer per l'architettura Disjoint.
    """
    # Nome file allineato con la funzione di salvataggio
    json_path = BASE_DIR / f"results_layers_disjoint_{strategy_name}.json"
    
    if not json_path.exists():
        print(f"❓ File '{json_path.name}' non trovato. Salto questo grafico.")
        return

    with open(json_path, "r") as f:
        results = json.load(f)

    plt.figure(figsize=(10, 6))
    configs = {"16x16": ('tab:purple', 'o'), "32x32": ('tab:red', 's'), "64x64": ('tab:brown', '^')}

    for label, (color, marker) in configs.items():
        if label in results:
            data = results[label]
            layers = sorted([int(l) for l in data.keys()])
            accs = [data[str(l)] for l in layers]
            plt.plot(layers, accs, label=f'Disjoint {label}', color=color, marker=marker, linestyle='-', linewidth=2)

    plt.axhline(y=95.58, color='black', linestyle=':', alpha=0.6, label='Oracle Baseline (95.58%)')

    plt.title(f'Disjoint Architecture: Accuracy vs SIM Layers ($L$) - {strategy_name}', pad=15, fontsize=14)
    plt.xlabel('SIM Layers ($L$)', fontsize=12)
    plt.ylabel('Downstream Classification Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    
    if results and list(results.keys()):
        first_key = list(results.keys())[0]
        plt.xticks(sorted([int(l) for l in results[first_key].keys()]))

    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_path = BASE_DIR / f"plot_layers_disjoint_{strategy_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Grafico 'Disjoint: Accuracy vs Layers' salvato in: {save_path.name}")


def plot_accuracy_vs_snr_disjoint(strategy_name="PPFE"):
    """
    Genera il grafico Accuratezza vs SNR per l'architettura Disjoint.
    """
    json_path = BASE_DIR / f"results_snr_disjoint_{strategy_name}.json"
    
    if not json_path.exists():
        print(f"❓ File '{json_path.name}' non trovato. Salto questo grafico.")
        return

    with open(json_path, "r") as f:
        results = json.load(f)

    plt.figure(figsize=(10, 6))
    configs = {"16x16": ('tab:purple', 'o'), "32x32": ('tab:red', 's'), "64x64": ('tab:brown', '^')}

    for label, (color, marker) in configs.items():
        if label in results:
            data = results[label]
            snrs_numeric = sorted([int(s) for s in data.keys() if s != "Inf"])
            accs = [data[str(s)] for s in snrs_numeric]
            
            plt.plot(snrs_numeric, accs, label=f'Disjoint {label} (L=10)', color=color, marker=marker, linewidth=2)
            
            if "Inf" in data:
                plt.axhline(y=data["Inf"], color=color, linestyle='--', alpha=0.5, label=f'No Noise limit ({label})')

    plt.title(f'Disjoint Architecture: Accuracy vs SNR - {strategy_name}', pad=15, fontsize=14)
    plt.xlabel('Signal-to-Noise Ratio (dB)', fontsize=12)
    plt.ylabel('Downstream Classification Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    
    plt.legend(loc='lower right', frameon=True, shadow=True, ncol=1 if len(configs) < 3 else 2, fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.7)

    save_path = BASE_DIR / f"plot_snr_disjoint_{strategy_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Grafico 'Disjoint: Accuracy vs SNR' salvato in: {save_path.name}")

################################################################################################################################

def plot_depth_lr_comparison(strategy_name="PPFE"):
    """Crea una griglia di grafici per confrontare i LR a diverse profondità."""
    json_path = BASE_DIR / f"grid_search_L_vs_LR_{strategy_name}.json"
    if not json_path.exists(): return

    with open(json_path, "r") as f:
        data = json.load(f)

    depths = list(data.keys())
    fig, axes = plt.subplots(1, len(depths), figsize=(18, 5), sharey=True)
    
    for i, L_key in enumerate(depths):
        ax = axes[i]
        for lr, metrics in data[L_key].items():
            ax.plot(metrics["loss_history"], label=f"LR={lr}")
        
        ax.set_title(f"Depth {L_key} (M=16x16)")
        ax.set_yscale('log')
        ax.set_xlabel("Iteration")
        if i == 0: ax.set_ylabel("Semantic Loss")
        ax.legend(fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(f"Grid Search Results: How Depth affects LR sensitivity ({strategy_name})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(BASE_DIR / f"plot_grid_search_L_vs_LR_{strategy_name}.png", dpi=300)
    print(f"✅ Grafico Cross-Grid salvato.")


##############################################################################################


def plot_ultimate_layers_comparison():
    """
    Genera il grafico finale Accuracy vs SIM Layers (L).
    Mappatura aggiornata:
    - Colore -> Architettura (Joint, Disjoint, MonoSIM)
    - Marker -> Meta-Atomi (16x16, 32x32)
    - Stile Linea -> Strategia (Linear, PPFE)
    """
    # 1. Definizione della mappa dei file
    files_map = {
        "Joint": {
            "Linear": BASE_DIR / "results_layers_Linear.json",
            "PPFE": BASE_DIR / "results_layers_PPFE.json"
        },
        "Disjoint": {
            "Linear": BASE_DIR / "results_layers_disjoint_Linear.json",
            "PPFE": BASE_DIR / "results_layers_disjoint_PPFE.json"
        },
        "MonoSIM": {
            "Linear": BASE_DIR / "results_layers_monosim_Linear.json",
            "PPFE": BASE_DIR / "results_layers_monosim_PPFE.json"
        }
    }

    # 2. Nuova Definizione dello Stile Visivo
    # Colori per evidenziare le Architetture
    colors = {"Joint": "tab:blue", "Disjoint": "tab:orange", "MonoSIM": "tab:green"}
    # Marker (Ingranditi) per distinguere la grandezza della metasuperficie
    markers = {"16x16": "o", "32x32": "^"} 
    # Stile linea per le Strategie
    linestyles = {"Linear": "-", "PPFE": "--"}

    plt.figure(figsize=(12, 8))
    
    # 3. Iterazione e Plotting dei Dati
    for arch_name, arch_dict in files_map.items():
        color = colors[arch_name]  # Il colore è ora dettato dall'architettura
        
        for strat_name, file_path in arch_dict.items():
            if not file_path.exists():
                print(f"⚠️ File mancante, verrà ignorato: {file_path.name}")
                continue
                
            with open(file_path, "r") as f:
                data = json.load(f)
                
            for atoms, marker in markers.items():
                if atoms in data:
                    layers_str = sorted([int(l) for l in data[atoms].keys()])
                    accs = [data[atoms][str(l)] for l in layers_str]
                    
                    plt.plot(
                        layers_str, accs, 
                        color=color, 
                        linestyle=linestyles[strat_name], 
                        marker=marker, 
                        markersize=10,  # <-- MARKER PIÙ GRANDI
                        linewidth=2.5,  # <-- LINEA PIÙ SPESSA
                        alpha=0.85
                    )

    # 4. Aggiunta Baseline 
    plt.axhline(y=95.58, color='black', linestyle=':', alpha=0.8, linewidth=2.5, label='No Missmatch(95.58%)')

    # 5. COSTRUZIONE LEGENDA CUSTOM 
    legend_elements = [
        # Sezione Architetture (Ora sono i COLORI)
        Line2D([0], [0], color="w", label=r"$\bf{Architecture:}$"),
        Line2D([0], [0], color=colors["Joint"], lw=3, label='Dual-SIM (Joint)'),
        Line2D([0], [0], color=colors["Disjoint"], lw=3, label='Dual-SIM (Disjoint)'),
        Line2D([0], [0], color=colors["MonoSIM"], lw=3, label='Mono-SIM (TX)'),
        
        Line2D([0], [0], color="w", label=""), # Spaziatore
        
        # Sezione Meta-Atomi (Ora sono i MARKER)
        Line2D([0], [0], color="w", label=r"$\bf{Meta-Atoms:}$"),
        Line2D([0], [0], color='gray', marker=markers["16x16"], linestyle='none', markersize=10, label='16x16'),
        Line2D([0], [0], color='gray', marker=markers["32x32"], linestyle='none', markersize=10, label='32x32'),
        
        Line2D([0], [0], color="w", label=""), # Spaziatore
        
        # Sezione Strategie (Linestyles)
        Line2D([0], [0], color="w", label=r"$\bf{Strategy:}$"),
        Line2D([0], [0], color='gray', linestyle=linestyles["Linear"], lw=2.5, label='Linear (Supervised)'),
        Line2D([0], [0], color='gray', linestyle=linestyles["PPFE"], lw=2.5, label='PPFE (Zero-Shot)'),
        
        Line2D([0], [0], color="w", label=""), # Spaziatore
        
        # Oracle
        Line2D([0], [0], color='black', linestyle=':', lw=2.5, label='No missmatch')
    ]

    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, shadow=True, fontsize=11)

    # 6. Formattazione Finale
    plt.title('Architecture Comparison: Downstream Accuracy vs SIM Layers', pad=15, fontsize=16, fontweight='bold')
    plt.xlabel('Number of SIM Layers ($L$)', fontsize=14)
    plt.ylabel('Downstream Classification Accuracy (%)', fontsize=14)
    plt.ylim(0, 100)
    plt.xticks([2, 5, 10, 15, 20, 25], fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 7. Salvataggio
    save_path = BASE_DIR / "plot_ultimate_comparison_layers.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Grafico finale definitivo salvato in: {save_path.name}")





import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def plot_ultimate_snr_comparison():
    """
    Genera il grafico finale Accuracy vs SNR.
    Mappatura coerente con il plot Layers:
    - Colore -> Architettura (Joint, Disjoint, MonoSIM)
    - Marker -> Meta-Atomi (16x16, 32x32)
    - Stile Linea -> Strategia (Linear, PPFE)
    """
    # 1. Definizione della mappa dei file (Aggiornata per SNR)
    files_map = {
        "Joint": {
            "Linear": BASE_DIR / "results_snr_Linear.json",
            "PPFE": BASE_DIR / "results_snr_PPFE.json"
        },
        "Disjoint": {
            "Linear": BASE_DIR / "results_snr_disjoint_Linear.json",
            "PPFE": BASE_DIR / "results_snr_disjoint_PPFE.json"
        },
        "MonoSIM": {
            "Linear": BASE_DIR / "results_snr_monosim_Linear.json",
            "PPFE": BASE_DIR / "results_snr_monosim_PPFE.json"
        }
    }

    # 2. Stile Visivo Identico
    colors = {"Joint": "tab:blue", "Disjoint": "tab:orange", "MonoSIM": "tab:green"}
    markers = {"16x16": "o", "32x32": "^"} 
    linestyles = {"Linear": "-", "PPFE": "--"}

    plt.figure(figsize=(12, 8))
    
    # 3. Iterazione e Plotting dei Dati
    for arch_name, arch_dict in files_map.items():
        color = colors[arch_name]
        
        for strat_name, file_path in arch_dict.items():
            if not file_path.exists():
                print(f"⚠️ File mancante, verrà ignorato: {file_path.name}")
                continue
                
            with open(file_path, "r") as f:
                data = json.load(f)
                
            for atoms, marker in markers.items():
                if atoms in data:
                    # Estraiamo l'SNR numerico (escludendo l'eventuale caso senza rumore "Inf")
                    snrs_str = sorted([int(s) for s in data[atoms].keys() if s != "Inf"])
                    accs = [data[atoms][str(s)] for s in snrs_str]
                    
                    if len(snrs_str) > 0:
                        plt.plot(
                            snrs_str, accs, 
                            color=color, 
                            linestyle=linestyles[strat_name], 
                            marker=marker, 
                            markersize=10, 
                            linewidth=2.5, 
                            alpha=0.85
                        )

    # 4. Aggiunta Baseline 
    plt.axhline(y=95.58, color='black', linestyle=':', alpha=0.8, linewidth=2.5, label='No Missmatch (95.58%)')

    # 5. COSTRUZIONE LEGENDA CUSTOM
    legend_elements = [
        # Sezione Architetture
        Line2D([0], [0], color="w", label=r"$\bf{Architecture:}$"),
        Line2D([0], [0], color=colors["Joint"], lw=3, label='Dual-SIM (Joint)'),
        Line2D([0], [0], color=colors["Disjoint"], lw=3, label='Dual-SIM (Disjoint)'),
        Line2D([0], [0], color=colors["MonoSIM"], lw=3, label='Mono-SIM (TX)'),
        
        Line2D([0], [0], color="w", label=""), # Spaziatore
        
        # Sezione Meta-Atomi
        Line2D([0], [0], color="w", label=r"$\bf{Meta-Atoms:}$"),
        Line2D([0], [0], color='gray', marker=markers["16x16"], linestyle='none', markersize=10, label='16x16'),
        Line2D([0], [0], color='gray', marker=markers["32x32"], linestyle='none', markersize=10, label='32x32'),
        
        Line2D([0], [0], color="w", label=""), # Spaziatore
        
        # Sezione Strategie
        Line2D([0], [0], color="w", label=r"$\bf{Strategy:}$"),
        Line2D([0], [0], color='gray', linestyle=linestyles["Linear"], lw=2.5, label='Linear (Supervised)'),
        Line2D([0], [0], color='gray', linestyle=linestyles["PPFE"], lw=2.5, label='PPFE (Zero-Shot)'),
        
        Line2D([0], [0], color="w", label=""), # Spaziatore
        
        # Oracle
        Line2D([0], [0], color='black', linestyle=':', lw=2.5, label='No Mismatch')
    ]

    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, shadow=True, fontsize=11)

    # 6. Formattazione Finale
    plt.title('Architecture Comparison: Downstream Accuracy vs SNR (L=10)', pad=15, fontsize=16, fontweight='bold')
    plt.xlabel('Signal-to-Noise Ratio (dB)', fontsize=14)
    plt.ylabel('Downstream Classification Accuracy (%)', fontsize=14)
    plt.ylim(0, 100)
    
    # Adattamento dei Ticks specifici per l'SNR
    plt.xticks([-30, -20, -10, 0, 10, 20, 30], fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 7. Salvataggio
    save_path = BASE_DIR / "plot_ultimate_comparison_snr.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Grafico finale definitivo SNR salvato in: {save_path.name}")

#####################################################################
############### SEABORN PLOTS  ##################
#####################################################################    

def plot_ultimate_comparison_layers():
    """Confronta Linear vs PPFE usando Seaborn per media e varianza (shading)."""
    plt.rcParams.update({
        "figure.figsize": (16, 8),
        "font.size": 22,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 18,
        "legend.title_fontsize": 20,
        "lines.markersize": 14,
        "lines.linewidth": 3,
        "text.usetex": False, # Disattivato per evitare l'errore 'latex not found'
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"]
    })

    seeds = [27, 42, 123]
    strategies = ["Linear", "PPFE"]
    configs = ["32x32", "64x64"]
    custom_palette = { "32x32": "#ff7f0e", "64x64": "#2ca02c"}

    rows = []
    for strategy in strategies:
        for seed in seeds:
            json_path = Path(f"results_layers_{strategy}_seed{seed}.json")
            if not json_path.exists(): continue
            with open(json_path, "r") as f:
                data = json.load(f)
            for config in configs:
                if config in data:
                    for L, acc in data[config].items():
                        # PORTIAMO TUTTO IN SCALA 0-100
                        val = acc if acc > 1 else acc * 100
                        rows.append({
                            "Layers": int(L),
                            "Accuracy": val,
                            "Config": config,
                            "Strategy": strategy
                        })
    
    df = pd.DataFrame(rows)
    if df.empty:
        print("❌ ERRORE: Nessun dato trovato nei file JSON!")
        return

    # CREAZIONE PLOT
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.5})
    fig, ax = plt.subplots()

    # Plot principale
    sns.lineplot(
        data=df, x="Layers", y="Accuracy", hue="Config", style="Strategy",
        palette=custom_palette, markers={'Linear': 'o', 'PPFE': 'X'},
        dashes={'Linear': '', 'PPFE': (6, 3)}, errorbar='sd', ax=ax
    )

    # BASELINES (ORACLE) - Anche queste in scala 100
    try:
        with open("all_seeds_baselines.json", "r") as f:
            oracles = json.load(f)
            avg_lin = np.mean([oracles[str(s)]['Linear'] for s in seeds])
            avg_ppfe = np.mean([oracles[str(s)]['PPFE'] for s in seeds])
            # Se gli oracle sono 0.95, moltiplichiamo per 100
            if avg_lin < 1: avg_lin *= 100
            if avg_ppfe < 1: avg_ppfe *= 100
            
            ax.axhline(y=avg_lin, color='gray', ls='-.', lw=1.5, label='Original Linear', alpha=0.6)
            ax.axhline(y=avg_ppfe, color='gray', ls=':', lw=1.5, label='Original PPFE', alpha=0.6)
    except: pass

    # ESTETICA FINALE
    ax.set_title('Dual-SIM Scalability: Accuracy vs. Number of Layers ($L$)', pad=25, weight='bold')
    ax.set_xlabel('Number of SIM Layers $L$')
    ax.set_ylabel(' Accuracy (%)')
    
    # Limiti coerenti con il tuo screenshot
    ax.set_ylim(0, 100) 
    ax.set_xlim(1, 20)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20])

    # Spostiamo la legenda fuori a destra per non coprire i dati
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.savefig("plot_ultimate_comparison_layers_v2.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Grafico salvato correttamente: plot_ultimate_comparison_layers_v2.pdf")




def plot_ultimate_architecture_comparison_ppfe():
    # --- 1. CONFIGURAZIONE ESTETICA (Times New Roman / Scientific) ---
    plt.rcParams.update({
        "figure.figsize": (16, 9),
        "font.size": 22,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 18,
        "legend.title_fontsize": 20,
        "lines.markersize": 14,
        "lines.linewidth": 3,
        "text.usetex": False, 
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"]
    })

    seeds = [27, 42, 123]
    configs = [ "32x32"] # Ci focalizziamo su questi due come da tua richiesta
    
    # Mapping Architetture -> Prefisso File -> Colore
    arch_map = {
        "Joint": {"prefix": "results_layers_PPFE", "color": "#1f77b4"},      # Blu
        "Disjoint": {"prefix": "results_layers_disjoint_PPFE", "color": "#ff7f0e"}, # Arancio
        "Mono-SIM": {"prefix": "results_layers_monosim_PPFE", "color": "#2ca02c"}   # Verde
    }

    rows = []

    # --- 2. CARICAMENTO E AGGREGAZIONE DATI ---
    for arch_name, info in arch_map.items():
        for seed in seeds:
            file_path = Path(f"{info['prefix']}_seed{seed}.json")
            
            if not file_path.exists():
                print(f"⚠️ File mancante: {file_path.name}")
                continue
                
            with open(file_path, "r") as f:
                data = json.load(f)
            
            for config in configs:
                if config in data:
                    for L, acc in data[config].items():
                        # Normalizzazione 0-100
                        val = acc if acc > 1 else acc * 100
                        rows.append({
                            "Layers": int(L),
                            "Accuracy": val,
                            "Architecture": arch_name,
                            "Meta-Atoms": config
                        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("❌ Nessun dato trovato! Controlla i nomi dei file .json")
        return

    # --- 3. CREAZIONE PLOT CON SEABORN ---
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.5})
    fig, ax = plt.subplots()

    # hue="Architecture" -> Cambia Colore
    # style="Meta-Atoms" -> Cambia Marker e Linea
    plot = sns.lineplot(
        data=df, 
        x="Layers", y="Accuracy", 
        hue="Architecture", 
        style="Meta-Atoms",
        palette={k: v['color'] for k, v in arch_map.items()},
        markers={ '32x32': '^'},
        dashes={'32x32': (None, None)}, # Continua per 16, tratteggiata per 32
        errorbar='sd', # Sfumatura della deviazione standard
        ax=ax
    )

    # --- 4. AGGIUNTA ORACLE (Baseline PPFE media) ---
    #try:
     #   with open("all_seeds_baselines.json", "r") as f:
      #      oracles = json.load(f)
       #     avg_ppfe = np.mean([oracles[str(s)]['PPFE'] for s in seeds])
        #    if avg_ppfe < 1: avg_ppfe *= 100
         #   ax.axhline(y=avg_ppfe, color='gray', ls=':', lw=2, alpha=0.8, label='Baseline PPFE (Ideal)')
    #except: pass

    # --- 5. REFINEMENT ASSI E TITOLI ---
    ax.set_title('Architecture Comparison', pad=25)
    ax.set_xlabel('Number of SIM Layers $L$')
    ax.set_ylabel(' Accuracy (%)')
    
    # Range richiesto: 0-100
    ax.set_ylim(0, 105) 
    ax.set_xlim(1, 20)
    
    # Ticks specifici richiesti
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20])

    # Legenda esterna per pulizia
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    
    # Salvataggio
    save_path = "plot_architecture_comparison_PPFE.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Grafico finale salvato in: {save_path}")


#######################################
######################################    

if __name__ == "__main__":
    # Puoi inserire qui il valore dell'Oracle che ti ha stampato il terminale
    # Esempio: plot_ablation_mono(oracle_acc=95.4)
    plot_ablation_mono(oracle_acc=90.0) 
    plot_exp_layers()
    plot_comparison_layers()
    plot_comparison_snr()
    plot_asymmetric_rx_study()
    plot_asymmetric_tx_study()
    plot_full_comparison()
    plot_accuracy_vs_layers_disjoint(strategy_name="Linear")
    plot_accuracy_vs_snr_disjoint(strategy_name="Linear")
    plot_accuracy_vs_layers_disjoint(strategy_name="PPFE")
    plot_accuracy_vs_snr_disjoint(strategy_name="PPFE")
    plot_depth_lr_comparison()
    plot_ultimate_layers_comparison()
    plot_ultimate_snr_comparison()
    plot_ultimate_comparison_layers()
    plot_ultimate_architecture_comparison_ppfe()



