# Deep Unrolling del Dual-SIM — guida ai file

Questa è la **versione attuale**  del deep unrolling dell'Algorithm 2
(SCA proiettata). Serve a configurare le fasi della
Dual-SIM in **pochi passi** invece delle migliaia di iterazioni dell'algoritmo classico.

> **Idea in una riga.** Il deep unrolling è l'algoritmo iterativo classico "srotolato"
> in `K` layer: **un layer = un passo di SCA proiettata**. Le **fasi ξ sono lo stato**
> (ricalcolate da zero per ogni canale `H`), **non** pesi appresi. L'unica cosa
> eventualmente *appresa* è lo **schedule dei passi per-layer** `W^k` (più momentum e
> coupling `S^k`). Con `epochs=0` non si allena nulla: è un ottimizzatore a passi fissi
> (momentum + step diagonale + innovazione SCA) che già converge più in fretta del classico.

---

## I 5 file dello stack

| File | Cosa è | Si lancia? |
|---|---|---|
| **`sca_unrolled.py`** | Il **cuore**: il modello `DualSIMUnrolledSCA` + il trainer `train_sca_unrolled`. | No (libreria, importata dagli altri) |
| **`compare_unroll_vs_analog.py`** | Deliverable **NMSE-vs-layer**: unroll vs Dual-SIM classica, stesso `A` e stessi canali. | Sì |
| **`accuracy_unrolled.py`** | **Accuracy CIFAR-10 vs L** (over-the-air) dell'unroll vs Dual-SIM classica. | Sì |
| **`accuracy_vs_k.py`** | **Accuracy vs K** (complessità): due sweep, uno per `L`, uno per atomi/layer `M`. | Sì |
| **`diag_nmse_vs_k.py`** | Sonda rapida: quanto scende l'NMSE (untrained) al variare di `K`, per ogni `L`. | Sì |

Tutto riusa **la stessa fisica SIM e la stessa pipeline CIFAR-10** dell'esperimento
classico, così i numeri sono direttamente confrontabili.

---

## Come lanciarli

Sempre dentro l'ambiente:
```bash
source venv/bin/activate
```

### 1) NMSE-vs-layer — `compare_unroll_vs_analog.py`
Mostra che la curva NMSE dell'unroll sta **sotto** quella della classica a parità di iterazioni.
```bash
python compare_unroll_vs_analog.py --in-xy 16 12 --out-xy 24 16 -M 16 -L 5 -K 100
```
**Output** → `results_sca/nmse_vs_layer_<tag>.{csv,pdf,png}`
CSV: colonne `layer, nmse_analog, nmse_unroll`. Curva blu = unroll, rossa = classica.

### 2) Accuracy vs L — `accuracy_unrolled.py`
```bash
# untrained (default, veloce): un forward a K layer per ogni L
python accuracy_unrolled.py -M 32 -L 2 5 10 15 -K 400 --alignment PPFE --seed 42

# confronto a PARI iterazioni con la classica (es. 100 it)
python accuracy_unrolled.py -M 32 -L 2 5 10 15 -K 100 --with-classic --classic-iters 100
```
**Output** → `results_accuracy_unrolled/accuracy_unrolled_<tag>.{csv,pdf,png}`
Il `<tag>` codifica la config: `{PPFE|Linear}_M{M}_K{K}_{untrained|ep{E}}[_vsclassic{N}it]_seed{seed}`
(così run diversi **non si sovrascrivono**).

### 3) Accuracy vs K (complessità) — `accuracy_vs_k.py`
Due plot in una run: curve **per L** (M fisso) e curve **per M** (L fisso); per ogni config la
classica è la tratteggiata dello stesso colore.
```bash
# untrained, con confronto classico
python accuracy_vs_k.py --alignment PPFE --seed 42

# versione ALLENATA con split train/test dei canali (alleno su H random, valuto su H held-out)
python accuracy_vs_k.py --epochs 8 --n-channels 5 --ks 5 20 50 100 200 --only byL

# rifare SOLO il plot da un CSV già esistente, cappato a K=200 (niente ri-run)
python accuracy_vs_k.py --replot --only byL --kmax 200 --oracle 95.29
```
**Output** → `results_accuracy_unrolled/acc_vs_k_{byL|byM}_<tag>.{csv,pdf,png}`
Flag utili: `--n-channels N` (media su N canali held-out, seed+1000), `--epochs E` (0 = untrained),
`--kmax` (cappa l'asse K, aggiunge suffisso `_kmax<N>` per non sovrascrivere), `--no-classic`.

### 4) Sonda diagnostica — `diag_nmse_vs_k.py`
Nessun argomento: stampa una tabella NMSE(untrained) vs `K` per `L∈{2,5,10,15}`, confrontata col
floor della classica@3000 iter. Serve a capire **quanti K servono** a ogni `L`.
```bash
python diag_nmse_vs_k.py
```

---

## `sca_unrolled.py` in dettaglio (per chi vuole modificare)

`DualSIMUnrolledSCA(sim, K, ...)` — la rete srotolata. Per ogni layer `k`:
1. calcola la cascata `Z = G_R(ξ_R)·H·G_T(ξ_T)`;
2. innovazione SCA `g^k = Re{Jᴴ β* r}` (= `−∇_ξ` dell'NMSE in forma coseno);
3. normalizza (`innovation_norm='rms'`) + momentum (Adam-style) + step diagonale `W^k`;
4. aggiorna `ξ ← S^k ξ + W^k g^k` (con `S^k = I` di default).

Metodi chiave:
- `infer_phases(H, A)` → esegue i `K` layer e ritorna le fasi finali (inferenza single-shot).
- `eval_curve(H, A)` → curva NMSE-vs-layer (per i plot).

`train_sca_unrolled(sim, A, K, ...)` — allena **solo** `W^k`, `μ^k`, `S^k` (le fasi non sono
parametri) su canali `H` campionati freschi a ogni minibatch, con **deep supervision** (media
NMSE su tutti i layer), LR cosine e **best-restore** (tiene il miglior modello; l'untrained è un
candidato → l'allenato non può fare peggio).

Opzioni principali: `coupling` (`scalar`/`diagonal`), `innovation_norm` (`rms`/`sign`/`unit`/`none`),
`momentum`, `learn_S`, `first_order` (innovazione detached, per config grandi), `analytic`
(jacobiana esplicita + gradient-checkpointing).



---

## Cartelle di output
- `results_sca/` — curve NMSE-vs-layer.
- `results_accuracy_unrolled/` — accuracy vs L e accuracy vs K (CSV + PDF + PNG).
