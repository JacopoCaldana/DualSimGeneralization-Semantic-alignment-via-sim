import torch
from pathlib import Path
from gdown import download
from zipfile import ZipFile
from dotenv import dotenv_values

# =======================================================
#
#                 METHODS DEFINITION
#
# =======================================================

# Definiamo il percorso assoluto del file .env.txt rispetto a questo script
# Questo evita l'errore KeyError se lanci lo script da cartelle diverse
CURRENT_DIR = Path(__file__).parent.absolute()
ENV_PATH = CURRENT_DIR / 'env.txt'

def download_zip_from_gdrive(
    id: str,
    name: str,
    path: str,
) -> None:
    """Metodo per scaricare file zip da Google Drive."""
    DATA_DIR = Path(path)
    ZIP_PATH = DATA_DIR / 'data.zip'
    DIR_PATH = DATA_DIR / f'{name}/'

    DATA_DIR.mkdir(exist_ok=True, parents=True)

    if not ZIP_PATH.exists():
        print(f"📥 Download in corso (ID: {id})...")
        download(id=id, output=str(ZIP_PATH))

    if not DIR_PATH.is_dir():
        print(f"📦 Estrazione file in {DIR_PATH}...")
        with ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(ZIP_PATH.parent)
    return None


def download_models_ckpt(
    models_path: Path,
    model_name: str,
) -> None:
    """Scarica i checkpoint dei modelli leggendo l'ID dal file .env.txt."""
    print('\n🚀 Avvio procedura di setup modelli...')

    # Caricamento configurazione dal percorso assoluto
    config = dotenv_values(ENV_PATH)
    
    if 'MODELS_ID' not in config:
        raise KeyError(f"❌ Errore: 'MODELS_ID' non trovato nel file {ENV_PATH}")

    id = config['MODELS_ID']
    download_zip_from_gdrive(id=id, name=model_name, path=str(models_path))

    print('✅ Setup completato.\n')
    return None


# =======================================================
#
#                     MAIN LOOP
#
# =======================================================

def main() -> None:
    """Test di funzionamento per il download dei dati."""
    print('🧪 Esecuzione test di integrazione...')
    
    # Caricamento configurazione per il test
    config = dotenv_values(ENV_PATH)
    
    if not ENV_PATH.exists():
        print(f"❌ Errore: Il file {ENV_PATH} non esiste!")
        return

    # 1. Test Download Latents
    print('Running Latents Test...', end='\t')
    if 'DATA_ID' in config:
        download_zip_from_gdrive(id=config['DATA_ID'], path='data', name='latents')
        print('[PASSED]')
    else:
        print('[FAILED: DATA_ID missing]')

    # 2. Test Download Classifiers
    print('Running Classifiers Test...', end='\t')
    MODELS_PATH = CURRENT_DIR / 'models'
    download_models_ckpt(models_path=MODELS_PATH, model_name='classifiers')
    print('[PASSED]')

    return None


if __name__ == '__main__':
    main()
