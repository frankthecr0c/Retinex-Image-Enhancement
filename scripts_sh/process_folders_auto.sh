#!/bin/bash

# Controlla se è stato fornito un argomento per il percorso
if [ -z "$1" ]; then
  echo "Errore: per favore fornisci il percorso della cartella come argomento."
  echo "Uso: $0 <percorso_cartella>"
  exit 1
fi

# Imposta il percorso della cartella dal primo argomento
dataset_folder="$1"

# Array degli algoritmi da utilizzare
#algos=("SSR")
#algos=("MSR")

algos=( "SSR" "MSR")

python_script="../scripts/process_imgs_folder.py"  # Modifica con il nome effettivo del tuo script


# Cicla attraverso ogni algoritmo
for algo in "${algos[@]}"; do
  echo "Elaborazione con algoritmo $algo"

  # Cicla attraverso ogni cartella di dataset
  find "$dataset_folder" -mindepth 1 -maxdepth 1 -type d | while read dataset_path; do
    # Estrai il nome del dataset dal percorso
    dataset_name=$(basename "$dataset_path")

    # Costruisci il percorso della cartella delle immagini
    image_folder="$dataset_path/images"

    # Esegui lo script Python e controlla il codice di uscita
    python3 "$python_script" --image_folder "$image_folder" --algo "$algo"
    if [ $? -ne 0 ]; then
      echo "Errore durante l'elaborazione del dataset $dataset_name con $algo. Interruzione dello script."
      exit 1
    fi

    # (Opzionale) Stampa un messaggio di conferma
    echo "Dataset $dataset_name processato con successo con $algo."
  done
done