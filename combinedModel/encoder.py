#!/usr/bin/env python
import os
import sys
import subprocess
import pickle

# --- Etape 1 : Cloner le dépôt et installer le package ---

# Cloner le dépôt si le dossier n'existe pas
repo_dir = "neuralgcmPFE"
if not os.path.exists(repo_dir):
    print("Clonage du dépôt neuralgcmPFE...")
    subprocess.run(["git", "clone", "https://github.com/mylrcd/neuralgcmPFE.git"], check=True)
else:
    print("Le dépôt neuralgcmPFE existe déjà.")

# Installer le package en mode editable
print("Installation de neuralgcmPFE en mode editable...")
subprocess.run([sys.executable, "-m", "pip", "install", "-e", repo_dir], check=True)

# --- Etape 2 : Configuration du path (à modifier si besoin) ---
# Si vous avez un chemin spécifique à ajouter (ici l'exemple donné), conservez-le sinon adaptez ou supprimez.
custom_path = r"c:/Users/mayeu/SynologyDrive/DossiersMayeul/Mes Documents/TSP/3A/PFE/NeuralGCM/quickstart/neuralgcmPFE"
if custom_path not in sys.path:
    sys.path.append(custom_path)

# --- Etape 3 : Importer les modules nécessaires ---
import gcsfs
from neuralgcmPFE import neuralgcm
from import_era5 import generate_final_xarray
import xarray as xr

# --- Etape 4 : Définition des paramètres et génération du xarray ---
def main():
    # Paramètres temporels et de simulation
    start_time = '2020-01-01'
    end_time = '2020-03-01'
    data_inner_steps = 6

    # Variables d'entrée et de forçage
    input_variables = [
        'geopotential', 
        'specific_humidity',
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
        'specific_cloud_ice_water_content',
        'specific_cloud_liquid_water_content'
    ]
    forcing_variables = ['sea_ice_cover', 'sea_surface_temperature']

    # Paramètres de la grille d'origine
    longitude_nodes_original = 256
    latitude_nodes_original = 128
    level_original = list(range(37))

    # Nom du checkpoint du modèle
    model_name = 'neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl'
    
    # Chemin d'accès aux données ERA5 sur GCS
    path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
    gcs = gcsfs.GCSFileSystem(token='anon')

    # Chargement du checkpoint depuis GCS
    ckpt_path = f'gs://gresearch/neuralgcm/04_30_2024/{model_name}'
    print("Chargement du checkpoint depuis :", ckpt_path)
    with gcs.open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    # Création du modèle à partir du checkpoint
    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

    # Génération du xarray final
    print("Génération du xarray final...")
    regridded_encode = generate_final_xarray(
        path,
        gcs,
        model,
        start_time,
        end_time,
        data_inner_steps,
        input_variables,
        forcing_variables,
        longitude_nodes_original,
        latitude_nodes_original,
        level_original
    )

    # Sauvegarde du xarray au format NetCDF
    output_filename = "regridded_encode_64x32_3months.nc"
    print("Sauvegarde du xarray dans le fichier :", output_filename)
    regridded_encode.to_netcdf(output_filename)
    print("Sauvegarde terminée.")

if __name__ == "__main__":
    main()
