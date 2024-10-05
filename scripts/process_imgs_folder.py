import argparse
import sys

import numpy as np
from tqdm import tqdm
import os
from tools import yaml_parser, FolderCreator, check_folder, TxtWriter
import retinex
from pathlib import Path
import cv2
from datetime import datetime
import time

######################################################################
# SET CONFIGS
######################################################################
script_path = Path(__file__).parent
config_path = Path(script_path.parent, "config", "config.yaml")
options = yaml_parser(config_path)

# Get the Alias of the algorithms
algorithms_known = {str(options[algorithm]["Alias"]): algorithm for algorithm in options}

# COMMAND LINE ARGS
parser = argparse.ArgumentParser(description='Data configuration')

parser.add_argument('--image_folder', type=str, required=True,
                    help="The folder containing the image; i.e.: <abs_path>/images")
parser.add_argument('--algo', type=str, required=True,
                    help="supported aliases: MSR or SSR")

# Parse Arguments
args = parser.parse_args()

# Set Folders and Algorithms
image_folder = check_folder(Path(args.image_folder)).__str__()
sel_alg = args.algo

######################################################################
# CHECKING CONFIGURATIONS AND FILES
######################################################################

# CHECKS
if sel_alg not in algorithms_known.keys():
    msg = "check the spelling of the Aliases, accepted are: {}".format(algorithms_known.keys())
    raise AttributeError

check_folder(image_folder)


######################################################################
# Main
######################################################################

# Get the algorithm config
retinex_algo = algorithms_known[sel_alg]
settings = options[retinex_algo]["Retinex"]

algorithm = getattr(retinex, retinex_algo)
algorthm_inst = algorithm()
algorthm_inst.variance = settings["Variance"]

# Create the folder for the output, if it exists we'll quit
out_folder_name = "out"
file_info = "log_Retinex.txt"

image_folder_content = os.listdir(image_folder)
if out_folder_name in image_folder_content:
    print("Error, the out folder already exists! Exiting...")
    sys.exit(1)
else:
    out_folder_path = FolderCreator(Path(image_folder, out_folder_name))

# define time list
times_proc = []
# Process the images
for image_file in tqdm(image_folder_content, "{}:enhancing".format(retinex_algo)):
    try:
        img = cv2.imread(Path(image_folder, image_file).__str__())
        t_preproc = time.time()
        img_enh = algorthm_inst.do(img)
        t_postproc = time.time()
        times_proc.append(t_postproc-t_preproc)
        cv2.imwrite(Path(out_folder_path.get_path(), image_file).__str__(), img_enh)
    except cv2.error as cve:
        print("Error while converting the image file {}:{}".format(image_file, cve))
        sys.exit(1)
    except Exception as exc:
        print("General error while converting the image file {}:{}".format(image_file, exc))
        sys.exit(1)

# Create info File
try:
    info_file = TxtWriter(Path(image_folder).parent, file_info)
    study_info = "".join([datetime.now().strftime("%d_%m_%Y_%H_%M_%S"), "->"
                "Algorithm:{}, Variance: {}".format(retinex_algo, algorthm_inst.variance)])
    avg_time = "".join([datetime.now().strftime("%d_%m_%Y_%H_%M_%S"), "->"
                "Processing avg speed: {} [s]".format(np.sum(times_proc)/len(times_proc))])
    info_file.append_row(study_info)
    info_file.append_row(avg_time)
except Exception as e:
    print("Error while writing info file: {}".format(e))






