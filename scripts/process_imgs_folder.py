import argparse
import sys
from tqdm import tqdm
import os
import shutil
from tools import yaml_parser, FolderCreator, check_folder, are_all_images
from retinex import MultiScaleRetinex, SingleScaleRetinex
from pathlib import Path


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

settings = options[retinex_algo]



