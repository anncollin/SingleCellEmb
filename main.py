import argparse
import os
import yaml
import socket

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

from SSL.train_dino import run_dino_experiment
from SSL.eval_dino import evaluate_dino_experiment


#######################################################################################################
# PARSE COMMAND-LINE ARGUMENTS
#######################################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Run DINO experiment(s) from YAML file or directory.")
    parser.add_argument(
        "--todo",
        type=str,
        required=True,
        help="Path to a YAML file OR a directory containing multiple YAML files.",
    )
    parser.add_argument(
        "--wandb",
        type=str,
        default="false",
        help="Enable or disable wandb logging (true/false).",
    )
    return parser.parse_args()


#######################################################################################################
# LOAD CONFIGURATION FROM YAML FILE
#######################################################################################################
def load_config(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file {path} does not exist.")
    with open(path, "r") as f:
        return yaml.safe_load(f)


#######################################################################################################
# LIST YAML FILES FROM FILE OR DIRECTORY
#######################################################################################################
def list_yaml_files(path: str):
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        files = sorted([
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith((".yaml", ".yml"))
        ])
        if len(files) == 0:
            raise RuntimeError(f"No YAML files found in directory: {path}")
        return files

    raise FileNotFoundError(f"{path} is not a valid file or directory.")


#######################################################################################################
# MAIN ENTRY POINT
#######################################################################################################
def main():
    args = parse_args()
    yaml_files = list_yaml_files(args.todo)

    for yaml_path in yaml_files:

        print("\n============================================================")
        print(f"Running config: {yaml_path}")
        print("============================================================\n")

        cfg = load_config(yaml_path)

        # ----------------------------------------------------------------------
        # AUTO-DETECT MACHINE AND INJECT THE CORRECT DATA ROOT
        # ----------------------------------------------------------------------
        hostname = socket.gethostname()

        if "anncollin" in hostname:
            print("you are on local machine")
            cfg["data_root"]  = "/home/anncollin/Desktop/Nucleoles/dataset/MyDB_npy/"
            cfg["label_path"] = "/home/anncollin/Desktop/Nucleoles/dataset/labels/unique_drugs.csv"

        elif "orion" in hostname:
            print("you are on orion")
            cfg["data_root"]  = "/DATA/annso/MyDB_npy"
            cfg["label_path"] = "/DATA/annso/labels/unique_drugs.csv"

        else:
            raise RuntimeError(
                f"Unknown hostname '{hostname}'. "
                "Please add a data_root path for this machine."
            )
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # HANDLE WANDB OPTION FROM COMMAND LINE
        # ----------------------------------------------------------------------
        wandb_arg = args.wandb.lower()
        if wandb_arg in ["true", "1", "yes"]:
            cfg["use_wandb"] = True
        else:
            cfg["use_wandb"] = False
        # ----------------------------------------------------------------------

        print(f"Loaded experiment config from {yaml_path}")

        run_dino_experiment(cfg)
        evaluate_dino_experiment(cfg)


if __name__ == "__main__":
    main()
