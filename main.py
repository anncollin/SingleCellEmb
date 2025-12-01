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
        action="store_true",
        help="Enable wandb logging (default: disabled).",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run only the training (run_dino_experiment).",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run only the evaluation (evaluate_dino_experiment).",
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
            cfg["data_root"]         = "/home/anncollin/Desktop/Nucleoles/dataset/MyDB_npy/"
            cfg["label_path"]        = "/home/anncollin/Desktop/Nucleoles/dataset/labels/unique_drugs.csv"
            cfg["callibration_path"] = "/home/anncollin/Desktop/Nucleoles/dataset/labels/callibration.csv"

        elif "orion" in hostname:
            print("you are on orion")
            cfg["data_root"]         = "/DATA/annso/MyDB_npy"
            cfg["label_path"]        = "/DATA/annso/labels/unique_drugs.csv"
            cfg["callibration_path"] = "/DATA/annso/labels/callibration.csv"

        else:
            raise RuntimeError(
                f"Unknown hostname '{hostname}'. "
                "Please add a data_root path for this machine."
            )
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # HANDLE WANDB FLAG
        # ----------------------------------------------------------------------
        cfg["use_wandb"] = bool(args.wandb)
        print(f"wandb enabled: {cfg['use_wandb']}")
        # ----------------------------------------------------------------------

        print(f"Loaded experiment config from {yaml_path}")

        do_train = args.train
        do_eval  = args.eval

        # Default behavior: if neither flag is given, run both
        if not do_train and not do_eval:
            do_train = True
            do_eval = True

        if do_train:
            print("Starting TRAIN phase...")
            run_dino_experiment(cfg)

        if do_eval:
            print("Starting EVAL phase...")
            evaluate_dino_experiment(cfg)


if __name__ == "__main__":
    main()
