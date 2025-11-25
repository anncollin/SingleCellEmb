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
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    None
#
#    Returns:
#    ---------------------------
#    args : Namespace
#        Parsed command-line arguments containing path to the TODO config file.
#######################################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Run DINO experiment from TODO list.")
    parser.add_argument(
        "--todo",
        type=str,
        required=True,
        help="Path to a YAML configuration file inside Todo_List (e.g. Todo_List/exp_dino_cells.yaml).",
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
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    path : str
#        Path to the YAML configuration file.
#
#    Returns:
#    ---------------------------
#    cfg : dict
#        Configuration dictionary.
#######################################################################################################
def load_config(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file {path} does not exist.")
    with open(path, "r") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.safe_load(f)


#######################################################################################################
# MAIN ENTRY POINT
#######################################################################################################
def main():
    args = parse_args()
    cfg  = load_config(args.todo)

    # ----------------------------------------------------------------------
    # AUTO-DETECT MACHINE AND INJECT THE CORRECT DATA ROOT
    # ----------------------------------------------------------------------
    hostname = socket.gethostname()

    if "anncollin" in hostname:
        print('you are on local machine')
        cfg["data_root"] = "/home/anncollin/Desktop/Nucleoles/dataset/MyDB/"
    elif "orion" in hostname:
        print('you are on orion')
        cfg["data_root"] = "/BensonDATA_orion/anncollin/MyDB"
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
    
    print(f"Loaded experiment config from {args.todo}:")
    run_dino_experiment(cfg)
    evaluate_dino_experiment(cfg)


if __name__ == "__main__":
    main()
