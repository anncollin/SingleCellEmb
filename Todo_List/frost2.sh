#!/bin/bash

# Macros
main_folder="/home/anncollin/SingleCellEmb"

# ------------------------------------------------------------
# Function: wait_for_and_run
# Parameters:
#   $1 = exp_name
#   $2 = td_folder
# ------------------------------------------------------------
wait_for_and_run() {

    exp_name="$1"
    td_folder="$2"

    FILE="${main_folder}/Results/${exp_name}/DINO_${exp_name}_EMD_callibration.csv"

    echo "============================================"
    echo "Waiting for:  ${FILE}"
    echo "Experiment:   ${exp_name}"
    echo "Todo version: ${td_folder}"
    echo "============================================"
    echo ""

    # Wait loop
    while [ ! -e "$FILE" ]; do
        echo "$(date '+%H:%M:%S')  File not found yet…"
        sleep 120
    done

    python "${main_folder}/main.py" \
        --todo "${main_folder}/Todo_List/${td_folder}/${exp_name}.yaml" \
        --eval

}

# ------------------------------------------------------------
# Call the function for one or multiple experiments
# ------------------------------------------------------------

wait_for_and_run "tiny8-256_brightDAPI+EGFP_zoom90-100" "td2"

