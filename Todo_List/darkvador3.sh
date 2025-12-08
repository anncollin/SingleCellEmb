#!/bin/bash

exp_name="tiny8-256_NoBlur+NoBright_zoom70-90"
td_folder="td3"

FILE="/home/anncollin/SingleCellEmb/Results/${exp_name}/${exp_name}_EMD_callibration.csv"

# Wait until the file exists
while [ ! -e "$FILE" ]; do
    sleep 10
done

# Run your command
python main_py --todo Todo_List/${td_folder}/${exp_name}.yaml --eval

############################################
exp_namebis="tiny8-256_sharpness_zoom70-90"

FILEbis="/home/anncollin/SingleCellEmb/Results/${exp_name}/${exp_name}_EMD_callibration.csv"

# Wait until the file exists
while [ ! -e "$FILEbis" ]; do
    sleep 10
done

# Run your command
python main_py --todo Todo_List/${td_folder}/${exp_namebis}.yaml --eval
