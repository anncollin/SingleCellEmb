#!/bin/bash

exp_name="tiny8-256_brightDAPI_zoom90-100"
td_folder="td2"

FILE="/home/anncollin/SingleCellEmb/Results/${exp_name}/${exp_name}_EMD_callibration.csv"

# Wait until the file exists
while [ ! -e "$FILE" ]; do
    sleep 10
done

# Run your command
python main_py --todo Todo_List/${td_folder}/${exp_name}.yaml --eval

