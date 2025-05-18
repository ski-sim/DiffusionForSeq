import os
import numpy as np
import csv

def collect_and_save_tsv(data_folder, output_tsv):
    tsv_data = []
    
    for file in os.listdir(data_folder):
        if file.startswith("random") and file.endswith(".npz"):
            file_path = os.path.join(data_folder, file)
            data = np.load(file_path)
            
            data_x = data["X"]  # (10, 144)
            data_y = data["y"]  # (10, 3)
            
            for x, y in zip(data_x, data_y):
                layout = "".join(map(str, x))
                waiting_time = y[0]  # y의 첫 번째 값
                tsv_data.append([layout, waiting_time])
    
    with open(output_tsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["layout", "waiting_time"])
        writer.writerows(tsv_data)

# 나중에 argparse로 받을 수 있도록 수정
data_folder = "/home/son9ih/gs-project/data" 
output_tsv = "/home/son9ih/gs-project/discrete_guidance/applications/molecules/data/preprocessed/sumo_preprocessed_dataset.tsv"
collect_and_save_tsv(data_folder, output_tsv)
