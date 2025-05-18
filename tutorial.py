import os
import argparse
import subprocess
import xml.etree.ElementTree as ET

import random
import numpy as np

import traci
from sumolib import checkBinary
from sumolib.miscutils import getFreeSocketPort


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Network parameters
    parser.add_argument("--grid_number", type=int, default=12)
    parser.add_argument("--grid_length", type=float, default=50.0)
    
    # Route parameters
    parser.add_argument("--route_number", type=int, default=20)
    parser.add_argument("--min_num_vehicles", type=int, default=100)
    parser.add_argument("--max_num_vehicles", type=int, default=200)
    
    # Simulation parameters
    parser.add_argument("--simulation_time", type=int, default=1800)
    
    # seed
    parser.add_argument("--seed", type=int, default=42)
    
    # visualize
    parser.add_argument("--visualize", action="store_true")
    
    args = parser.parse_args()
    
    # Set seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Construct network file
    # Network should be open. Remove corner edges
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    remove_edges = []
    for i in range(args.grid_number):
        for j in range(args.grid_number):
            if i == 0 and j < args.grid_number - 1:
                remove_edges.append(f"{alphabet[i]}{j}{alphabet[i]}{j+1}")
                remove_edges.append(f"{alphabet[i]}{j+1}{alphabet[i]}{j}")
            if j == 0 and i < args.grid_number - 1:
                remove_edges.append(f"{alphabet[i]}{j}{alphabet[i+1]}{j}")
                remove_edges.append(f"{alphabet[i+1]}{j}{alphabet[i]}{j}")
            if i == args.grid_number - 1 and j < args.grid_number - 1:
                remove_edges.append(f"{alphabet[i]}{j}{alphabet[i]}{j+1}")
                remove_edges.append(f"{alphabet[i]}{j+1}{alphabet[i]}{j}")
            if j == args.grid_number - 1 and i < args.grid_number - 1:
                remove_edges.append(f"{alphabet[i]}{j}{alphabet[i+1]}{j}")
                remove_edges.append(f"{alphabet[i+1]}{j}{alphabet[i]}{j}")
    remove_edges = ', '.join(remove_edges)
    
    folder_name = f"sumo/N{args.grid_number}_L{args.grid_length}/R{args.route_number}_MIN{args.min_num_vehicles}_MAX{args.max_num_vehicles}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)

    network_file = f"{folder_name}/default.net.xml"
    if os.path.exists(network_file):
        print(f"File {network_file} already exists. Use existing file.")
    else:
        # Using netgenerate to create network file
        subprocess.call(f'netgenerate --grid --grid.number {args.grid_number} --grid.length {args.grid_length} --output-file {network_file} --remove-edges.explicit "{remove_edges}"', shell=True)
    
    # Construct route file
    # Extract start and end edges
    start_edges_up, start_edges_down, start_edges_left, start_edges_right = [], [], [], []
    end_edges_up, end_edges_down, end_edges_left, end_edges_right = [], [], [], []
    for i in range(args.grid_number):
        if i == 0:
            for j in range(args.grid_number):
                if j != 0 and j != args.grid_number - 1:
                    start_edges_left.append(f"{alphabet[i]}{j}{alphabet[i+1]}{j}")
                    end_edges_left.append(f"{alphabet[i+1]}{j}{alphabet[i]}{j}")
        elif i == args.grid_number - 1:
            for j in range(args.grid_number):
                if j != 0 and j != args.grid_number - 1:
                    start_edges_right.append(f"{alphabet[i]}{j}{alphabet[i-1]}{j}")
                    end_edges_right.append(f"{alphabet[i-1]}{j}{alphabet[i]}{j}")
        else:
            start_edges_up.append(f"{alphabet[i]}0{alphabet[i]}1")
            end_edges_up.append(f"{alphabet[i]}1{alphabet[i]}0")
            start_edges_down.append(f"{alphabet[i]}{args.grid_number-1}{alphabet[i]}{args.grid_number-2}")
            end_edges_down.append(f"{alphabet[i]}{args.grid_number-2}{alphabet[i]}{args.grid_number-1}")
    
    start_edges = [start_edges_up, start_edges_down, start_edges_left, start_edges_right]
    end_edges = [end_edges_up, end_edges_down, end_edges_left, end_edges_right]
    
    # Randomly generate routes
    routes = []
    for _ in range(args.route_number):
        i, j = np.random.choice(4, 2, replace=False)
        start_edge = np.random.choice(start_edges[i])
        end_edge = np.random.choice(end_edges[j])
        num_vehicles = np.random.randint(args.min_num_vehicles, args.max_num_vehicles)
        routes.append((start_edge, end_edge, num_vehicles))
        
    route_file = f"{folder_name}/default.rou.xml"
    if os.path.exists(route_file):
        print(f"File {route_file} already exists. Use existing file.")
    else:
        with open(route_file, 'w') as f:
            f.write(f'<routes>\n')
            for i, (start_edge, end_edge, num_vehicles) in enumerate(routes):
                f.write(f'    <flow id="{i}" begin="0" end="{args.simulation_time}" from="{start_edge}" to="{end_edge}" number="{num_vehicles}" />\n')
            f.write('</routes>\n')
        
    sumocfg_file = f"{folder_name}/default.sumocfg"
    tripinfo_file = f"{folder_name}/default.tripinfo.xml"
    if os.path.exists(sumocfg_file):
        print(f"File {sumocfg_file} already exists. Use existing file.")
    else:
        # Construct sumo command
        with open(sumocfg_file, 'w') as f:
            f.write(f'<configuration>\n')
            f.write(f'    <input>\n')
            f.write(f'        <net-file value="default.net.xml"/>\n')
            f.write(f'        <route-files value="default.rou.xml"/>\n')
            f.write(f'    </input>\n')
            f.write(f'    <time>\n')
            f.write(f'        <begin value="0"/>\n')
            f.write(f'        <end value="{args.simulation_time}"/>\n')
            f.write(f'    </time>\n')
            f.write(f'</configuration>\n')
        
    if args.visualize:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    sumo_cmd = [sumoBinary, '-c', sumocfg_file, '--no-warnings', '--no-step-log', '--tripinfo-output', tripinfo_file]
    port = getFreeSocketPort()
    
    # Simulation
    traci.start(sumo_cmd, port=port)
    # for _ in range(args.simulation_time):
    while traci.simulation.getMinExpectedNumber() != 0:
        traci.simulationStep()
    traci.close()
    
    # Post-processing
    waiting_time_list = []
    traveling_time_list = []
    last_vehicle_arrival_time = 0
    tree = ET.parse(tripinfo_file)
    root = tree.getroot()
    for child in root:
        waiting_time_list.append(float(child.attrib['waitingTime']))
        traveling_time_list.append(float(child.attrib['duration']))
        last_vehicle_arrival_time = max(last_vehicle_arrival_time, float(child.attrib['arrival']))
    print(f"Average waiting time: {np.mean(waiting_time_list):.2f}", end="\t")
    print(f"Average travel time: {np.mean(traveling_time_list):.2f}", end="\t")
    print(f"Last vehicle arrival time: {last_vehicle_arrival_time:.2f}")
        
    
    
    