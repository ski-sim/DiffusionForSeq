import numpy as np
import copy
from pathlib import Path
# Import custom modules
from discrete_guidance.applications.molecules.src import bookkeeping
from discrete_guidance.applications.molecules.src import config_handling
from discrete_guidance.applications.molecules.src import logging
import csv
import matplotlib.pyplot as plt
def initialize_config(args, round_idx):
    """Initialize and setup configuration for training."""
    # Load the configs from the passed path to the config file
    cfg = config_handling.load_cfg_from_yaml_file(args.config)
    cfg.seed = args.seed
    cfg.training.denoising_model.num_epochs = args.denoising_model_epoch
    cfg.training.reward_predictor_model.num_epochs = args.predictor_model_epoch

    # Deepcopy the original cfg
    original_cfg = copy.deepcopy(cfg)

    # Parse the overrides
    overrides = config_handling.parse_overrides(args.overrides)

    # Update the configs with the overrides
    cfg.update(overrides)
    
    return cfg, original_cfg, overrides

def setup_directories(cfg, args, round_idx):
    """Setup output directories and logging."""
    # Create a folder for the current training run
    if round_idx == 0:
        save_location = str(Path(cfg.base_dir, args.run_folder_path))
        outputs_dir = bookkeeping.create_run_folder(save_location, '', include_time=False)
    else:
        # For rounds > 0, use the same directory structure as round 0
        save_location = str(Path(cfg.base_dir, args.run_folder_path))
        outputs_dir = Path(save_location)

    config_handling.update_dirs_in_cfg(cfg, str(outputs_dir))

    # Define a logger
    log_file_path = str(Path(cfg.outputs_dir, 'logs'))
    logger = logging.define_logger(log_file_path, file_logging_level='INFO', stream_logging_level='DEBUG')

    # Set the logging level of matplotlib to 'info'
    plt.set_loglevel('info')
    
    return outputs_dir, logger

def preprocess_dataset(args, dataset, cfg):
    """Preprocess dataset and create CSV file."""
    sequences, scores = dataset.get_all_data(return_as_str=False)
    score_mean = np.mean(scores).item()
    score_std = np.std(scores).item()
    normalized_scores = (scores - score_mean) / score_std + 1e-8
    
    csv_data = []
    if args.task in ['aav', 'gfp', 'tfbind', 'rna1', 'rna2', 'rna3']:
        cfg.data.preprocessing.over_ten_unique_tokens = True
        for seq, score, norm_score in zip(sequences, scores, normalized_scores):
            sequence = ' '.join([str(i) for i in seq])
            csv_data.append([sequence, float(score), float(norm_score)])
    else:
        cfg.data.preprocessing.over_ten_unique_tokens = False
        for seq, score, norm_score in zip(sequences, scores, normalized_scores):
            sequence = "".join(map(str, seq))
            csv_data.append([sequence, float(score), float(norm_score)])

    sequence_data_path = args.preprocessed_dataset_path
    with open(sequence_data_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["sequence", "reward", "normalized_reward"])
        writer.writerows(csv_data)
    
    return sequence_data_path, score_mean, score_std

def save_configs(cfg, original_cfg, overrides):
    """Save configuration files."""
    file_path = str(Path(cfg.configs_dir, 'original_config.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, original_cfg.to_dict())
    file_path = str(Path(cfg.configs_dir, 'overrides.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, overrides)
    file_path = str(Path(cfg.configs_dir, 'config.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, cfg.to_dict())
