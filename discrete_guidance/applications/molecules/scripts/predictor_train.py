# Import public modules
import argparse
import copy
import matplotlib.pyplot as plt
from pathlib import Path

# Import custom modules
from discrete_guidance.applications.molecules.src import bookkeeping
from discrete_guidance.applications.molecules.src import config_handling
from discrete_guidance.applications.molecules.src import factory
from discrete_guidance.applications.molecules.src import logging
from discrete_guidance.applications.molecules.src import model_training
from discrete_guidance.applications.molecules.src import plotting

from collections import defaultdict

# Only run as main
# args, round, dataset를 받아서 알맞은 곳에서 모델을 불러오고, dataloader도 구성해
# 훌륭한 weighted sampple을 그대로 쓸 수 있으니까 이 diffusion_train 메서드에서
# dataLoader를 만들지 말고 dataset을 받아서 가공한다음 train_dataloader를 새로 만들어
# 대신 형태변환은 해줘야지
def predictor_train(args, round_idx, dataset):
    # Parse arguments
    # parser = argparse.ArgumentParser(description='Train a model.')
    # parser.add_argument('-c', '--config',    type=str, required=True, help='[Required] Path to (training) config file.')
    # parser.add_argument('-m', '--model',     type=str, required=True, help='[Required] Which model to train.\nUse "all" to train all models. Use "all_predictors" to train all predictor models (but not the denoising model). Use <model-name> (e.g. "denoising_model", "num_rings_predictor_model", "logp_predictor_model", or "num_heavy_atoms_predictor_model") to train a specific model.')
    # parser.add_argument('-o', '--overrides', type=str, default='',    help='[Optional] Which configs (in the config file) to override (pass configuration names and override-values in the format "<config-name-1>=<config-value-1>|<config-name-2>=<config-value-2>"). If this argument is not specified, no configurations will be overriden.')
    # args = parser.parse_args()
    # args = defaultdict()
    # args.config = './config_files/training_defaults_sequence.yaml'
    # args.config = '../discrete_guidance/applications/molecules/config_files/training_defaults_sequence.yaml'
    # args.model = 'reward_predictor_model'
    # args.overrides = ''

    # Load the configs from the passed path to the config file
    cfg = config_handling.load_cfg_from_yaml_file(args.config)

    # Deepcopy the original cfg
    original_cfg = copy.deepcopy(cfg)

    # Strip potenial '"' at beginning and end of args.overrides
    # args.overrides = args.overrides.s trip('"')

    # Parse the overrides
    overrides = config_handling.parse_overrides(args.overrides)

    # Update the configs with the overrides
    cfg.update(overrides)

    # Create a folder for the current training run
    if round_idx == 0:
        save_location = str(Path(cfg.base_dir, args.run_folder_path))
        outputs_dir = bookkeeping.create_run_folder(save_location, '', include_time=False)
    else:
        save_location = str(Path(cfg.base_dir, args.run_folder_path))
        outputs_dir = Path(save_location)

    config_handling.update_dirs_in_cfg(cfg, str(outputs_dir))

    # save_location = str(Path(cfg.base_dir, 'trained'))
    # if args.overrides=='':
    #     print('No overrides specified. Using default folder name.')
    #     run_folder_name = 'no_overrides'
    # else:
    #     run_folder_name = args.overrides
    # outputs_dir = bookkeeping.create_run_folder(save_location, run_folder_name, include_time=False)
    # config_handling.update_dirs_in_cfg(cfg, str(outputs_dir))

    # Define a logger
    log_file_path = str(Path(cfg.outputs_dir, 'logs'))
    logger = logging.define_logger(log_file_path, file_logging_level='INFO', stream_logging_level='DEBUG')

    # Set the logging level of matplotlib to 'info' (to avoid a plethora of irrelevant matplotlib DEBUG logs)
    plt.set_loglevel('info')

    # Log the overrides
    logger.info(f"Overrides: {overrides}")

    # Generate orchestrator from cfg
    # Remark: This will update cfg
    orchestrator = factory.Orchestrator(cfg, logger=logger)

    # Log the configs
    # What we care is here
    logger.info(f"Overriden config: {cfg}")

    # Save the cfg, original_cfg, and overrides as yaml files in cfg.config_dir
    file_path = str(Path(cfg.configs_dir, 'original_config.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, original_cfg.to_dict())
    file_path = str(Path(cfg.configs_dir, 'overrides.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, overrides)
    file_path = str(Path(cfg.configs_dir, 'config.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, cfg.to_dict())

    # Define a writer used to track training
    tensorboard_writer = bookkeeping.setup_tensorboard(cfg.outputs_dir, rank=0)

    # Determine the list of models to be trained
    if args.model=='all':
        # Train all models
        train_models = list(orchestrator.manager.models_dict.keys())
    elif args.model=='all_predictors':
        # Train all predictor models
        train_models = list(orchestrator.manager.predictor_models_dict.keys())
    elif args.model in orchestrator.manager.models_dict:
        train_models = [args.model]
    else:
        err_msg = f"Unknown model name. The passed 'model' must be either 'all' (train all models) or one of the defined models, which are: {list(orchestrator.manager.models_dict.keys())}"
        raise ValueError(err_msg)

    # Log the models to be trained
    logger.info(f"Models to be trained: {train_models}")

    # Loop over to be trained models
    for model_name in train_models:
        if model_name=='denoising_model':
            # Denoising model
            train_dataloader = orchestrator.dataloader_dict['train']
            print('I will train the denoising model')
        else:
            # Property models
            y_guide_name = orchestrator.manager.models_dict[model_name].y_guide_name
            property_set_name = f"train_{y_guide_name}"
            if property_set_name in orchestrator.dataloader_dict:
                logger.info(f"Using the property-model specific dataloader of the '{property_set_name}' set.")
                train_dataloader = orchestrator.dataloader_dict[property_set_name]
            else:
                logger.info(f"Using the dataloader of the 'train' set.")
                train_dataloader = orchestrator.dataloader_dict['train']

        # Train the model
        model_training.train_model(orchestrator.manager, 
                                   train_dataloader, 
                                   which_model=model_name, 
                                   num_epochs=cfg.training[model_name].num_epochs,
                                   validation_dataloader=orchestrator.dataloader_dict['validation'], 
                                   random_seed=cfg.training[model_name].seed,
                                   plot_training_curves=cfg.make_figs,
                                   figs_save_dir=cfg.figs_save_dir,
                                   tensorboard_writer=tensorboard_writer,
                                   logger=logger)

        # Save the trained model (allowing overwriting)
        orchestrator.manager.save_model(model_name, overwrite=True)

        # Make some figures in case the model is a predictor model (i.e. not the denoising model)
        if model_name!='denoising_model':
            if cfg.make_figs:
                # Sigma(t) plot
                fig = orchestrator.manager.models_dict[model_name].plot_sigma_t()
                if cfg.save_figs and cfg.figs_save_dir is not None:
                    file_path = str(Path(cfg.figs_save_dir, f"log_sigma_t_{model_name}.png"))
                    fig.savefig(file_path)

                # modified 0423 plotting error
                # # Correlation model to gt on train set
                # fig = plotting.make_correlation_plot(model_name, 
                #                                        orchestrator,
                #                                        set_name=property_set_name,
                #                                        t_eval=1, 
                #                                        seed=42)
                # if cfg.save_figs and cfg.figs_save_dir is not None:
                #     file_path = str(Path(cfg.figs_save_dir, f"correlation_plot_train_{model_name}.png"))
                #     fig.savefig(file_path)

                # # Correlation model to gt on validation set
                # fig = plotting.make_correlation_plot(model_name, 
                #                                        orchestrator,
                #                                        set_name='validation',
                #                                        t_eval=1, 
                #                                        seed=42)
                # if cfg.save_figs and cfg.figs_save_dir is not None:
                #     file_path = str(Path(cfg.figs_save_dir, f"correlation_plot_validation_{model_name}.png"))
                #     fig.savefig(file_path)
        
        if 1<len(train_models):
            logger.info('-'*100)


    orchestrator.manager.load_model(model_name) 
    model = orchestrator.manager.models_dict[model_name]
    model.to(cfg.device)
    model.eval()
    return model