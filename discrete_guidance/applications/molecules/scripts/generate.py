# Import public modules
import argparse
import torch
import collections
import copy
import time
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
# Import custom modules
from discrete_guidance.applications.molecules.src import bookkeeping
from discrete_guidance.applications.molecules.src import cheminf
from discrete_guidance.applications.molecules.src import config_handling
from discrete_guidance.applications.molecules.src import factory
from discrete_guidance.applications.molecules.src import logging
from discrete_guidance.applications.molecules.src import managers
from discrete_guidance.applications.molecules.src import plotting
from discrete_guidance.applications.molecules.src import utils
import matplotlib.pyplot as plt

from collections import defaultdict

# Only run as main
def diffusion_sample(args, predictor, oracle, round, dataset, ls_ratio, radius,target_property_value):

    # Parse arguments
    # parser = argparse.ArgumentParser(description='Train a model.')
    # parser.add_argument('-c', '--config',                     type=str, required=True, help='[Required] Path to (generation) config file.')
    # parser.add_argument('-n', '--num_valid_molecule_samples', type=int, required=True, help='[Required] Number of valid molecules to be sampled.')
    # parser.add_argument('-p', '--property_name_value',        type=str, default='',    help='[Optional] Which target property value the molecule generation should be guided to in the form "<property-name>=<property-value>" (for example "num_rings=0"). If this argument is not passed, use unconditional generation.')
    # parser.add_argument('-o', '--overrides',                  type=str, default='',    help='[Optional] Which configs (in the config file) to override (pass configuration names and override-values in the format "<config-name-1>=<config-value-1>|<config-name-2>=<config-value-2>"). If this argument is not passed, no configurations will be overriden.')
    # args = parser.parse_args()
    # args = defaultdict()
    # args.config = '../discrete_guidance/applications/molecules/config_files/generation_defaults.yaml'
    # args.num_valid_molecule_samples = 500
    # args.property_name_value = 'reward=1.0'
    # 가장 최근이나 seed별로 원하는 모델을 유동적으로 지정할 수 있게 추후 수정
    # 이걸 넣으면 path가 꼬이네
    # args.overrides = ''
    
    # Strip potenial '"' at beginning and end of args.overrides
    # args.overrides = args.overrides.strip('"')

    # Load the configs from the passed path to the config file
    generation_cfg = config_handling.load_cfg_from_yaml_file(args.config)
    generation_cfg.sampler.batch_size = args.gen_batch_size #* argument로 받은 guidetemp로 삽입입
    generation_cfg.sampler.guide_temp = args.guide_temp #* argument로 받은 guidetemp로 삽입입
    generation_cfg.seed = args.seed
    generation_cfg.sampler.x1_temp = args.diffusion_temp
    
    sequences, scores = dataset.get_all_data(return_as_str=False)
    score_mean = np.mean(scores)
    score_std = np.std(scores)
    
    # 여기서 문제가 생긴듯
    # generation_cfg.data.reward_mean = score_mean
    # generation_cfg.data.reward_std = score_std

    # Deepcopy the original cfg
    original_generation_cfg = copy.deepcopy(generation_cfg)

    # Parse the overrides
    overrides = config_handling.parse_overrides(args.overrides)

    # Update the configs with the overrides
    generation_cfg.update(overrides)
    
    generation_cfg.trained_run_folder_dir = f"./{args.run_folder_path}"

    # Parse the property name and target property value
    # args.property_name_value = args.property_name_value.strip('"')
    
    if args.property_name_value=='reward': # Default value if no property value was passed
        # property_name, target_property_value = args.property_name_value.split('=')
        property_name = args.property_name_value 
        target_property_value = target_property_value
        if target_property_value=='None':
            target_property_value = None
        else:
            target_property_value = float(target_property_value)
            if int(target_property_value)==target_property_value:
                target_property_value = int(target_property_value)

        run_folder_name_property_prefix = args.property_name_value
    else:
        property_name                   = 'None'
        target_property_value           = None
        run_folder_name_property_prefix = 'unconditional'

    # Extract the requested number of unique valid nswcs (uvnswcs) from the arguments
    num_uvnswcs_requested = int(args.num_valid_molecule_samples)
    # num_uvnswcs_requested = int(100*generation_cfg.sampler.max_iterations)

    # Create a folder for the current generation run
    # save_location = ./generated
    # save_location = str(Path(generation_cfg.base_dir, 'generated'))
    save_location = str(Path(generation_cfg.base_dir, args.gen_folder_path))
    # run_folder_name = num_rings1|n=1000
    run_folder_name = f"{run_folder_name_property_prefix}|n={num_uvnswcs_requested}"
    # 이건 path가 이상해져서 생략
    # if args.overrides!='':
    #     run_folder_name += f"|{args.overrides}"
    save_location = str(Path(args.gen_folder_path, run_folder_name))
    # outputs_dir = bookkeeping.create_run_folder(save_location, run_folder_name, include_time=False)

    outputs_dir = bookkeeping.create_run_folder(save_location, run_folder_name, include_time=False)


    # Define a logger
    log_file_path = str(Path(outputs_dir, 'logs'))
    logger = logging.define_logger(log_file_path, file_logging_level='INFO', stream_logging_level='DEBUG')

    # Set the logging level of matplotlib to 'info' (to avoid a plethora of irrelevant matplotlib DEBUG logs)
    plt.set_loglevel('info')

    # Log initial information
    if target_property_value is None:
        logger.info(f"Unconditional generation.")
    else:
        logger.info(f"Guided generation towards target: {property_name}={target_property_value}")
    logger.info(f"Generate until the sampled number of unique valid nswcs is: {num_uvnswcs_requested}")

    # Construct an orchestrator from a (trained) 'run folder' containing the saved model weights 
    # and other meta-information required to construct the models and generate from them.
    # Set some overrised for the train_cfg
    trained_overrides = {
        'make_figs': True,
        'save_figs': False,
    }
    config_handling.update_dirs_in_cfg(trained_overrides, outputs_dir=generation_cfg.trained_run_folder_dir)

    # Define an orchestrator from the run folder and overrides
    orchestrator = factory.Orchestrator.from_run_folder(run_folder_dir=generation_cfg.trained_run_folder_dir, overrides=trained_overrides, load_data=True, logger=logger)

    # Load all models
    # 여기에 훈련된 diffusion과 predictor 모델이 로드됨
    # manager.models_dict = {
    #     'denoising_model': denoising_model,'waiting_model': waiting_model, 'num_rings_predictor_model': num_rings_predictor_model, 'num_tokens_predictor_model': num_tokens_predictor_model, 'logp_predictor_model': logp_predictor_model, 'num_heavy_atoms_predictor_model': num_heavy_atoms_predictor_model
    orchestrator.manager.load_all_models()
    # orchestrator.manager.load_model('denoising_model')
    

    # Update the generation configurations without overwriting entries
    # 여기서 dict1에 없는 것만 append하는데 data를 만들어놓으니까 문제 발생
    # 그러면 애초에 train 폴더에 있는 config.yaml에 normalizing data를 넣을까?
    config_handling.update_without_overwrite(generation_cfg, orchestrator.cfg)

    # Update the directoris in the generation config file
    config_handling.update_dirs_in_cfg(generation_cfg, str(outputs_dir))

    # Update the 'save_figs' flag
    generation_cfg.save_figs = True
    
    # 이게 쓰이는 부분이 필요한데 아하 manager를 새로 만들때 포함되서 mg.generate에 쓰인다. 
    # 들어가는거 확인
    generation_cfg.data.reward_mean = score_mean
    generation_cfg.data.reward_std = score_std

    # Define manager for evaluation with trained model
    # 여기서 manager를 한번 더 만들어주는 이유는 generation_cfg가 update되었기 때문
    eval_manager = managers.DFMManager(generation_cfg, 
                                       denoising_model=orchestrator.manager.denoising_model,
                                       predictor_models_dict=orchestrator.manager.predictor_models_dict
                                    #    predictor_models_dict={}
                                       )

    # Log the overrides
    logger.info(f"Overrides: {overrides}")

    # Log the configs
    logger.info(f"Overriden config: {generation_cfg}")

    # Save the cfg, original_cfg, and overrides as yaml files in cfg.config_dir
    file_path = str(Path(generation_cfg.configs_dir, 'original_config.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, original_generation_cfg.to_dict())
    file_path = str(Path(generation_cfg.configs_dir, 'overrides.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, overrides)
    file_path = str(Path(generation_cfg.configs_dir, 'config.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, generation_cfg.to_dict())

    # Define a writer used to track training
    tensorboard_writer = bookkeeping.setup_tensorboard(generation_cfg.outputs_dir, rank=0)

    # Set a random seed
    utils.set_random_seed(generation_cfg.seed)

    # Generate until a certain number of unique valid nswcs (uvnswcs) has been sampled 
    sampled_uvnswcs_list = list() # Keep track of the sampled unique valid nswcs (uvnswcs)
    generated_df_list    = list()
    global_start_time    = time.time()
    logger.info(f"Will generate sequences are at least {num_uvnswcs_requested} sequences have been sampled.")
    total_x_generated = []
    
    # needed for excluding unvalid samples
    vocab_sizes = {
    'tfbind': 4,
    'rna1': 4,
    'rna2': 4, 
    'rna3': 4,
    'amp': 2,
    'gfp': 20,
    'aav': 20
    }
    mask_idx = vocab_sizes.get(args.task, 4)
    
    
    for iteration in range(generation_cfg.sampler.max_iterations):
        # 너무 짜치는데?
        if args.additive_guide_temp:
            generation_cfg.sampler.guide_temp = min(args.guide_temp_min + iteration * args.additive_guide_temp, 2.0)
            print(f"generation_cfg.sampler.guide_temp: {generation_cfg.sampler.guide_temp}")
        # If no property is specified, use unconditional sampling
        if target_property_value is None:
            x_generated = eval_manager.generate(num_samples=generation_cfg.sampler.batch_size, 
                                                seed=None, # Only use the external seed
                                                stochasticity=generation_cfg.sampler.noise,
                                                dt=generation_cfg.sampler.dt,
                                                batch_size=generation_cfg.sampler.batch_size,
                                                predictor = predictor,
                                                dataset=dataset,
                                                ls_ratio=ls_ratio,
                                                target_reward=target_property_value,)
        else:
            # Construct the predictor model name based on the property name
            predictor_model_name = f"{property_name}_predictor_model"
            logger.info(f"Predictor model name: {predictor_model_name}")

            # Check that the predictor model is valid
            if predictor_model_name not in orchestrator.manager.models_dict:
                err_msg = f"There is no predictor model with name '{predictor_model_name}'. Allowed predictor models are: {list(orchestrator.manager.models_dict.keys())}"
                raise ValueError(err_msg)

            # 현재시각 06/05 00:47 여기까지 왔다.
            # Here we use generation cfg
            x_generated = eval_manager.generate(num_samples=generation_cfg.sampler.batch_size, #500, 
                                                seed=None, # Only use the external seed
                                                stochasticity=generation_cfg.sampler.noise,
                                                dt=generation_cfg.sampler.dt,
                                                predictor_y_dict={predictor_model_name: target_property_value},
                                                guide_temp=generation_cfg.sampler.guide_temp,
                                                grad_approx=generation_cfg.sampler.grad_approx,
                                                batch_size=generation_cfg.sampler.batch_size,
                                                predictor = predictor,
                                                dataset=dataset,
                                                ls_ratio=ls_ratio,
                                                radius=radius,
                                                target_reward=target_property_value)

         
        # for proxy evaluation
        total_x_generated.append(x_generated)
        
        valid_samples = []
        for sample in x_generated:
            sample_np = sample.cpu().numpy() if torch.is_tensor(sample) else sample
            if mask_idx not in sample_np:
                valid_samples.append(sample)
            else:
                logger.warning(f"Found mask token in generated sample, skipping: {sample_np}")
        
        # Analyze the generated x
        # filtering here?
        
        # Decode the generated x to smiles (int->smiles)
        # generated_smiles_list = [orchestrator.molecules_data_handler.smiles_encoder.decode(utils.to_numpy(smiles_encoded)) for smiles_encoded in x_generated]
        generated_smiles_list = [orchestrator.molecules_data_handler.smiles_encoder.decode(utils.to_numpy(smiles_encoded)) for smiles_encoded in valid_samples]

        generated_df_list.extend(generated_smiles_list)
        
        # if len(generated_df_list) > num_uvnswcs_requested  :
        # if not args.filter:
        #     args.K = 1
        # if len(generated_df_list) >= num_uvnswcs_requested * args.K :
        #     if  not args.filter:
        #         generated_df_list = generated_df_list[:num_uvnswcs_requested]
        #     else:
        #         total_x = np.vstack(total_x_generated)[:num_uvnswcs_requested * args.K, :]
        #         batch_data_t = {}
        #         if isinstance(total_x, np.ndarray):
        #             x = torch.from_numpy(total_x).to(args.device)
        #         else:
        #             x = total_x
        #         batch_data_t['x'] = x
        #         t =  torch.ones(len(total_x), dtype=torch.long).to(args.device)
        #         vals_proxy = predictor(batch_data_t, t ,is_x_onehot=False)
        #         vals_proxy = vals_proxy.detach().cpu().numpy()
        #         total_x = total_x[np.argsort(vals_proxy)[-num_uvnswcs_requested:], :]
        #         generated_df_list = list()
        #         for x in total_x:
        #             generated_smiles_list = [orchestrator.molecules_data_handler.smiles_encoder.decode(utils.to_numpy(x))]
        #             generated_df_list.extend(generated_smiles_list)
        #     break
        
        # 06/02 invalid sample removal
        if not args.filter:
            args.K = 1
        if len(generated_df_list) >= num_uvnswcs_requested * args.K:
            if not args.filter:
                generated_df_list = generated_df_list[:num_uvnswcs_requested]
            else:
                # total_x_generated를 flat하게 만들되, mask token이 있는 샘플은 제외
                valid_total_x = []
                for batch in total_x_generated:
                    for sample in batch:
                        sample_np = sample.cpu().numpy() if torch.is_tensor(sample) else sample
                        if mask_idx not in sample_np:
                            valid_total_x.append(sample_np)
                
                # 충분한 샘플이 있는지 확인
                if len(valid_total_x) >= num_uvnswcs_requested * args.K:
                    total_x = np.vstack(valid_total_x[:num_uvnswcs_requested * args.K])
                    
                    batch_data_t = {}
                    if isinstance(total_x, np.ndarray):
                        x = torch.from_numpy(total_x).to(args.device)
                    else:
                        x = total_x
                    batch_data_t['x'] = x
                    t = torch.ones(len(total_x), dtype=torch.long).to(args.device)
                    vals_proxy = predictor(batch_data_t, t, is_x_onehot=False)
                    vals_proxy = vals_proxy.detach().cpu().numpy()
                    total_x = total_x[np.argsort(vals_proxy)[-num_uvnswcs_requested:], :]
                    
                    generated_df_list = list()
                    for x in total_x:
                        generated_smiles_list = [orchestrator.molecules_data_handler.smiles_encoder.decode(utils.to_numpy(x))]
                        generated_df_list.extend(generated_smiles_list)
                    break
                else:
                    logger.info(f"Not enough valid samples yet. Have {len(valid_total_x)}, need {num_uvnswcs_requested * args.K}")
                    continue
            break
        
    seen_seqs = set()
    for seq in total_x_generated:
        for x in seq:
            x_tuple = tuple(x.flatten())
            seen_seqs.add(x_tuple)
    total_x_uniqueness = len(seen_seqs) / (128 * args.K) #* gen batch size가 128보다 작으면 128에 맞게 추가로 돌아감.
    args.total_x_uniqueness = total_x_uniqueness

    # # stack generated samples , shape (500, 8)
    # if not args.filter:
    #     total_x  = np.vstack(total_x_generated)[:num_uvnswcs_requested, :]
    
    if not args.filter:
        valid_total_x = []
        for batch in total_x_generated:
            for sample in batch:
                sample_np = sample.cpu().numpy() if torch.is_tensor(sample) else sample
                if mask_idx not in sample_np:
                    valid_total_x.append(sample_np)
        
        if len(valid_total_x) >= num_uvnswcs_requested:
            total_x = np.vstack(valid_total_x[:num_uvnswcs_requested])
        else:
            logger.error(f"Not enough valid samples! Got {len(valid_total_x)}, needed {num_uvnswcs_requested}")
            # 부족한 경우 있는 것만 사용
            total_x = np.vstack(valid_total_x)
    
    # print(f"total_x {total_x}")
    for x in total_x:
        print(f"x {x}")
    # evaluate the generated samples by oracle
    total_x = total_x.astype(int)
    vals = oracle(total_x).reshape(-1)
    
    batch_data_t = {}
    if isinstance(total_x, np.ndarray):
        x = torch.from_numpy(total_x).to(args.device)
    else:
        x = total_x
    batch_data_t['x'] = x
    t =  torch.ones(len(vals), dtype=torch.long).to(args.device)
    proxy_scores = predictor(batch_data_t,t,is_x_onehot=False)
    proxy_scores = proxy_scores.detach().cpu().numpy()
    
    # total_X를 다시 generated_df_list로 바꿈
    

    # Save this DataFrame
    file_name = f'samples_table_t{generation_cfg.sampler.guide_temp}_w{target_property_value}_n{num_uvnswcs_requested}_r{round}.tsv'
    file_path = str(Path(generation_cfg.outputs_dir, file_name))
    # generated_df_list = pd.Series(generated_df_list)
    generated_df = pd.DataFrame({
    'sequence': generated_df_list,
    'reward': vals
    })
    generated_df.to_csv(file_path, index=False, sep='\t')



    # 히스토그램 그리기
    plt.figure(figsize=(8,6))
    plt.hist(vals, bins=30, color='blue', alpha=0.7)
    plt.title(f'Histogram of round{round}')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    hist_file_name = file_name.replace('.tsv', '_hist.png')
    hist_file_path = str(Path(generation_cfg.outputs_dir, hist_file_name))

    plt.savefig(hist_file_path)
    plt.close()

    logger.info(f"Stored the samples as table in: {file_path}")
    logger.info(f"Number of unique valid nswcs: {len(generated_df)}")

    return (total_x.tolist(), vals), proxy_scores




    # # Get all unique valid smiles
    # filtered_df = generated_df[generated_df['valid']==True]
    # unique_valid_gen_smiles_list = list(set(filtered_df['smiles']))

    # # Only make a plot if there are any unique valid generated smiles
    # if len(unique_valid_gen_smiles_list)>0 and generation_cfg.make_figs:
    #     property_names = ['num_tokens', 'logp', 'num_rings', 'num_heavy_atoms']
    #     fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    #     if target_property_value is None:
    #         guide_temp_label = None
    #     else:
    #         guide_temp_label = generation_cfg.sampler.guide_temp
    #     plt.suptitle(f"Target {property_name}: {target_property_value} | Stochasticity: {generation_cfg.sampler.noise} | T: {guide_temp_label}")
    #     for index, property_name in enumerate(property_names):
    #         index1 = index%2
    #         index2 = (index-index1)//2
    #         ax = axs[index1, index2]
    #         plotting.plot_gen_vs_train_distribution(property_name, 
    #                                                 orchestrator.molecules_data_handler.subset_df_dict['train'], 
    #                                                 unique_valid_gen_smiles_list,
    #                                                 ax=ax)
            
    #     # Save the figure
    #     if generation_cfg.save_figs and generation_cfg.figs_save_dir is not None:
    #         file_path = str(Path(generation_cfg.figs_save_dir, f"Visualization_samples.png"))
    #         fig.savefig(file_path)
                        