import argparse
import gzip
import pickle
import itertools
import time
import wandb

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

from lib.acquisition_fn import get_acq_fn
from lib.dataset import get_dataset
from lib.generator import get_generator
from lib.logging import get_logger
from lib.oracle_wrapper import get_oracle
from lib.proxy import get_proxy_model
from lib.utils.distance import is_similar, edit_dist
from lib.utils.env import get_tokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results/test_mlp.pkl.gz')
parser.add_argument("--tb_log_dir", default='results/test_mlp')
parser.add_argument("--name", default='test_mlp')
parser.add_argument("--load_scores_path", default='.')

# Multi-round
parser.add_argument("--num_rounds", default=10, type=int)
parser.add_argument("--task", default="amp", type=str)
parser.add_argument("--num_sampled_per_round", default=256*4, type=int) # 10k
parser.add_argument("--num_folds", default=5)
parser.add_argument("--vocab_size", default=21)
parser.add_argument("--max_len", default=65)
parser.add_argument("--gen_max_len", default=50+1)
parser.add_argument("--proxy_uncertainty", default="dropout")
parser.add_argument("--save_scores_path", default=".")
parser.add_argument("--save_scores", action="store_true")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--run", default=-1, type=int)
parser.add_argument("--noise_params", action="store_true")
parser.add_argument("--enable_tensorboard", action="store_true")
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--save_proxy_weights", action="store_true")
parser.add_argument("--use_uncertainty", action="store_true")
parser.add_argument("--filter", action="store_true")
parser.add_argument("--kappa", default=0.1, type=float)
parser.add_argument("--acq_fn", default="none", type=str)
parser.add_argument("--load_proxy_weights", type=str)
parser.add_argument("--max_percentile", default=80, type=int)
parser.add_argument("--filter_threshold", default=0.1, type=float)
parser.add_argument("--filter_distance_type", default="edit", type=str)
parser.add_argument("--oracle_split", default="D2_target", type=str)
parser.add_argument("--proxy_data_split", default="D1", type=str)
parser.add_argument("--oracle_type", default="MLP", type=str)
parser.add_argument("--oracle_features", default="AlBert", type=str)
parser.add_argument("--medoid_oracle_dist", default="edit", type=str)
parser.add_argument("--medoid_oracle_norm", default=1, type=int)
parser.add_argument("--medoid_oracle_exp_constant", default=6, type=int)


# Generator
parser.add_argument("--gen_learning_rate", default=5e-4, type=float)
parser.add_argument("--gen_Z_learning_rate", default=1e-3, type=float)
parser.add_argument("--gen_clip", default=10, type=float)
parser.add_argument("--gen_num_iterations", default=20000, type=int) # Maybe this is too low?
parser.add_argument("--gen_episodes_per_step", default=16, type=int)
parser.add_argument("--gen_num_hidden", default=128, type=int)
parser.add_argument("--gen_reward_norm", default=1, type=float)
parser.add_argument("--gen_reward_exp", default=3, type=float)
parser.add_argument("--gen_reward_min", default=-8, type=float)
parser.add_argument("--gen_L2", default=0, type=float)
parser.add_argument("--gen_partition_init", default=50, type=float)

# Soft-QLearning/GFlownet gen
parser.add_argument("--gen_reward_exp_ramping", default=1, type=float)
parser.add_argument("--gen_balanced_loss", default=1, type=float)
parser.add_argument("--gen_output_coef", default=10, type=float)
parser.add_argument("--gen_loss_eps", default=1e-5, type=float)
parser.add_argument("--gen_random_action_prob", default=0.001, type=float)
parser.add_argument("--gen_sampling_temperature", default=1., type=float)
parser.add_argument("--gen_leaf_coef", default=25, type=float)
parser.add_argument("--gen_data_sample_per_step", default=16, type=int)
# PG gen
parser.add_argument("--gen_do_pg", default=0, type=int)
parser.add_argument("--gen_pg_entropy_coef", default=1e-2, type=float)
# learning partition Z explicitly
parser.add_argument("--gen_do_explicit_Z", default=0, type=int)
parser.add_argument("--gen_model_type", default="mlp")

############# temp!!
parser.add_argument("--rank_based_proxy_training", action="store_true")

# Proxy
parser.add_argument("--proxy_learning_rate", default=1e-4)
parser.add_argument("--proxy_type", default="regression")
parser.add_argument("--proxy_arch", default="mlp")
parser.add_argument("--proxy_num_layers", default=2)
parser.add_argument("--proxy_dropout", default=0.1)

parser.add_argument("--proxy_num_hid", default=2048, type=int)
parser.add_argument("--proxy_L2", default=1e-4, type=float)
parser.add_argument("--proxy_num_per_minibatch", default=256, type=int)
parser.add_argument("--proxy_early_stop_tol", default=5, type=int)
parser.add_argument("--proxy_early_stop_to_best_params", default=0, type=int)
parser.add_argument("--proxy_num_iterations", default=30000, type=int)
parser.add_argument("--proxy_num_dropout_samples", default=25, type=int)


class MbStack:
    def __init__(self, f):
        self.stack = []
        self.f = f

    def push(self, x, i):
        self.stack.append((x, i))

    def pop_all(self):
        if not len(self.stack):
            return []
        with torch.no_grad():
            ys = self.f([i[0] for i in self.stack])
        idxs = [i[1] for i in self.stack]
        self.stack = []
        return zip(ys, idxs)


def filter_len(x, y, max_len):
    res = ([], [])
    for i in range(len(x)):
        if len(x[i]) < max_len:
            res[0].append(x[i])
            res[1].append(y[i])
    return res


class RolloutWorker:
    def __init__(self, args, oracle, tokenizer):
        self.oracle = oracle
        self.max_len = args.max_len
        self.max_len = args.gen_max_len - 2
        self.episodes_per_step = args.gen_episodes_per_step
        self.random_action_prob = args.gen_random_action_prob
        self.reward_exp = args.gen_reward_exp
        self.sampling_temperature = args.gen_sampling_temperature
        self.eos_tok = -1
        self.out_coef = args.gen_output_coef

        self.eos_char = tokenizer.eos_token
        self.pad_tok = 22
        self.balanced_loss = args.gen_balanced_loss == 1
        self.reward_norm = args.gen_reward_norm
        self.reward_min = torch.tensor(float(args.gen_reward_min))
        self.loss_eps = torch.tensor(float(args.gen_loss_eps)).to(args.device)
        self.leaf_coef = args.gen_leaf_coef
        self.exp_ramping_factor = args.gen_reward_exp_ramping
        
        self.tokenizer = tokenizer
        if self.exp_ramping_factor > 0:
            self.l2r = lambda x, t=0: (x) ** (1 + (self.reward_exp - 1) * (1 - 1/(1 + t / self.exp_ramping_factor)))
        else:
            self.l2r = lambda x, t=0: (x) ** self.reward_exp
        self.device = args.device
        self.args = args
        self.workers = MbStack(oracle)

    def rollout(self, model, episodes, use_rand_policy=True):
        visited = []
        lists = lambda n: [list() for i in range(n)]
        states = [''] * episodes
        traj_states = [[''] for i in range(episodes)]
        traj_actions = lists(episodes)
        traj_rewards = lists(episodes)
        traj_dones = lists(episodes)

        for t in (range(self.max_len) if episodes > 0 else []):
            active_indices = np.int32([i for i in range(episodes)
                                       if not states[i].endswith(self.eos_char)])
            x = self.tokenizer.process([states[i] for i in active_indices]).to(self.device)
            lens = torch.tensor([len(i) for i in states
                                if not i.endswith(self.eos_char)]).long().to(self.device)
            with torch.no_grad():
                
                logits = model(x, lens, coef=self.out_coef, pad=self.pad_tok)
            if t == 0:
                logits[:, 0] = -1000 # Prevent model from stopping
                                     # without having output anything
            try:
                cat = Categorical(logits=logits / self.sampling_temperature)
            except:
                print(states)
                print(x)
                print(logits)
                print(list(model.model.parameters()))
                import pdb; pdb.set_trace()
            actions = cat.sample()
            if use_rand_policy and self.random_action_prob > 0:
                for i in range(actions.shape[0]):
                    if np.random.uniform(0,1) < self.random_action_prob:
                        actions[i] = torch.tensor(np.random.randint(t == 0, logits.shape[1])).to(self.device)
            chars = [self.tokenizer.vocab.itos[i.item()] for i in actions]
            
            # Append predicted characters for active trajectories
            for i, c, a in zip(active_indices, chars, actions):
                if c == self.eos_char or t == self.max_len - 1:
                    self.workers.push(states[i] + (c if c != self.eos_char else ''), i)
                    r = 0
                    d = 1
                else:
                    r = 0
                    d = 0
                traj_states[i].append(states[i] + c)
                traj_actions[i].append(a)
                traj_rewards[i].append(r)
                traj_dones[i].append(d)
                states[i] += c
            if all(i.endswith(self.eos_char) for i in states):
                break
        return visited, states, traj_states, traj_actions, traj_rewards, traj_dones

    def execute_train_episode_batch(self, model, it=0, dataset=None, use_rand_policy=True):
        # run an episode
        lists = lambda n: [list() for i in range(n)]
        visited, states, traj_states, \
            traj_actions, traj_rewards, traj_dones = self.rollout(model, self.episodes_per_step, use_rand_policy=use_rand_policy) 
        lens = np.mean([len(i) for i in traj_rewards])
        bulk_trajs = []
        rq = []
        for (r, mbidx) in self.workers.pop_all():
            traj_rewards[mbidx][-1] = self.l2r(r, it)
            rq.append(r.item())
            s = states[mbidx]
            s = s + (self.eos_char if not s.endswith(self.eos_char) else '')
            visited.append((s, traj_rewards[mbidx][-1].item(), r.item()))
            bulk_trajs.append((s, traj_rewards[mbidx][-1].item()))
        if args.gen_data_sample_per_step > 0 and dataset is not None:
            n = args.gen_data_sample_per_step
            m = len(traj_states)
            if self.args.proxy_type == "classification":
                x, y = dataset.sample(n, 0.5)
            elif self.args.proxy_type == "regression":
                x, y = dataset.sample(n)
            x, y = filter_len(x, y, self.max_len)
            n = len(x)
            traj_states += lists(n)
            traj_actions += lists(n)
            traj_rewards += lists(n)
            traj_dones += lists(n)
            bulk_trajs += list(zip([i+self.eos_char for i in x],
                                   [self.l2r(torch.tensor(i), it) for i in y]))
            for i in range(len(x)):
                traj_states[i+m].append('')
                for c, a in zip(x[i] + self.eos_char, self.tokenizer.process([x[i] + self.eos_char])[0]-2):
                    traj_states[i+m].append(traj_states[i+m][-1] + c)
                    traj_actions[i+m].append(a)
                    traj_rewards[i+m].append(0 if c != self.eos_char else self.l2r(y[i], it))
                    traj_dones[i+m].append(float(c == self.eos_char))
        return {
            "visited": visited,
            "trajectories": {
                "traj_states": traj_states,
                "traj_actions": traj_actions,
                "traj_rewards": traj_rewards,
                "traj_dones": traj_dones,
                "states": states,
                "bulk_trajs": bulk_trajs
            }
        }


def train_generator(args, generator, oracle, tokenizer, dataset):
    print("Training generator")
    visited = []
    rollout_worker = RolloutWorker(args, oracle, tokenizer)
    for it in tqdm(range(args.gen_num_iterations + 1)):
        rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it, dataset)
        visited.extend(rollout_artifacts["visited"])

        loss, loss_info = generator.train_step(rollout_artifacts["trajectories"])        
        args.logger.add_scalar("generator_total_loss", loss.item())
        wandb_log = {"generator_total_loss": loss.item()}
        for key, val in loss_info.items():
            args.logger.add_scalar(f"generator_{key}", val.item())
            wandb_log[f"generator_{key}"] = val.item()
        if it % 100 == 0:
            rs = torch.tensor([i[-1] for i in rollout_artifacts["trajectories"]["traj_rewards"]]).mean()
            args.logger.add_scalar("gen_reward", rs.item())
            wandb_log["gen_reward"] = rs.item()
        if it % 5000 == 0:
            args.logger.save(args.save_path, args)
        if args.use_wandb:
            wandb.log(wandb_log)
    return rollout_worker, None


def filter_samples(args, samples, reference_set):
    filtered_samples = []
    for sample in samples:
        similar = False
        for example in reference_set:
            if is_similar(sample, example, args.filter_distance_type, args.filter_threshold):
                similar = True
                break
        if not similar:
            filtered_samples.append(sample)
    return filtered_samples


def sample_batch(args, rollout_worker, generator, current_dataset, oracle):
    print("Generating samples")
    samples = ([], [])
    scores = []
    while len(samples[0]) < args.num_sampled_per_round * 5:
        rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it=0, use_rand_policy = False)  # w/o sample from dataset
        states = rollout_artifacts["trajectories"]["states"]
        if args.filter:
            if args.proxy_type == "classification":
                states = filter_samples(args, states, current_dataset.pos_train)
                states = filter_samples(args, states, current_dataset.pos_valid)
            else:
                states = filter_samples(args, states, current_dataset.train)
                states = filter_samples(args, states, current_dataset.valid)
            states = filter_samples(args, states, samples[0])
        samples[0].extend(states)
        scores.extend([rews[-1].cpu().item() for rews in rollout_artifacts["trajectories"]["traj_rewards"]])
    idx_pick = np.argsort(scores)[::-1][:args.num_sampled_per_round]
    picked_states = np.array(samples[0])[idx_pick].tolist()
    return (picked_states, np.array(oracle(picked_states)).tolist())


def construct_proxy(args, tokenizer, dataset=None):
    proxy = get_proxy_model(args, tokenizer)
    sigmoid = nn.Sigmoid()
    if args.proxy_type == "classification":
        l2r = lambda x: sigmoid(x.clamp(min=args.gen_reward_min)) / args.gen_reward_norm
    elif args.proxy_type == "regression":
        l2r = lambda x: x.clamp(min=args.gen_reward_min) / args.gen_reward_norm
    args.reward_exp_min = max(l2r(torch.tensor(args.gen_reward_min)), 1e-32)
    acq_fn = get_acq_fn(args)
    return acq_fn(args, proxy, l2r, dataset)


def mean_pairwise_distances(args, seqs):
    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(edit_dist(*pair))
    return np.mean(dists)


def mean_novelty(seqs, ref_seqs):
    novelty = [min([edit_dist(seq, ref) for ref in ref_seqs]) for seq in seqs]
    return np.mean(novelty)


def log_overall_metrics(args, dataset, round, collected=False):
    top100 = dataset.top_k(100)
    top1000 = dataset.top_k(1000)
    args.logger.add_scalar("top-100-scores", np.mean(top100[1]), use_context=False)
    args.logger.add_scalar("top-1000-scores", np.mean(top1000[1]), use_context=False)
    dist100 = mean_pairwise_distances(args, top100[0])
    dist1000 = mean_pairwise_distances(args, top1000[0])
    ref_seqs, _ = dataset.get_ref_data()
    novelty1000 = mean_novelty(top1000[0], ref_seqs)
    args.logger.add_scalar("top-100-dists", dist100, use_context=False)
    args.logger.add_scalar("top-1000-dists", dist1000, use_context=False)
    args.logger.add_object("top-100-seqs", top100[0])
    args.logger.add_object("top-1000-seqs", top1000[0])
    print("Scores, 100, 1000", np.mean(top100[1]), np.mean(top1000[1]))
    print("Dist, 100, 1000", dist100, dist1000)
    print("Novelty, 1000", novelty1000)
    log = {'top-100': np.mean(top100[1]),
           'top-1000': np.mean(top1000[1]),
           'dist-100': dist100,
           'dist-1000': dist1000,
           'novelty-1000': novelty1000,
           'round': round}
    if collected:
        top100 = dataset.top_k_collected(100)
        top1000 = dataset.top_k_collected(1000)
        args.logger.add_scalar("top-100-collected-scores", np.mean(top100[1]), use_context=False)
        args.logger.add_scalar("top-1000-collected-scores", np.mean(top1000[1]), use_context=False)
        dist100 = mean_pairwise_distances(args, top100[0])
        dist1000 = mean_pairwise_distances(args, top1000[0])
        novelty1000 = mean_novelty(top1000[0], ref_seqs)
        args.logger.add_scalar("top-100-collected-dists", dist100, use_context=False)
        args.logger.add_scalar("top-1000-collected-dists", dist1000, use_context=False)
        args.logger.add_object("top-100-collected-seqs", top100[0])
        args.logger.add_object("top-1000-collected-seqs", top1000[0])
        print("Collected Scores, 100, 1000", np.mean(top100[1]), np.mean(top1000[1]))
        print("Collected Dist, 100, 1000", dist100, dist1000)
        print("Collected Novelty, 1000", novelty1000)
        log["collected_max"] = np.max(top100[1])
        log["collected_top-100"] = np.mean(top100[1])
        log["collected_top-1000"] = np.mean(top1000[1])
        log["collected_dist-100"] = dist100
        log["collected_dist-1000"] = dist1000
        log["collected_novelty-1000"] = novelty1000
    
    if args.use_wandb:
        wandb.log(log)


def train(args, oracle, dataset):
    tokenizer = get_tokenizer(args)
    args.logger.set_context("iter_0")
    proxy = construct_proxy(args, tokenizer, dataset=dataset)
    log_overall_metrics(args, dataset, 0)
    proxy.update(dataset)
    for round in range(args.num_rounds):
        args.logger.set_context(f"iter_{round+1}")
        generator = get_generator(args, tokenizer)
        rollout_worker, losses = train_generator(args, generator, proxy, tokenizer, dataset)
        batch = sample_batch(args, rollout_worker, generator, dataset, oracle)
        args.logger.add_object("collected_seqs", batch[0])
        args.logger.add_object("collected_seqs_scores", batch[1])
        dataset.add(batch)
        log_overall_metrics(args, dataset, round+1, collected=True)
        proxy.update(dataset)
        args.logger.save(args.save_path, args)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.logger = get_logger(args)
    args.device = torch.device('cuda')
    oracle = get_oracle(args)
    dataset = get_dataset(args, oracle)
    
    if args.use_wandb:
        run = wandb.init(project='delta-cs', group=args.task, config=args, reinit=True)
        wandb.run.name = args.name + "_" + wandb.run.id

    train(args, oracle, dataset)
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)