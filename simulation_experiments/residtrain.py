import platform
print(platform.node())
import copy
import math
import os
import pickle as pkl
import sys
import random

import time

from shutil import copyfile

import numpy as np
import gc

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import core.utils as utils
from core.logger import Logger
from core.replay_buffer_3 import ReplayBufferDoubleRewardEpisodes as ReplayBuffer
from core.video import VideoRecorder
import gym
import csv

from robosuite import make
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config

torch.backends.cudnn.benchmark = True


from custom_environments.indicatorboxBlock import IndicatorBoxBlock
from custom_environments.blocked_pick_place import BlockedPickPlace

def make_env(cfg):
    env = make(
        cfg.environmentName,
        robots=["Panda"], #the robot
        controller_configs=load_controller_config(default_controller="OSC_POSITION"), #this decides which controller settings to use!
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs = True,
        reward_shaping=True,
        control_freq=20,
        horizon = cfg.horizon,
        camera_names="agentview",
        camera_heights=cfg.image_size,
        camera_widths=cfg.image_size
        )

    allmodlist = list(cfg.modalities)
    allmodlist.append(cfg.cameraName)
    env = GymWrapper(env, keys = allmodlist)

    cfg.agent.params.lowdim_dim = cfg.lowdim_stack *  env.get_lowdim_dims(cfg.modalities)
    env = utils.FrameStack_StackCat(env, cfg, k=cfg.frame_stack,  l_k = cfg.lowdim_stack, stack_depth = cfg.stack, demo = False, audio = False)

    np.random.seed(cfg.seed)
    torch.multiprocessing.set_start_method('spawn') #is this needed?
    torch.multiprocessing.set_sharing_strategy('file_system')

    return env

def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = cfg.log_dir
        print(f'workspace: {self.work_dir}')
        print("cuda status: ", torch.cuda.is_available())
        self.cfg = cfg
#         copyfile("../../../../residtrain.yaml", "hyperparameters.yaml") #makes sure we keep track of hyperparameters
#         copyfile("../../../../residtrain.py", "residtrain.py")

        self.logger = Logger(cfg = cfg,
                             log_dir = self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.resid_agent.name,
                             action_repeat=cfg.action_repeat)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)
        self.env.seed(cfg.seed)

        lowdim, obs = self.env.reset() #let's mold our model to what the environment outputs

        cfg.agent.params.obs_shape = np.shape(obs)[1:]

        #setting the appropiate configuration parameters
        cfg.agent.params.action_shape = self.env.action_space.shape

        cfg.resid_agent.params.lowdim_dim = cfg.agent.params.lowdim_dim
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.base_agent = hydra.utils.instantiate(cfg.agent) #weird syntax; it makes an agent (see drq.py)
        self.base_agent.load(self.cfg.actor_name, load_critic = False, prefix = cfg.actor_root)

        self.resid_agent = hydra.utils.instantiate(cfg.resid_agent) #weird syntax; it makes an agent (see drq.py)
        #replay buffer is used to train the model during update steps

        self.resid_scale = 1
        self.expert_buffer = None
        assert "resid" in self.cfg.demo_file

        print("loading the buffer!")
        self.expert_buffer_obj = pkl.load(open(cfg.actor_root + cfg.demo_file, "rb")) #the expert replay demo
        self.expert_buffer_obj.set_sample_settings(trainprop = 1, train = True, length = 10)
        self.expert_buffer = iter(torch.utils.data.DataLoader(self.expert_buffer_obj,
                     batch_size=self.cfg.batch_size,
                     num_workers=cfg.expert_workers,
                     pin_memory=False,
                     worker_init_fn=worker_init_fn))

        modified_lowdim_space = self.env.lowdim_space.copy()
        modified_lowdim_space[1] += self.env.action_space.shape[0]

        print("making another buffer!")
        assert cfg.replay_buffer_capacity % cfg.episodeLength == 0, "capacity must hold an integer number of episodes"
        self.replay_buffer_obj = ReplayBuffer(self.expert_buffer_obj.lowdim_shape, self.expert_buffer_obj.obs_shape,
                                          self.expert_buffer_obj.action_shape, cfg.replay_buffer_capacity / cfg.episodeLength,
                                          cfg.episodeLength, cfg.image_pad,
                                          cfg.device)

        self.replay_buffer_obj.set_sample_settings(trainprop = 1, train = True, length = 10)
        self.replay_buffer = torch.utils.data.DataLoader(self.replay_buffer_obj,
                         batch_size=self.cfg.batch_size,
                         num_workers=cfg.memory_workers,
                         pin_memory=False,
                         worker_init_fn=worker_init_fn)
        self.replay_buffer_iterable = iter(self.replay_buffer)


        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0


    #this function is a self-contained evaluator of the agent in a new environment
    def evaluate(self, episodes = None):
        average_episode_reward = 0
        first_success = 0
        first_episode_reward = 0
        prop_success = 0
        eval_episodes = episodes if episodes is not None else self.cfg.num_eval_episodes

        for episode in range(eval_episodes):
            lowdim, obs = self.env.reset()
            self.video_recorder.new_recorder_init(f'{self.step}_{episode}.gif', enabled=(episode == 0 or episodes is not None))
            done = False
            episode_reward = 0
            episode_step = 0
            this_success = False
            beg_ = time.time()
            while not done:
                with utils.eval_mode(self.base_agent) and utils.eval_mode(self.resid_agent):
                    base_action = self.base_agent.act(lowdim, obs, sample=False)
                    base_action_gpu = torch.as_tensor(base_action, device = self.device)
                    resid_action = self.resid_agent.act(lowdim, obs, sample=False, base_action = base_action_gpu)
                lowdim, obs, reward, done, info = self.env.step(base_action + self.resid_scale * resid_action)
                sys.stdout.write("..")
                sys.stdout.flush()
                if episode == 0 or episodes is not None:
                    self.video_recorder.simple_record(self.env.render_highdim_list(200, 200, ["agentview", "sideview"]))
                    first_episode_reward += reward

                prop_success = prop_success + 1 if not(this_success) and reward > 0.99 else prop_success
                this_success = True if reward > 0.99 else this_success
                first_success = True if this_success and episode == 0 else first_success
                episode_reward += reward
                episode_step += 1

            print("Evaluate episode {0} done".format(episode))
            average_episode_reward += episode_reward
            self.video_recorder.clean_up()

        average_episode_reward /= eval_episodes
        prop_success /= eval_episodes
        self.logger.log('eval/average_episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/prop_success', prop_success, self.step)
        self.logger.dump(self.step)

    #this function will load pretrained models and evaluate them
    def restoreAndRun(self):
        self.resid_agent.load(self.cfg.load_dir)
        self.evaluate(250)

    #this self-contained function trains the model and evaluates it
    def run(self):
        #For BC-boosted runs
        start_time = time.time()

        episode, episode_reward, episode_step, done = 0, 0, 1, True
        buffer_list = list()
        while self.step < self.cfg.num_train_steps:
            beg = time.time()
            if done: #this runs at the end of every episode
                if self.step > 0:
                    self.replay_buffer_obj.add(buffer_list)
                    buffer_list.clear()
#                     k = next(iter(self.replay_buffer))
#                     input("test")
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps)) #UNCOMMENT THIS LATER

                    print("UPDATING BUFFER ITERABLE")
                    del self.replay_buffer_iterable
                    gc.collect()
                    self.replay_buffer_iterable = iter(self.replay_buffer)
                    print("FINISHED UPDATING BUFFER ITERABLE")

                ######### EVALUATION #########
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    print("eval time!")
                    self.resid_agent.save(self.step, self.work_dir)
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/episode', episode + 1, self.step)

                lowdim, obs = self.env.reset()
                with utils.eval_mode(self.base_agent) and torch.no_grad():
                    base_action = self.base_agent.act(lowdim, obs, sample=False)

                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

            #MAKE SURE NO GRADIENT
            ########## GETTING THE ACTION ##########
            if self.step < self.cfg.num_seed_steps: #if we have existing actor, we don't do random sampling
                action = base_action
                resid_action = np.zeros_like(action)
            else:
                with utils.eval_mode(self.resid_agent):
                    resid_action = self.resid_agent.act(lowdim, obs, sample=True, base_action = base_action)
                    action = base_action + self.resid_scale * resid_action

            ###############################33
            ######## THE TRAINING PROCESS ######
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.resid_agent.update_resid(self.replay_buffer_iterable, self.logger, self.step, self.cfg, self.base_agent, self.expert_buffer)
            ################################

            ###### RUNNING THE SIM ########
            next_lowdim, next_obs, reward, done, info = self.env.step(action)
            with utils.eval_mode(self.base_agent) and torch.no_grad():
                next_base_action = self.base_agent.act(next_lowdim, next_obs, sample=True)
            #############################

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward
            buffer_list.append((lowdim[-1], obs[-1], action, reward, 0, next_lowdim[-1], next_obs[-1], done, done_no_max))

            if(self.step % 5 == 0):
                print("Global step: ", self.step, "\tReward: ", reward, "base_action: ", base_action, "resid_action ", resid_action)
            obs = next_obs
            lowdim = next_lowdim
            base_action = next_base_action
            episode_step += 1
            self.step += 1
#             print(f"Time to step: {time.time() - beg}")


@hydra.main(config_path='residtrain.yaml', strict=True)
def main(cfg):
    from residtrain import Workspace as W
    workspace = W(cfg)
    if cfg.eval_only:
        workspace.restoreAndRun()
    else:
        workspace.run()


if __name__ == '__main__':
    main()
