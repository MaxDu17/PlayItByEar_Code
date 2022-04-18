import platform
print(platform.node())
import copy
import math
import os
import pickle as pkl
import sys

import time

from shutil import copyfile

import numpy as np

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import core.utils as utils
# from core.logger import Logger
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
    env = utils.FrameStack_StackCat(env, cfg, k = cfg.frame_stack,  l_k = cfg.lowdim_stack, stack_depth = cfg.stack, demo = False, audio = False)
    return env


class Workspace(object):
    def __init__(self, cfg):
#         input(cfg.runName)
        self.work_dir = cfg.log_dir
        print(f'workspace: {self.work_dir}')
        print("cuda status: ", torch.cuda.is_available())
        self.cfg = cfg
#         copyfile("../../generate_resid_demos_episodes.yaml", "hyperparameters.yaml") #makes sure we keep track of hyperparameters 
        
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)
        self.env.seed(cfg.seed)
        
        lowdim, obs = self.env.reset() #let's mold our model to what the environment outputs
        cfg.agent.params.obs_shape = np.shape(obs)[1:] 
        
        #setting the appropiate configuration parameters 
        cfg.agent.params.action_shape = self.env.action_space.shape
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        print("creating and instantiating base agent")
        self.base_agent = hydra.utils.instantiate(cfg.agent) 
        self.base_agent.load(self.cfg.actor_name, prefix = cfg.actor_root, load_critic = False)
        
        print("loading the buffer!")
        self.expert_buffer = pkl.load(open(cfg.demo_root + cfg.demo_file, "rb")) #the expert replay demo 
        
        print("making another buffer!")
        self.replay_buffer = ReplayBuffer(self.expert_buffer.lowdim_shape, self.expert_buffer.obs_shape, 
                                          self.expert_buffer.action_shape, self.expert_buffer.numEpisodes, 
                                          self.expert_buffer.episodeLength, self.expert_buffer.image_pad,
                                          self.expert_buffer.device)
        

    #this self-contained function trains the model and evaluates it 
    def run(self):   
        for index, episode in enumerate(self.expert_buffer.allEpisodes):
            print("step ", index, ". Prop done: ", index / (self.expert_buffer.numEpisodes))
            buffer_list = list()
            for window in range(episode.episode_length):
                lowdim, obses, actions, shaped_rewards, sparse_rewards, next_lowdim, next_obses, not_dones, not_dones_no_max = episode.indexed_rollout(10, window)
                action = actions[-1]
                shaped_rewards = shaped_rewards[-1]
                sparse_rewards = sparse_rewards[-1]
                done = not(not_dones[-1])
                done_no_max = not(not_dones_no_max[-1])
                resid_input = lowdim[-1]
                next_resid_input = next_lowdim[-1]
                
                base_action = self.base_agent.act(lowdim, obses, sample=False)                
                resid_action = action - base_action #this is the modification               
                buffer_list.append((resid_input, obses[-1], resid_action, shaped_rewards, sparse_rewards, next_resid_input, next_obses[-1], done, done_no_max))
            self.replay_buffer.add(buffer_list)
        pkl.dump(self.replay_buffer, open(f"{self.work_dir}/demos.pkl", "wb" ), protocol=4 )

@hydra.main(config_path='generate_resid_demos_episodes.yaml', strict=True)
def main(cfg):
    from generate_resid_demos_episodes import Workspace as W
    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
