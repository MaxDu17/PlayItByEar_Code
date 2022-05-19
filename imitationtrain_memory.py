# Behavior cloning implementation

import platform
print(platform.node())
import copy
import csv
import math
import os
import pickle as pkl
import sys
import random
import time
from shutil import copyfile
import numpy as np
import hydra
import gym

import torch

import core.utils as utils
from core.logger import Logger
from core.replay_buffer_3 import ReplayBufferDoubleRewardEpisodes as ReplayBuffer
from core.video import VideoRecorder

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

    ob_dict = env.env.reset()
    dims = 0
    for key in ob_dict:
        if key in cfg.modalities:
            dims += np.shape(ob_dict[key])[0]

    cfg.agent.params.lowdim_dim = cfg.lowdim_stack *  dims
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
        self.work_dir += f"/balanced_batches/" if cfg.balanced_batches else ""

        try:
            os.mkdir(self.work_dir)
        except:
            print("directory already exists")
        os.chmod(self.work_dir, 0o777)

        print(f'workspace: {self.work_dir}')
        print("cuda status: ", torch.cuda.is_available())
        self.cfg = cfg

        self.logger = Logger(cfg = cfg,
                             log_dir = self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        lowdim, obs = self.env.reset() #let's mold our model to what the environment outputs
        cfg.agent.params.obs_shape = np.shape(obs)[1:]
        #obs is returned as [stack, 3, 84, 84]. We want [3, 84, 84]
        #batch comes first, so [batch, stack, 3, 84, 84]

        #setting the appropiate configuration parameters
        cfg.agent.params.action_shape = self.env.action_space.shape
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        if not(cfg.eval_only):
            print("loading the buffer!")
            if cfg.balanced_batches:
                self.correction_buffer_obj = pkl.load(open(cfg.actor_root + cfg.corrections_file, "rb"))
                self.correction_buffer_obj.set_sample_settings(trainprop = 1, train = True, length = self.cfg.stack, correctionsOnly = self.cfg.priority)
                self.correction_buffer = iter(torch.utils.data.DataLoader(self.correction_buffer_obj,
                                                 batch_size=self.cfg.batch_size,
                                                 num_workers=self.cfg.correction_workers,
                                                 pin_memory = True,
                                                 worker_init_fn=worker_init_fn))
            self.expert_buffer_obj = pkl.load(open(cfg.demo_root + cfg.demo_file, "rb")) #the expert replay demo
            self.expert_buffer_obj.set_sample_settings(trainprop = 1, train = True, length = self.cfg.stack)
            self.expert_buffer = iter(torch.utils.data.DataLoader(self.expert_buffer_obj,
                                                 batch_size=self.cfg.batch_size,
                                                 num_workers=self.cfg.expert_workers,
                                                 pin_memory = True,
                                                 worker_init_fn=worker_init_fn))


            print("Done loading the buffer")
        if cfg.balanced_batches: #used for balanced batches
            print("Loading Actor")
            self.agent.load(self.cfg.load_dir, prefix = cfg.actor_root)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)

        self.step = 0

    #this function is a self-contained evaluator of the agent in a new environment
    def evaluate(self, episodes = None):
        first_episode_reward = 0
        average_episode_reward = 0
        first_success = 0
        prop_success = 0
        numIterations = self.cfg.num_eval_episodes if episodes is None else episodes

        for episode in range(numIterations):
            lowdim, obs = self.env.reset()
            obs = obs / 255
            self.video_recorder.new_recorder_init(f'{self.step}_{episode}.gif', enabled=(episode % 10 == 0 or episodes is not None))
            done = False
            episode_reward = 0
            episode_step = 0
            this_success = False
            beg_ = time.time()
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(lowdim, obs, sample=False, squash = self.cfg.use_squashed)
                lowdim, obs, reward, done, info = self.env.step(action)
                obs = obs / 255.
                sys.stdout.write("..")
                sys.stdout.flush()
                if episode % 10 == 0 or episodes is not None: #record every 10 episodes
                    self.video_recorder.simple_record(self.env.render_highdim_list(200, 200, ["agentview", "sideview"]))
                    first_episode_reward += reward

                #this just monitors the successes
                prop_success = prop_success + 1 if not(this_success) and reward > 0.99 else prop_success
                this_success = True if reward > 0.99 else this_success #marks current success
                first_success = True if this_success and episode == 0 else first_success
                episode_reward += reward
                episode_step += 1

            print("Evaluate episode {0} done".format(episode))
            average_episode_reward += episode_reward
            self.video_recorder.clean_up()

        average_episode_reward /= numIterations
        prop_success /= numIterations
        self.logger.log('eval/average_episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/prop_success', prop_success, self.step)
        self.logger.dump(self.step)
        if episodes is not None:
            print(prop_success)
            self.logger.dump(self.step)

    #this self-contained function trains the model and evaluates it
    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        while self.step <= self.cfg.num_train_steps:
            if self.step % 10 == 0:
                print(self.step)
                self.logger.dump(self.step, save=True)
            #main training stuff is here
            if self.cfg.balanced_batches:
                self.agent.update_bc_balanced(self.expert_buffer, self.correction_buffer, self.logger, self.step)
            else:
                self.agent.update_bc(self.expert_buffer, self.logger, self.step, squash = self.cfg.use_squashed)

            if self.step >= 0 and self.step % self.cfg.eval_frequency == 0:
                print("eval time!")
                self.agent.save(self.step, self.work_dir)
                self.evaluate()
            self.step += 1

        #this function will load pretrained models and evaluate them
    def restoreAndRun(self):
        self.agent.load(self.cfg.load_dir, prefix = self.cfg.actor_root)
        self.evaluate(episodes = 250)

@hydra.main(config_path='imitationtrain_memory.yaml', strict=True)
def main(cfg):
    from imitationtrain_memory import Workspace as W
    workspace = W(cfg)
    if cfg.eval_only:
        workspace.restoreAndRun()
    else:
        workspace.run()


if __name__ == '__main__':
    main()
