# this file is imitation training only
import librosa
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

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import core.utils as utils
from core.logger import Logger
from core.replay_buffer_audio_episode import ReplayBufferAudioEpisodes as ReplayBufferEpisode
from core.video import VideoRecorder
import gym
import csv


torch.backends.cudnn.benchmark = True

import franka_env

easyResetPos = [0.5, 0, 0.125, 0.0, 0.0, 1.0, 0.0]
defaultResetPos = [0.5, -0.1, 0.2, 0.0, 0.0, 1.0, 0.0]

def make_env(cfg):
    env = gym.make("Franka-v0", resetpos = np.array(easyResetPos), max_path_length = cfg.episodeLength)
    env.set_nuc(3) # Change as requierd
    cfg.agent.params.lowdim_dim = cfg.lowdim_stack *  7
    env = utils.FrameStack_StackCat(env, cfg, k=cfg.frame_stack,  l_k = cfg.lowdim_stack, stack_depth = cfg.stack, demo = False, audio = True)
    np.random.seed(cfg.seed)

    return env


def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


class Workspace(object):
    def __init__(self, cfg):
        torch.multiprocessing.set_start_method('spawn') #is this needed?
        torch.multiprocessing.set_sharing_strategy('file_system')

        self.work_dir = cfg.log_dir
        print(f'workspace: {self.work_dir}')
        print("cuda status: ", torch.cuda.is_available())
        self.cfg = cfg
        self.work_dir += f"/finalsystem_zeroshot_cloth/" if cfg.eval_only else ""

        self.logger = Logger(cfg = cfg,
                             log_dir = self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.env = None #don't initialize environment for imitation learning

        cfg.agent.params.obs_shape = [cfg.frame_stack * 3, cfg.image_size, cfg.image_size]
        cfg.agent.params.lowdim_dim = cfg.raw_lowdim * cfg.lowdim_stack

        cfg.audio_shape = [57, 160]
        cfg.audio_steps = cfg.audio_shape[0] #get the audio dimensions for later
        cfg.audio_bins = cfg.audio_shape [1]

        #setting the appropiate configuration parameters
        cfg.agent.params.action_shape = [4]

        cfg.agent.params.action_range = [-0.05, 1.0]

        self.agent = hydra.utils.instantiate(cfg.agent)

        if not cfg.eval_only:
            print("loading the buffer!")
            self.expert_buffer_obj = pkl.load(open(cfg.demo_file, "rb")) #the expert replay demo
            self.expert_buffer_obj.set_sample_settings(trainprop = 1, train = True, length = cfg.stack)
            self.expert_buffer = iter(torch.utils.data.DataLoader(self.expert_buffer_obj,
                                                 batch_size=self.cfg.batch_size,
                                                 pin_memory=True,
                                                 num_workers=cfg.expert_workers,
                                                 worker_init_fn=worker_init_fn))
            print("Done loading the buffer!")


        self.step = 0

    def final_evaluate(self, episodes):
        print("making the environment!")
        self.env = make_env(self.cfg)
        prop_success = 0

        for episode in range(episodes):
            input("reset the environment. Press enter to continue")
            spec_list = list()
            img_list = list()

            lowdim, obs, audio = self.env.reset()
            self.video_recorder.new_recorder_init(f'{self.step}_{episode}.gif', enabled=(episode == 0 or episodes is not None))
            done = False
            episode_step = 0
            while not done:
                beg_time = time.time()
                with utils.eval_mode(self.agent):
                    action = self.agent.act(lowdim, obs, audio, sample=False, squash = self.cfg.use_squashed)

                lowdim, obs, audio, reward, done, info = self.env.step(action)
                sys.stdout.write("..")
                sys.stdout.flush()
                spec_list.append(audio)
                img_list.append(self.env.high_dim_camera())
                episode_step += 1
                print(time.time() - beg_time)
            for i in range(10):
                _, _, _, _, _, _ = self.env.step([0,0,0,0]) #quirk of the system
            lowdim, obs, audio = self.env.reset()
            #rendering images at the end to save time
            for rendered_image, audio in zip(img_list, spec_list): #picture in picture of the spectrogram
                img_arr = librosa.power_to_db(audio[-1])
                img_arr = 255 * (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
                img_arr = img_arr.astype(np.uint8)
                rendered_image[-57:, 0:160, :] = np.expand_dims(img_arr, axis = 2)
                self.video_recorder.simple_record(rendered_image, flip = False)

            invalid = True
            while invalid: #asks for human input because a reset is needed
                isSucc = input("was this run successful? (y/n)")
                if isSucc == 'y':
                    invalid = False
                    prop_success += 1
                elif isSucc == 'n':
                    invalid = False

            print("Evaluate episode {0} done".format(episode))
            self.video_recorder.clean_up()
        prop_success /= episodes
        self.logger.log('eval/prop_success', prop_success, self.step)
        self.logger.dump(self.step)


    #this function will load pretrained models and evaluate them
    def restoreAndRun(self):
        self.agent.load(self.cfg.load_dir, prefix = self.cfg.results_root)
        self.final_evaluate(20)

    #this self-contained function trains the model and evaluates it
    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        while self.step < self.cfg.num_train_steps:
            if self.step % 10 == 0:
                print(self.step)
                self.logger.dump(self.step, save=True)
            self.agent.update_bc(self.expert_buffer, self.logger, self.step, squash = self.cfg.use_squashed)
            self.step += 1
            if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                print("save!")
                self.agent.save(self.step, self.work_dir)


@hydra.main(config_path='audio_imitationtrain_memory.yaml', strict=True)
def main(cfg):
    from audio_imitationtrain_memory import Workspace as W
    workspace = W(cfg)
    if cfg.eval_only:
        workspace.restoreAndRun()
    else:
        workspace.run()



if __name__ == '__main__':
    main()
