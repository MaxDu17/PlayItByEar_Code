import librosa
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
from core.logger import Logger
from core.replay_buffer_audio_episode import ReplayBufferAudioEpisodes as ReplayBufferEpisode
from core.video import VideoRecorder
import gym
import csv
from matplotlib import pyplot as plt
import io
torch.backends.cudnn.benchmark = True
import franka_env

easyResetPos = [0.5, 0, 0.125, 0.0, 0.0, 1.0, 0.0]
defaultResetPos = [0.5, -0.1, 0.2, 0.0, 0.0, 1.0, 0.0]

def make_env(cfg):
    env = gym.make("Franka-v0", resetpos = np.array(easyResetPos), max_path_length = cfg.episodeLength)
    env.set_nuc(2) # Change as requierd
    #env.reset()
    cfg.agent.params.lowdim_dim = cfg.raw_lowdim * cfg.lowdim_stack
#     cfg.agent.params.lowdim_dim = cfg.lowdim_stack * env.get_lowdim_dims(cfg.modalities)
    env = utils.FrameStack_Lowdim(env, cfg, k=cfg.frame_stack,  l_k = cfg.lowdim_stack, frameMode = 'cat', demo = False, audio = True)
    np.random.seed(cfg.seed)
    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = cfg.log_dir
        print(f'workspace: {self.work_dir}')
        print("cuda status: ", torch.cuda.is_available())
        self.cfg = cfg
#         copyfile("../../oculusDemos.yaml", "hyperparameters.yaml") #makes sure we keep track of hyperparameters

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        self.env.seed(cfg.seed)
        lowdim, obs, audio = self.env.reset() #let's mold our model to what the environment outputs
        cfg.agent.params.obs_shape = np.shape(obs)
        cfg.audio_steps = audio.shape[0] #get the audio diemnsions for later
        cfg.audio_bins = audio.shape[1]
        cfg.audio_shape = audio.shape

        if self.cfg.load_demo:
            print("loading!!")
            self.replay_buffer = pkl.load(open(cfg.demo_root + cfg.demo_file, "rb")) #load previous collection attempt (allows you to collect in multiple sessions)
            print("your replay buffer is at ", self.replay_buffer.idx)
            input("press enter to continue")
        else:
#             assert cfg.replay_buffer_capacity % cfg.episodeLength == 0
            self.replay_buffer = ReplayBufferEpisode([1, cfg.raw_lowdim * cfg.lowdim_stack],
                                              self.env.observation_space.shape,
                                              cfg.audio_shape,
                                              self.env.action_space.shape,
                                              cfg.demos,
                                              cfg.episodeLength,
                                              self.cfg.image_pad, self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None, fps=5)
#         input("here")
        self.step = 0

    #transforms the delta into a human friendly form
    def transform_basis(self, delta):
#         mod_delta[1] = -delta[0]
#         mod_delta[2] = delta[1]
#         mod_delta[0] = -delta[2]
#         delta[0], delta[1], delta[2] = -delta[2], -delta[0], delta[1]
#         delta[0], delta[1], delta[2] = -delta[2], -delta[0], delta[1]
        delta[0], delta[1] = -1 * delta[0], -1 * delta[1]
        return delta

    def momentum(self, delta, previous_delta, gamma):
        previous_delta = np.asarray(previous_delta)
        return (1 - gamma) * delta + gamma * previous_delta


    def single_demo(self, iteration, cfg):
        success, episode_reward, episode_step, done = 0, 0, 0, False
        gripperStatus, liftStatus, CAP = 0, False, 0.15
        gamma = 0.05 #how much we should factor past actions in
        prev_delta = np.zeros(3)

        self.video_recorder.new_recorder_init(f'{iteration}.gif', enabled= True)
        lowdim, obs, audio = self.env.reset()
        prev_position, gripperStatus, quitStatus = self.env.get_ps_info()
        # prev_position, trigger, gripperStatus, quitStatus = self.env.get_oculus_info()
        done = False
        trigger = 0

        #keep track of things during running and then debrief after
        buffer_list = list()
        spec_list = list()
        img_list = list()

        while not done: #self.step < cfg.episodeLength:
            beg_time = time.time()
            position, gripperStatus, quitStatus = self.env.get_ps_info()
#             print(position)
            # position, trigger, gripperStatus, quitStatus = self.env.get_oculus_info()
#             if quitStatus:
#                 print("early termination!")
#                 break

#             delta = position - prev_position
            delta = position 
            mod_delta = self.transform_basis(delta)

            if trigger == 1: #freeze movements; allows for repositioning of the hand control
                prev_position = position
                prev_delta = mod_delta
                continue #if this happens, skip this step

#             mod_delta = self.momentum(mod_delta, prev_delta, gamma)

#             print("\t", self.step) #the step variable increments until the end of episode
#             print(mod_delta)
#             mod_delta[0:2] *= 2
     
#             mod_delta[0:2] *=0.75 #used to be 0.5
#             mod_delta[2] *= 2
            mod_delta = [0 if -0.04 < component <= 0.04 else component for component in mod_delta] #prevent sudden movements
            mod_delta = [component if -CAP <= component <= CAP else -CAP if component < -CAP else CAP for component in mod_delta] #prevent sudden movements
            print("---")
            print(mod_delta[0:2])
            print("-----")
            action = np.append(mod_delta, gripperStatus) #concatenate with gripper command
            next_lowdim, next_obs, next_audio, reward, done, info = self.env.step(action)
#
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            buffer_list.append((lowdim, obs, audio, action, 0,
                                (1.0 if self.step > cfg.sparseProp * cfg.episodeLength else 0.0), next_lowdim, next_obs, next_audio, done, done_no_max))

            img_list.append(self.env.high_dim_camera())
            spec_list.append(next_audio)
            obs, lowdim, audio = next_obs, next_lowdim, next_audio #s = s'
            prev_position, prev_delta = position, mod_delta
            episode_step += 1
            self.step += 1

#             print("time: ", time.time() - beg_time)

                #rendering images at the end to save time
        for rendered_image, audio in zip(img_list, spec_list):
            #picture-in-picture construction
            img_arr = librosa.power_to_db(audio)
            img_arr = 255 * (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
            img_arr = img_arr.astype(np.uint8)
            rendered_image[-57:, 0:160, :] = np.expand_dims(img_arr, axis = 2)
            self.video_recorder.simple_record(rendered_image, flip = False)

        self.video_recorder.clean_up()
        self.env.reset() #to move robot arm back to position



        userInput = input("Press enter to keep, enter (d) to discard, and enter (q) to save and quit")
        if userInput == 'q':
            return userInput
        elif userInput == '':
            success = 1
            self.replay_buffer.add(buffer_list)
            print("****** ADDED ****** and we are at ", self.replay_buffer.idx)
        else:
            success = 0 #if demo wasn't successful, don't add to the real replay buffer
            print("discarded!")

        return success


    def run(self, cfg):
        try:
            counter = 0
            successes = 0
            assert not self.replay_buffer.full, "your buffer is already full!"
            numMoreSuccesses = self.replay_buffer.numEpisodes - self.replay_buffer.idx
            numCurrSuccesses = self.replay_buffer.idx
            while successes < numMoreSuccesses:
                isSuccessful = self.single_demo(successes + numCurrSuccesses, cfg) #change to cloth if needed
                if isSuccessful == 'q':
                    break
                counter += 1
                self.step = 0
                successes += isSuccessful
                print("\t", successes, " successes. ", (numMoreSuccesses - successes), " to go.")
                time.sleep(2)
                if successes % 20 == 0:
                    print("dumping!")
                    pkl.dump(self.replay_buffer, open(cfg.log_dir + "demos" + str(successes) + ".pkl", "wb" ), protocol=4 )
            pkl.dump(self.replay_buffer, open(cfg.log_dir + "demos_finished.pkl", "wb" ), protocol=4 )
        finally:
            print("running end routine")
            pkl.dump(self.replay_buffer, open(cfg.log_dir + "demos_backup.pkl", "wb" ), protocol=4 )



@hydra.main(config_path='oculusDemos.yaml', strict=True)
def main(cfg):
    from oculusDemos import Workspace as W
    workspace = W(cfg)
    workspace.run(cfg)

if __name__ == '__main__':
    main()

