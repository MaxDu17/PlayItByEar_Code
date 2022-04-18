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

from matplotlib import pyplot as plt
import io
torch.backends.cudnn.benchmark = True
import franka_env

easyResetPos = [0.5, 0, 0.125, 0.0, 0.0, 1.0, 0.0]
defaultResetPos = [0.5, -0.1, 0.2, 0.0, 0.0, 1.0, 0.0]

def make_env(cfg):
    env = gym.make("Franka-v0", resetpos = np.array(easyResetPos), max_path_length = cfg.episodeLength)
    env.set_nuc(3) # Change as requierd
    #env.reset()
    cfg.agent.params.lowdim_dim = cfg.lowdim_stack *  7
    env = utils.FrameStack_StackCat(env, cfg, k= cfg.frame_stack,  l_k = cfg.lowdim_stack, stack_depth = cfg.stack, demo = False, audio = True)
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
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        print("cuda status: ", torch.cuda.is_available())
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        self.env.seed(cfg.seed)
        lowdim, obs, audio = self.env.reset() #let's mold our model to what the environment outputs
        cfg.agent.params.obs_shape = np.shape(obs)[1:]

        cfg.audio_steps = audio.shape[1] #get the audio diemnsions for later
        cfg.audio_bins = audio.shape[2]
        cfg.audio_shape = audio.shape[1:] #NO THIS IS WRONG THIS SHOULD BE NON-BATCHED

        cfg.agent.params.action_shape = self.env.action_space.shape

        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.agent = hydra.utils.instantiate(cfg.agent) #weird syntax; it makes an agent (see drq.py)

        print("weights are being loaded")
        self.agent.load(self.cfg.load_dir, prefix = self.cfg.results_root)
        print("done!")
        print("loading!!")

        self.old_replay_buffer_obj = pkl.load(open(cfg.demo_file, "rb")) #load previous collection attempt (allows you to collect in multiple sessions)

        self.old_replay_buffer_obj.set_sample_settings(trainprop = 1, train = True, length = 10, correctionsOnly = False)
        self.old_replay_buffer = iter(torch.utils.data.DataLoader(self.old_replay_buffer_obj,
                                                 batch_size=self.cfg.batch_size,
                                                 num_workers=self.cfg.num_old_workers,
                                                 worker_init_fn=worker_init_fn))

        print("your replay buffer is at ", self.old_replay_buffer_obj.idx)
        print("is full? ", self.old_replay_buffer_obj.full)
        input("press enter to continue")

        assert cfg.replay_buffer_capacity % cfg.episodeLength == 0

        if cfg.first_time: #robust to crashes in an experiment
            self.new_replay_buffer_obj = ReplayBufferEpisode([1, cfg.raw_lowdim * cfg.lowdim_stack],
                                  self.env.observation_space.shape[1:],
                                  cfg.audio_shape,
                                  self.env.action_space.shape,
                                  cfg.num_corrections,
                                  cfg.episodeLength,
                                  self.cfg.image_pad, self.device)

            self.new_replay_buffer_obj.set_sample_settings(trainprop = 1, train = True, length = 10, correctionsOnly = True)
            self.new_replay_buffer = torch.utils.data.DataLoader(self.new_replay_buffer_obj,
                                                 batch_size=self.cfg.batch_size,
                                                 num_workers=self.cfg.num_new_workers,
                                                 worker_init_fn=worker_init_fn)
            #dataloader object can be made into a generator, but we update the original _obj object, which
            #implicily updates the dataloader. The generator is only updated when we explicitly regenerate
        else:
            print("loading!!")
            self.new_replay_buffer_obj = pkl.load(open(cfg.results_root + cfg.demo_file_working, "rb")) #load previous collection attempt (allows you to collect in multiple sessions)
            self.new_replay_buffer_obj.set_sample_settings(trainprop = 1, train = True, length = 10, correctionsOnly = True)
            self.new_replay_buffer_dataloader = torch.utils.data.DataLoader(self.new_replay_buffer_obj,
                                                 batch_size=self.cfg.batch_size,
                                                 num_workers=self.cfg.num_new_workers,
                                                 worker_init_fn=worker_init_fn)
            self.new_replay_buffer = iter(self.new_replay_buffer_dataloader)

            print("your replay buffer is at ", self.new_replay_buffer_obj.idx)
            print("is full? ", self.new_replay_buffer_obj.full)
            input("press enter to continue")

        self.logger = Logger(cfg = cfg,
             log_dir = self.work_dir,
             save_tb=cfg.log_save_tb,
             log_frequency=cfg.log_frequency_step,
             agent=cfg.agent.name,
             action_repeat=cfg.action_repeat)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None, fps=5)

        self.step = 0

        #transforms the delta into a human friendly form
    def transform_basis(self, delta):
        delta[0], delta[1], delta[2] = -delta[2], -delta[0], delta[1]
        return delta

    def momentum(self, delta, previous_delta, gamma):
        previous_delta = np.asarray(previous_delta)
        return (1 - gamma) * delta + gamma * previous_delta

    def single_demo(self, iteration, cfg):
        intervention_list = list()
        success, gripperStatus, episode_reward, episode_step, done = 0, 0, 0, 1, True
        CAP = 0.15
        done = False
        gamma = 0.05
        buffer_list, spec_list, img_list = list(), list(), list()
        liftStatus = False
        prev_delta = np.zeros(3)

        prev_position, trigger, gripperStatus, quitStatus = self.env.get_oculus_info()
        self.video_recorder.new_recorder_init(f'{iteration}.gif', enabled= True)
        lowdim, obs, audio = self.env.reset()

        intervention_steps = 0
        input("reset the environment. Press enter to continue")
        while not done:
            beg_time = time.time()
            position, trigger, gripperStatus, quitStatus = self.env.get_oculus_info()
            if quitStatus:
                print("early termination!")
                break

            delta = position - prev_position
            mod_delta = self.transform_basis(delta)
            mod_delta = self.momentum(mod_delta, prev_delta, gamma)

            mod_delta[2] *= 2 #vertical motion is hardest to do on oculus, so we enhance

            mod_delta = [component if -CAP <= component <= CAP else -CAP if component < -CAP else CAP for component in mod_delta] #prevent sudden movements
            action = np.append(mod_delta, gripperStatus) #concatenate with gripper command

            # we can intervene with pure gripper if needed
            if trigger == 0: #allow the robot to take over
                action = self.agent.act(lowdim, obs, audio, sample=False, squash = True)
                action[3] = action[3] if gripperStatus != 1 else gripperStatus
            #if the trigger is not depressed or if the gripper controller is depressed, record intervention
            if trigger == 1 or gripperStatus == 1:
                intervention_steps += 1
                intervention_list.append(self.step)

            print("\t", self.step) #the step variable increments until the end of episode
            next_lowdim, next_obs, next_audio, reward, done, info = self.env.step(action)

            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            buffer_list.append((lowdim[-1], obs[-1], audio[-1], action, 0,
                                (1.0 if self.step > cfg.sparseProp * cfg.episodeLength else 0.0), next_lowdim[-1], next_obs[-1], next_audio[-1], done, done_no_max))

            img_list.append(self.env.high_dim_camera())
            spec_list.append(next_audio)
            obs, lowdim, audio = next_obs, next_lowdim, next_audio #update s = s'
            prev_delta, prev_position = mod_delta, position
            episode_step += 1
            self.step += 1

        print("saving images")
        for rendered_image, audio in zip(img_list, spec_list):
            #picture-in-picture construction
            img_arr = librosa.power_to_db(audio)
            img_arr = img_arr[-1]
            img_arr = 255 * (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
            img_arr = img_arr.astype(np.uint8)
            rendered_image[-57:, 0:160, :] = np.expand_dims(img_arr, axis = 2)
            self.video_recorder.simple_record(rendered_image, flip = False)
        print("saving done")

        self.video_recorder.clean_up()
        lowdim, obs, audio = self.env.reset()

        print(f"Intervention prop: {intervention_steps / episode_step}")
        userInput = input("Press enter to keep, enter (d) to discard, and enter (q) to save and quit")
        if userInput == 'q':
            return userInput
        elif userInput == '':
            success = 1
        else:
            success = 0 #if demo wasn't successful, don't add to the real replay buffer
            print("discarded!")

        if success == 1:
            self.new_replay_buffer_obj.add(buffer_list, priority = intervention_list)
            print("****** ADDED ****** and we are at ", self.new_replay_buffer_obj.idx)
        return success

    def evaluate(self, episodes, intervention_step):
        prop_success = 0
        for episode in range(episodes):
            input("reset the environment. Press enter to continue")
            spec_list = list()
            img_list = list()

            lowdim, obs, audio = self.env.reset()
            self.video_recorder.new_recorder_init(f'eval_{intervention_step}_{episode}.gif', enabled=(episode == 0 or episodes is not None))
            done = False
            episode_step = 0
            while not done:
                _, _, _, quitStatus = self.env.get_oculus_info()

                beg_time = time.time()
                with utils.eval_mode(self.agent):
                    action = self.agent.act(lowdim, obs, audio, sample=False, squash = self.cfg.use_squashed)
                lowdim, obs, audio, reward, done, info = self.env.step(action)
                print(action)
                sys.stdout.write("..")
                sys.stdout.flush()
                spec_list.append(audio)
                img_list.append(self.env.high_dim_camera())
                episode_step += 1
                print(time.time() - beg_time)
                if quitStatus:
                    done = True

            lowdim, obs, audio = self.env.reset()
            #rendering images at the end to save time
            for rendered_image, audio in zip(img_list, spec_list):
                img_arr = librosa.power_to_db(audio[-1])
                img_arr = 255 * (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
                img_arr = img_arr.astype(np.uint8)
                rendered_image[-57:, 0:160, :] = np.expand_dims(img_arr, axis = 2)
                self.video_recorder.simple_record(rendered_image, flip = False)

            invalid = True
            while invalid:
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

    def run(self, cfg):
        try:
            counter = 0
            assert not self.new_replay_buffer_obj.full, "your buffer is already full!"
            numCurrSuccesses = int(self.new_replay_buffer_obj.idx)
            while numCurrSuccesses < int(self.new_replay_buffer_obj.numEpisodes):
                if self.new_replay_buffer_obj.idx > 3 and self.new_replay_buffer_obj.idx % 1 == 0:
                    answer = input("would you like to train now? (y/n)")
                    if answer == "y":
                        print("UPDATING BUFFER ITERABLE")
                        updated_buf = iter(self.new_replay_buffer)
                        print("FINISHED UPDATING BUFFER ITERABLE")
                        for i in range(self.cfg.train_per_step):  #train model
                            print("\t", i)
                            self.agent.update_bc_balanced(updated_buf, self.old_replay_buffer, self.logger,
                                                          self.new_replay_buffer_obj.idx, squash = self.cfg.use_squashed)
                        input("done training. Press enter to begin next round!")

                if self.new_replay_buffer_obj.idx % 2 == 0:
                    print("SAVING SAVING")
                    self.agent.save(self.new_replay_buffer_obj.idx, work_dir = self.work_dir)

                isSuccessful = self.single_demo(numCurrSuccesses, cfg) #change to cloth if needed
                if isSuccessful == 'q':
                    break

                numCurrSuccesses += isSuccessful
                if numCurrSuccesses % self.cfg.eval_frequency == 0 and isSuccessful:
                    input("eval time!")
                    self.evaluate(self.cfg.eval_episodes, numCurrSuccesses)

                counter += 1
                self.step = 0

                print("\t", numCurrSuccesses, " successes")

                time.sleep(2)

            pkl.dump(self.new_replay_buffer_obj, open( "demos_finished.pkl", "wb" ), protocol=4 )
        finally:
            print("running end routine")
            pkl.dump(self.new_replay_buffer_obj, open( "demos_backup.pkl", "wb" ), protocol=4 )
            self.agent.save(self.new_replay_buffer_obj.idx, work_dir = self.work_dir)


@hydra.main(config_path='intervention_episodes.yaml', strict=True)
def main(cfg):
    from intervention_episodes import Workspace as W
    workspace = W(cfg)
    workspace.run(cfg)

if __name__ == '__main__':
    main()

