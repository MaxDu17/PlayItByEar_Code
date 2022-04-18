# scripted intervention implementation

import librosa
import platform
print(platform.node())
import copy
import math
import os
import psutil
import pickle as pkl
import sys
import random

import time
import gc

from shutil import copyfile

import numpy as np

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import core.utils as utils
from core.logger import Logger
from core.replay_buffer_3 import ReplayBufferDoubleRewardEpisodes as ReplayBufferEpisode
from core.video import VideoRecorder
import gym
import csv

from robosuite import make
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config

from matplotlib import pyplot as plt
import io

torch.backends.cudnn.benchmark = True

from custom_environments.indicatorboxBlock import IndicatorBoxBlock
from custom_environments.blocked_pick_place import BlockedPickPlace


import franka_env

def make_env(cfg):
    env = make(
            cfg.environmentName,
            robots=["Panda"], #the robot
            controller_configs=load_controller_config(default_controller="OSC_POSITION"),
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
    env = utils.FrameStack_StackCat(env, cfg, k=cfg.frame_stack,  l_k = cfg.lowdim_stack, stack_depth = cfg.stack, demo = True, audio = False)
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
        self.work_dir = cfg.log_dir + "/corrections_improved"
        try:
            os.mkdir(self.work_dir)
        except:
            print("directory already exists")
        os.chmod(self.work_dir, 0o777)

        print(f'workspace: {self.work_dir}')
        print("cuda status: ", torch.cuda.is_available())
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        self.env.seed(cfg.seed)
        raw_dict, lowdim, obs= self.env.reset() #let's mold our model to what the environment outputs
        cfg.agent.params.obs_shape = np.shape(obs)[1:]

        cfg.agent.params.action_shape = self.env.action_space.shape

        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.agent = hydra.utils.instantiate(cfg.agent) #weird syntax; it makes an agent (see drq.py)


        print("weights are being loaded")
        self.agent.load(cfg.load_dir, prefix = cfg.actor_root)
        print("done!")
        print("loading!!")
        self.old_replay_buffer_obj = pkl.load(open(cfg.demo_root + cfg.demo_file, "rb")) #load previous collection attempt (allows you to collect in multiple sessions)
        self.old_replay_buffer_obj.set_sample_settings(trainprop = 1, train = True, length = cfg.stack, correctionsOnly = False)
        self.old_replay_buffer_dataloader = torch.utils.data.DataLoader(self.old_replay_buffer_obj,
                             batch_size=self.cfg.batch_size,
                             num_workers=cfg.num_workers,
                             worker_init_fn=worker_init_fn)
        self.old_replay_buffer = iter(self.old_replay_buffer_dataloader)


        self.new_replay_buffer = ReplayBufferEpisode(self.old_replay_buffer_obj.lowdim_shape, self.old_replay_buffer_obj.obs_shape,
                                  self.old_replay_buffer_obj.action_shape, cfg.num_corrections,
                                  cfg.episodeLength, cfg.image_pad,
                                  cfg.device)

        self.new_replay_buffer.set_sample_settings(trainprop = 1, train = True, length = cfg.stack, correctionsOnly = True)
        self.new_replay_buffer_dataloader = torch.utils.data.DataLoader(self.new_replay_buffer,
                                     batch_size=self.cfg.batch_size,
                                     num_workers=cfg.num_workers,
                                     worker_init_fn=worker_init_fn)
        self.new_replay_buffer_iterable = iter(self.new_replay_buffer_dataloader)


        self.logger = Logger(cfg = cfg,
             log_dir = self.work_dir,
             save_tb=cfg.log_save_tb,
             log_frequency=cfg.log_frequency_step,
             agent=cfg.agent.name,
             action_repeat=cfg.action_repeat)


        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None, fps=5)
        self.step = 0



    def single_demo_indicator_boxblock(self, iteration, cfg):
        #essentially, we have an oracle policy and we switch to the oracle if certain checkpoints are not met
        #control is given back to the agent after the oracle gets past the tricky task

        episode, episode_reward, episode_step, done = 0, 0, 1, True
        CAP = 0.15
        done = False
        episode_step = 0
        episode += 1

        success = 0
        buffer_list = list()
        gripperStatus = 0
        liftStatus = False
        buffer_list = list()

        self.video_recorder.new_recorder_init(f'{iteration}.gif', enabled= True)
        raw_dict, lowdim, obs = self.env.reset()

        #some housekeeping variables
        status = -1
        gripperMagnitude = 1
        gripperStatus = -gripperMagnitude
        status_dict = {0 : "sidereach", 1 : "sidestep", 2 : "asdfasdad", 3: "moveup", 4 : "positioning",
                      5 : "blockreach", 6 : "grabbing", 7 : "lifting", 8 : "HALT"}
        reward = 0 #init reward
        hasContacted = False
        intervention = False
        intervention_prop = 0
        intervention_list = list()
        timeSinceContact = 0
        prev_action = [0, 0, 0]

        while self.step < cfg.episodeLength: #avoid magic numbers; I'll make this code more elegant later
            if hasContacted:
                timeSinceContact += 1 #essentially once we touch once, we start a timer
            if intervention: #keeps track of how much of this episode was controlled by intervention
                intervention_prop += 1

            with utils.eval_mode(self.agent):
                action = self.agent.act(lowdim, obs / 255, sample=False, squash = True) #get the agent's action first
                delta = np.linalg.norm(action[0:3] - prev_action)
            #for scripted calculations
            cube_pos = raw_dict["cube_pos"]
            claw_pos = raw_dict["robot0_eef_pos"]
            gripper_pos = raw_dict["robot0_gripper_qpos"]

            # contact logic
            if np.linalg.norm(raw_dict["robot0_gripper_joint_force"]) > 0.35:
                print("\t\tcontact")
                hasContacted = True #marks first contact
                timeSinceContact = 0 #resets the contact counter
            # intervention for searching
            if not intervention and (hasContacted or self.step > 50) and claw_pos[2] > 0.92 and reward < 0.99:
                print("\tINTERVENTION SEARCH TIME")
                status = 0 # starts the grabbing sequence
                intervention = True
            #next, an intervention for grasping
            elif not intervention and np.linalg.norm(gripper_pos) > 0.05 and np.linalg.norm(claw_pos - cube_pos) < 0.05 and delta < 0.05 and self.step > 200:
                print("\tINTERVENTION GRAB TIME")
                status = 3 #starts the grabbing sequence
                intervention = True

            #if we are in intervention mode, this is how we generate the actions
            if intervention:
                 #what to do at each state
                if status == 0: #reach for table
                    destination = cube_pos.copy()
                    destination[1] -= 0.12 # some magic numbers that are tuned to this specific environment
                    destination[2] += 0.02
                elif status == 1: #move to the side
                    destination = cube_pos.copy()
                    destination[1] += 0.1
                elif status == 3: #move up
                    destination = claw_pos.copy()
                    destination[2] = cube_pos[2] + 0.026
                    gripperStatus = -gripperMagnitude
                elif status == 4: #move over cube
                    destination = cube_pos
                    destination[2] = cube_pos[2] + 0.026
                    gripperStatus = -gripperMagnitude
                elif status == 5: #reach
                    gripperStatus = -gripperMagnitude
                    destination = cube_pos
                elif status == 6: #grab
                    gripperStatus = gripperMagnitude
                elif status == 7: #lift
                    destination = cube_pos
                    destination[2] = cube_pos[2] + 0.1
                    gripperStatus = gripperMagnitude
                elif status == 8:
                    destination = claw_pos
                else:
                    raise Exception("this should not have happened")

                #displacement is the vector that we want to travel in
                displacement = destination - claw_pos

                #switchboard
                if claw_pos[2] > 0.95 and status == 7:
                    status = 8
                if np.linalg.norm(gripper_pos) < 0.045 and status == 6: #used to be 0.04
                    intervention = False #RELINQUISH CONTROL see if the thing can lift
                    status = 7
                    print("RELINQUISH CONTROL (to lift)")
                if np.linalg.norm(claw_pos - cube_pos) > 0.1 and status == 7: #close to lift
                    print("regrasping!")
                    status = 5
                if np.linalg.norm(displacement) < 0.02 and status == 5: #plunge to close #used to be 0.01
                    status = 6
                if np.linalg.norm(displacement) < 0.02 and status == 4: #approach to plunge
                    status = 5
                if np.linalg.norm(displacement) < 0.02 and status == 3: #raise to approach
                    status = 4
                if np.linalg.norm(raw_dict["robot0_gripper_joint_force"]) > 0.35 and (status == 1 or status == 0): #remove to raise
                    print("CONTACT")
                    status = 3
                    intervention = False #RELINQUISH CONTROL see if thing can grab
                    print("RELINQUISH CONTROL (to grab)")
                    hasContacted = True
                if np.linalg.norm(displacement) < 0.02 and status == 0: #reach next to the cube
                    status = 1

                displacement = np.multiply(displacement, 5)
                displacement = [component if -1 <= component <= 1 else -1 if component < -1 else 1 for component in displacement] #capping this vector
                print(status_dict[status])

                action = np.append(displacement, gripperStatus)
                assert max(map(abs, action)) <= 1 #making sure we are within our action space
                intervention_list.append(self.step)

            if self.step % 10 == 0:
                print(self.step)
                print(reward)
            if reward > 0.99:
                success = 1 #any success means we can accept this run

            raw_dict, next_lowdim, next_obs, reward, done, info = self.env.step(action)

            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            #adds to a list, which will be added to the replay buffer as an episode
            buffer_list.append((lowdim[-1], obs[-1], action, 0,
                                (1.0 if self.step > cfg.sparseProp * cfg.episodeLength else 0.0), next_lowdim[-1], next_obs[-1], done, done_no_max))

            video_frame = self.env.render_highdim_list(200, 200, ["sideview", "agentview"])

            if intervention:
                video_frame[0:30, 0:30] = 0
                video_frame[0:30, 0:30, 0] = 255
            else:
                video_frame[0:30, 0:30] = 0
                video_frame[0:30, 0:30, 1] = 255
            self.video_recorder.simple_record(video_frame) #"agentview",

            #advancing things
            obs = next_obs #update s = s'
            lowdim = next_lowdim
            episode_step += 1
            self.step += 1
            prev_action = action[0:3]

        self.logger.log('train_actor/intervention_prop', intervention_prop / cfg.episodeLength, self.new_replay_buffer.idx)
        print("intervention prop: ", intervention_prop / cfg.episodeLength)

        self.video_recorder.clean_up()
        if success and intervention_prop > 0:
            self.new_replay_buffer.add(buffer_list, priority = intervention_list)
            print("****** ADDED ****** and we are at ", self.new_replay_buffer.idx)
        return success and intervention_prop > 0 #discard runs where there is no corrections

    def single_demo_pick_place(self, iteration, cfg):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        episode_step = 0
        episode += 1

        success = 0
        buffer_list = list()

        self.video_recorder.new_recorder_init(f'{iteration}.gif', enabled= True)
        raw_dict, lowdim, obs = self.env.reset()

        #some housekeeping variables
        status = -1
        gripperMagnitude = 1
        gripperStatus = -gripperMagnitude
        status_dict = {0 : "sidereach", 1 : "sidestep", 2 : "asdfasdad", 3: "moveup", 4 : "positioning",
                       5 : "blockreach", 6 : "grabbing", 7 : "lifting", 8: "position", 9: "drop"}
        reward = 0 #init reward

        hasContacted = False
        hasMoved = False #these are related, but moved uses the "sound"
        intervention = False
        intervention_prop = 0
        intervention_list = list()
        timeSinceContact = 0
        bin_pos = raw_dict['bin_pos']
        prev_action = [0, 0, 0]

        while self.step < cfg.episodeLength: #avoid magic numbers; I'll make this code more elegant later
            with utils.eval_mode(self.agent):
                action = self.agent.act(lowdim, obs / 255, sample=False, squash = True) #get the agent's action first
                delta = np.linalg.norm(action[0:3] - prev_action)

            if hasContacted:
                timeSinceContact += 1 #essentially once we touch once, we start a timer

            if intervention: #keeps track of how much of this episode was controlled by intervention
                intervention_prop += 1

            #for scripted calculations
            cube_pos = raw_dict["cube_pos"]
            claw_pos = raw_dict["robot0_eef_pos"]
            gripper_pos = raw_dict["robot0_gripper_qpos"]

            if np.linalg.norm(raw_dict["robot0_gripper_joint_force"]) > 0.35:
                print("\t\tcontact")
                hasContacted = True #marks first contact
                timeSinceContact = 0 #resets the contact counter
            # searching intervention
            if not intervention and (hasContacted or self.step > 50) and claw_pos[2] > 0.92 and np.linalg.norm(claw_pos - cube_pos) > 0.05 and claw_pos[1] > -0.1:
                print("\tINTERVENTION SEARCH TIME")
                status = 0 # starts the grabbing sequence
                intervention = True
            #next, an intervention for grasping
            elif not intervention and np.linalg.norm(gripper_pos) > 0.05 and np.linalg.norm(claw_pos - cube_pos) < 0.05 and delta < 0.05 and self.step > 200:
                print("\tINTERVENTION GRAB TIME")
                status = 3 #starts the grabbing sequence
                intervention = True

            #if we are in intervention mode, this is how we generate the actions
            if intervention:
                 #what to do at each state
                if status == 0: #reach for table
                    destination = cube_pos.copy()
                    destination[1] -= 0.12
                    destination[2] += 0.02
                elif status == 1: #move to the side
                    destination = cube_pos.copy()
                    destination[1] += 0.1
                elif status == 3: #move up
                    destination = claw_pos.copy()
                    destination[2] = cube_pos[2] + 0.026
                    gripperStatus = -gripperMagnitude
                elif status == 4: #move over cube
                    destination = cube_pos
                    destination[2] = cube_pos[2] + 0.026
                    gripperStatus = -gripperMagnitude
                elif status == 5: #reach
                    gripperStatus = -gripperMagnitude
                    destination = cube_pos
                elif status == 6: #grab
                    gripperStatus = gripperMagnitude
                elif status == 7: #lift
                    destination = cube_pos
                    destination[2] = cube_pos[2] + 0.1
                    gripperStatus = gripperMagnitude
                elif status == 8:
                    destination = bin_pos
                    destination[2] = claw_pos[2] #keep altidude, move to center of bin
                    gripperStatus = gripperMagnitude
                elif status == 9:
                    gripperStatus = -gripperMagnitude
                    destination = claw_pos
                else:
                    raise Exception("this should not have happened")

                #displacement is the vector that we want to travel in
                displacement = destination - claw_pos

                #state transtions
                if np.linalg.norm(displacement) < 0.02 and status == 8: #drop
                    status = 9
                if claw_pos[2] > 0.95 and status == 7: #position
                    status = 8
                    intervention = False
                    print("RELINQUISH CONTROL (to position and drop)")
                if np.linalg.norm(gripper_pos) < 0.045 and status == 6: #used to be 0.04
                    intervention = False #RELINQUISH CONTROL see if the thing can lift
                    status = 7
                    print("RELINQUISH CONTROL (to lift)")
                if np.linalg.norm(gripper_pos) < 0.001 and status == 7: #close to lift
                    print("regrasping!")
                    status = 5
                if np.linalg.norm(displacement) < 0.02 and status == 5: #plunge to close #used to be 0.01
                    status = 6
                if np.linalg.norm(displacement) < 0.02 and status == 4: #approach to plunge
                    status = 5
                if np.linalg.norm(displacement) < 0.02 and status == 3: #raise to approach
                    status = 4
                if np.linalg.norm(raw_dict["robot0_gripper_joint_force"]) > 0.35 and (status == 1 or status == 0): #remove to raise
                    print("CONTACT")
                    status = 3
                    intervention = False #RELINQUISH CONTROL see if thing can grab
                    print("RELINQUISH CONTROL (to grab)")
                    hasContacted = True
                if np.linalg.norm(displacement) < 0.02 and status == 0: #reach next to the cube
                    status = 1

                displacement = np.multiply(displacement, 5)
                displacement = [component if -1 <= component <= 1 else -1 if component < -1 else 1 for component in displacement] #capping this vector
                print(status_dict[status])

                action = np.append(displacement, gripperStatus)
                assert max(map(abs, action)) <= 1 #making sure we are within our action space
                intervention_list.append(self.step)

            if self.step % 10 == 0:
                print(self.step)
                print('reward', reward)

            raw_dict, next_lowdim, next_obs, reward, done, info = self.env.step(action)

            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            #adds to a list, which will be added to the replay buffer as an episode
            buffer_list.append((lowdim[-1], obs[-1], action, 0,
                                (1.0 if self.step > cfg.sparseProp * cfg.episodeLength else 0.0), next_lowdim[-1], next_obs[-1], done, done_no_max))


            video_frame = self.env.render_highdim_list(200, 200, ["sideview", "agentview"])
            if intervention:
                video_frame[0:30, 0:30] = 0
                video_frame[0:30, 0:30, 0] = 255
            else:
                video_frame[0:30, 0:30] = 0
                video_frame[0:30, 0:30, 1] = 255

            self.video_recorder.simple_record(video_frame) #"agentview",

            #advancing things
            obs = next_obs #update s = s'
            lowdim = next_lowdim
            episode_step += 1
            self.step += 1
            prev_action = action[0:3]

        self.logger.log('train_actor/intervention_prop', intervention_prop / cfg.episodeLength, self.new_replay_buffer.idx)
        print("intervention prop: ", intervention_prop / cfg.episodeLength)
        if reward > 0.99:
            success = 1
        self.video_recorder.clean_up()

        if success and intervention_prop > 0:
            self.new_replay_buffer.add(buffer_list, priority = intervention_list)
            print("****** ADDED ****** and we are at ", self.new_replay_buffer.idx)
        return success and intervention_prop > 0 #discard runs where there is no corrections


    def evaluate(self, successes):
        prop_success = 0
        for episode in range(self.cfg.num_eval_episodes):
            _, lowdim, obs = self.env.reset()
            self.video_recorder.new_recorder_init(f'{self.new_replay_buffer.idx}_eval_{episode}.gif', enabled=True)
            done = False
            episode_reward = 0
            episode_step = 0
            this_success = False
            beg_ = time.time()
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(lowdim, obs/255, sample=False, squash = self.cfg.use_squashed)
                _, lowdim, obs, reward, done, info = self.env.step(action)
                sys.stdout.write("..")
                sys.stdout.flush()

                self.video_recorder.simple_record(self.env.render_highdim_list(200, 200, ["agentview", "sideview"]))
                prop_success = prop_success + 1 if not(this_success) and reward > 0.99 else prop_success
                this_success = True if reward > 0.99 else this_success
                episode_reward += reward
                episode_step += 1

            print("Evaluate episode {0} done".format(episode))
            self.video_recorder.clean_up()

        prop_success /= self.cfg.num_eval_episodes
        self.logger.log('eval/prop_success', prop_success, self.new_replay_buffer.idx)
        self.logger.dump(successes)


    def run(self, cfg):
        #add mapping to different functions
        self.logger.log('train_actor/loss', 0, 0) #to suppress the error
        self.logger.log('train_actor/old_loss', 0, 0) #to suppress the error
        self.logger.log('train_actor/new_loss', 0, 0) #to suppress the error
        self.logger.log('train_actor/intervention_prop', 0, 0) #to suppress the error
        func_dict = {"IndicatorBoxBlock": self.single_demo_indicator_boxblock,
                    "BlockedPickPlace": self.single_demo_pick_place}
        task = func_dict[cfg.environmentName]
        counter = 0
        successes = 0
        assert not self.new_replay_buffer.full, "your buffer is already full!"
        numMoreSuccesses = int(self.cfg.num_corrections - self.new_replay_buffer.idx)
        numCurrSuccesses = int(self.new_replay_buffer.idx)
        memory_logger = open("memory.txt", "w")
        self.evaluate(successes)
        try_counter = 0

        while successes <= numMoreSuccesses:
            if self.new_replay_buffer.idx > self.cfg.warmup and isSuccessful: #prevents double-training on a failure
                print("UPDATING BUFFER ITERABLE")
                del self.new_replay_buffer_iterable
                gc.collect()
                self.new_replay_buffer_iterable = iter(self.new_replay_buffer_dataloader)
                process = psutil.Process(os.getpid())
                memory_logger.write(str((process.memory_info().rss) / 1e9) + "\n")
                print("Memory usage in gb: ", (process.memory_info().rss) / 1e9)
                print("FINISHED UPDATING BUFFER ITERABLE")
                for i in range(cfg.updates_per_episode):  #train model
                    if i % 100 == 0:
                        print("\t", i)
                    self.agent.update_bc_balanced(self.old_replay_buffer, self.new_replay_buffer_iterable, self.logger,
                                                  cfg.updates_per_episode * (self.new_replay_buffer.idx - self.cfg.warmup - 1) + i, squash = self.cfg.use_squashed)
                print("trained!")
            if self.new_replay_buffer.idx % self.cfg.rollouts_per_eval == 0:
                print("SAVING SAVING")
                self.agent.save(self.new_replay_buffer.idx, self.work_dir)

            isSuccessful = task(successes + numCurrSuccesses, cfg)
            counter += 1
            self.step = 0
            successes += isSuccessful
            try_counter += 1
            print(f"\t{successes} successes out of {try_counter} tries. {numMoreSuccesses - successes} to go.")

            if successes % self.cfg.rollouts_per_eval == 0 and isSuccessful: #the isSuccessful prevents double-evals
                self.evaluate(successes)
            self.logger.dump(successes, save=True)
        pkl.dump(self.new_replay_buffer, open(self.work_dir + "/demos_finished.pkl", "wb" ), protocol=4 )


@hydra.main(config_path='sim_intervention_episodes.yaml', strict=True)
def main(cfg):
    from sim_intervention_episodes import Workspace as W
    workspace = W(cfg)
    workspace.run(cfg)

if __name__ == '__main__':
    main()

