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
    
    if cfg.stacked:
        print("I AM STACKED")
        cfg.agent.params.lowdim_dim = 10 * env.get_lowdim_dims(cfg.modalities)
        env = utils.FrameStack_Lowdim(env, cfg, k=3, l_k = 10, frameMode = 'cat', demo = True, audio = False)
    else:
        cfg.agent.params.lowdim_dim = env.get_lowdim_dims(cfg.modalities)
        env = utils.FrameStack_Lowdim(env, cfg, k=1, l_k = 1, frameMode = 'cat', demo = True, audio = False)
        
    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        print("cuda status: ", torch.cuda.is_available())
        self.cfg = cfg
        copyfile("../../writeDemos_episodes.yaml", "hyperparameters.yaml") #makes sure we keep track of hyperparameters 
        copyfile("../../writeDemos_episodes.py", "hyperparameters.py") #makes sure we keep track of hyperparameters 

        
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)
        
#         input(self.env.lowdim_space)
#         input(self.env.observation_space.shape)
        #replay buffer is used to train the model during update steps 
        input(self.env.observation_space.shape)
        self.replay_buffer = ReplayBuffer(self.env.lowdim_space,
                                          self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          self.cfg.episodes,
                                          self.cfg.episodeLength,
                                          self.cfg.image_pad, self.device)
        
#           self.squashed_replay_buffer = ReplayBufferEpisode([1, 7],
#                                   (3, 84, 84),
#                                   (57, 160),
#                                   (4,),
#                                   self.replay_buffer.numEpisodes,
#                                   self.replay_buffer.episodeLength,
#                                   self.cfg.image_pad, self.device) #hard coded for now 
        
        
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0
        
    def single_demo_lift(self, iteration, cfg, record = False):     
        ZFACTOR = 0.01 #put this in config later
        XYFACTOR = 0.005
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        
        raw_dict, lowdim, obs = self.env.reset()
        done = False
        episode_reward = 0
        episode_step = 0
        episode += 1  
        if record:
            self.video_recorder.new_recorder_init(f'demo_' + str(iteration) + '.gif')
        
        success = 0
        buffer_list = list()
        gripperStatus = -1
        liftStatus = False 
        buffer_list = list()
        reward = 0
        while self.step < cfg.episodeLength: #avoid magic numbers; I'll make this code more elegant later 
            cube_pos = raw_dict["cube_pos"]
            claw_pos = raw_dict["robot0_eef_pos"]
#             input(raw_dict["robot0_gripper_joint_force"])
#             print(self.step, raw_dict["object_sound"])
#             print(raw_dict["gripper_tip_force"])
#             print("\t", raw_dict["gripper_force"])
            target_pos = cube_pos
            
            if reward > 0.51: #lift, uses 0.51, soft uses 0.435
                liftStatus = True 
                
            if reward > 0.99 and claw_pos[2] > 0.95:
                liftStatus = False
                target_pos = claw_pos
                
            gripperStatus = 1 if reward > 0.42 else -1
                
            displacement = target_pos - claw_pos
            displacement = np.multiply(displacement, 5)
            displacement[0] += (np.random.rand() - 0.5) * XYFACTOR
            displacement[1] += (np.random.rand() - 0.5) * XYFACTOR
            displacement[2] += (np.random.rand() - 0.5) * ZFACTOR
            
            # make sure actions within action space
            displacement = [component if -1 <= component <= 1 else -1 if component < -1 else 1 for component in displacement] 
            
            if not(liftStatus):
                action = np.append(displacement, gripperStatus)
            else:
                action = np.array([(np.random.rand() - 0.5) * 0.5, (np.random.rand() - 0.5) * 0.5, 1, gripperStatus])
                
            assert max(map(abs, action)) <= 1 #making sure we are within our action space 
            
            raw_dict, next_lowdim, next_obs, reward, done, info = self.env.step(action)
            
            if record:
                self.video_recorder.simple_record(self.env.render_highdim_list(256, 256, ["agentview", "sideview"]))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward
            sparse_reward = None
#             print((1 if self.step > cfg.sparseProp * cfg.episodeLength else 0))
      
            buffer_list.append((lowdim, obs, action, reward, 
                                (1.0 if self.step > cfg.sparseProp * cfg.episodeLength else 0.0), next_lowdim, next_obs, done, done_no_max))
            if self.step % 10 == 0:
                print(reward)
            if reward > 0.99:
                success = 1 
                
            obs = next_obs
            lowdim = next_lowdim
            episode_step += 1
            self.step += 1
            
        if record:
                self.video_recorder.clean_up()    
        if success == 1:
            self.replay_buffer.add(buffer_list)
#             for buf in buffer_list:
#                 self.replay_buffer.add(*buf)
            print("****** ADDED ****** and we are at ", self.replay_buffer.idx)
        return success 
    
    def record_env(self, iteration, cfg, record = False):     
        self.env.reset()
        self.video_recorder.new_recorder_init(f'demo_' + str(iteration) + '.gif')
        for i in range(10):
            self.video_recorder.simple_record(self.env.render_highdim_list(256, 256, ["agentview", "birdview"]))
            self.env.step([0, 0, 0, 0])
        self.video_recorder.clean_up()    
        return 1

    def single_demo_cloth(self, iteration, cfg, record = False):
        assert cfg.environmentName == "ClothCube"
        SCALINGFACTOR = 25 #put this in config later
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        raw_dict, lowdim, obs = self.env.reset()
        done = False
        episode_reward = 0
        episode_step = 0
        episode += 1  
        if record:
            self.video_recorder.new_recorder_init(f'demo_' + str(iteration) + '.gif')

        status = 0 
        
        gripperMagnitude = 1
        gripperStatus = -gripperMagnitude
        liftStatus = False 
        destination = np.zeros(3)
        success = 0
        
        buffer_list = list()
        raw_dict, next_lowdim, next_obs, reward, done, info = self.env.step([0, 0, 0, gripperMagnitude])
        initial_cube = raw_dict["cube_pos"]
    
        while self.step < cfg.episodeLength: 
            cube_pos = raw_dict["cube_pos"]
            claw_pos = raw_dict["robot0_eef_pos"]
#             print("\t", raw_dict["object_sound"])
            gripper_pos = raw_dict["robot0_gripper_qpos"]
            status_dict = {0 : "clothreach", 1 : "clothgrip", 2 : "removecloth", 3: "moveup", 4 : "positioning", 
                          5 : "blockreach", 6 : "grabbing", 7 : "lifting", 8: "HALT"}
            if status == 0: #reach for cloth
                destination = cube_pos
                destination[0] -= 0.05
                destination[1] -= 0.05
                destination[2] = 0.7
                
            elif status == 1: #grip cloth
                destination = claw_pos.copy()
                destination[2] = 0.8
                gripperStatus = gripperMagnitude
                
            elif status == 2: #move cloth to the side
                destination = cube_pos
                if raw_dict["gripper_force"][2] < 0.4:
                    destination[2] = 0.6
                else:
                    destination[2] = 0.8
                destination[1] -= 0.3
                destination[0] -= 0.1
                gripperStatus = gripperMagnitude
                
            elif status == 3: #move up 
                destination = claw_pos.copy()
                destination[2] = cube_pos[2] + 0.1
                gripperStatus = -gripperMagnitude
                
            elif status == 4: #move over cube
                destination = cube_pos
                destination[2] = cube_pos[2] + 0.1
                gripperStatus = -gripperMagnitude
                
            elif status == 5: #reach
                gripperStatus = -gripperMagnitude
                destination = cube_pos
                
            elif status == 6: #grab 
                gripperStatus = gripperMagnitude
                
            elif status == 7: #lift
                liftStatus = True 
                gripperStatus = gripperMagnitude
            
            elif status == 8:
                destination = claw_pos
                liftStatus = False
            
            else:
                raise Exception("whoops!")
  
            displacement = destination - claw_pos

            if claw_pos[2] > 0.95 and status == 7:
                status = 8
            if np.linalg.norm(gripper_pos) < 0.045 and status == 6: #used to be 0.04
                status = 7
            if np.linalg.norm(claw_pos - cube_pos) > 0.1 and status == 7: #close to lift
                print("regrasping!")
                status = 5 
            if np.linalg.norm(displacement) < 0.02 and status == 5: #plunge to close #used to be 0.01
                status = 6
            if np.linalg.norm(displacement) < 0.02 and status == 4: #approach to plunge
                status = 5 
            if np.linalg.norm(displacement) < 0.02 and status == 3: #raise to approach
                status = 4
            if np.linalg.norm(displacement) < 0.12 and status == 2: #remove to raise
                status = 3
            if np.linalg.norm(gripper_pos) < 0.03 and status == 1: #grip to remove
                status = 2
            if raw_dict["gripper_force"][2] > 0 and status == 0: #reach to grip
                status = 1

            displacement = np.multiply(displacement, 5)
            
            if status in (0, 3, 4): #add randomness to non-sensitive positions 
                displacement = [component + (np.random.rand() - 0.5) * 0.05 for component in displacement] #adding randomness 

            displacement = [component if -1 <= component <= 1 else -1 if component < -1 else 1 for component in displacement]

            if not(liftStatus):
                action = np.append(displacement, gripperStatus)
            else:
                action = np.array([(np.random.rand() - 0.5) * 0.5, (np.random.rand() - 0.5) * 0.5, 1, gripperStatus])
                
            assert max(map(abs, action)) <= 1 #making sure we are within our action space 
            
            raw_dict, next_lowdim, next_obs, reward, done, info = self.env.step(action)
            
            if record:
                self.video_recorder.simple_record(self.env.render_highdim(256, 256, "agentview"))
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            buffer_list.append((lowdim, obs, action, reward, 
                                (1.0 if self.step > cfg.sparseProp * cfg.episodeLength else 0.0), next_lowdim, next_obs, done, done_no_max))
            
            if self.step % 10 == 0:
                print(reward)
                print(status, " : ", status_dict[status])
            if reward > 0.99:
                success = 1 
            obs = next_obs
            lowdim = next_lowdim
            episode_step += 1
            self.step += 1
            
        if record:
            self.video_recorder.clean_up()
        if success == 1:
            self.replay_buffer.add(buffer_list)
#             for buf in buffer_list:
#                 self.replay_buffer.add(*buf)
            print("****** ADDED ****** and we are at ", self.replay_buffer.idx)
        return success 
    
    def single_demo_pick_place(self, iteration, cfg, record = False):
        assert cfg.environmentName == "BlockedPickPlace"
        SCALINGFACTOR = 25 #put this in config later
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        raw_dict, lowdim, obs = self.env.reset()

        done = False
        episode_reward = 0
        episode_step = 0
        episode += 1  
        if record:
            self.video_recorder.new_recorder_init(f'demo_' + str(iteration) + '.gif')

        status = 0 
        
        gripperMagnitude = 1
        gripperStatus = -gripperMagnitude
        liftStatus = False 
        destination = np.zeros(3)
        success = 0
        
        buffer_list = list()
        raw_dict, next_lowdim, next_obs, reward, done, info = self.env.step([0, 0, 0, gripperMagnitude])
        initial_cube = raw_dict["cube_pos"]
        
        bin_pos = raw_dict['bin_pos']
    
        while self.step < cfg.episodeLength: 
            cube_pos = raw_dict["cube_pos"]
            claw_pos = raw_dict["robot0_eef_pos"]
#             print("\t", raw_dict["object_sound"])
            gripper_pos = raw_dict["robot0_gripper_qpos"]
            status_dict = {0 : "sidereach", 1 : "sidestep", 2 : "asdfasdad", 3: "moveup", 4 : "positioning", 
                          5 : "blockreach", 6 : "grabbing", 7 : "lifting", 8: "position", 9: "drop"}
            if status == 0: #reach for cloth
                destination = [0, 0.08, 0.82]
#                 destination[1] -= 0.2
              
            elif status == 1: #move to the side
                destination = cube_pos
                destination[1] += 0.01
                
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
                liftStatus = True 
                gripperStatus = gripperMagnitude
            
            elif status == 8:
                destination = bin_pos
                destination[2] = claw_pos[2] #keep altidude, move to center of bin 
                liftStatus = False
                gripperStatus = gripperMagnitude
            
            elif status == 9: 
                gripperStatus = -gripperMagnitude 
                
            else:
                raise Exception("whoops!")
  
            displacement = destination - claw_pos
            
            if np.linalg.norm(displacement) < 0.02 and status == 8:
                status = 9
            if claw_pos[2] > 0.95 and status == 7:
                status = 8
            if np.linalg.norm(gripper_pos) < 0.045 and status == 6: #used to be 0.04
                status = 7
            if np.linalg.norm(claw_pos - cube_pos) > 0.1 and status == 7: #close to lift
                print("regrasping!")
                status = 0 
            if np.linalg.norm(displacement) < 0.02 and status == 5: #plunge to close #used to be 0.01
                status = 6
            if np.linalg.norm(displacement) < 0.02 and status == 4: #approach to plunge
                status = 5 
            if np.linalg.norm(displacement) < 0.02 and status == 3: #raise to approach
                status = 4
            if np.linalg.norm(raw_dict["robot0_gripper_joint_force"]) > 0.35 and status == 1: #remove to raise
                print("CONTACT")
                status = 3
            if np.linalg.norm(displacement) < 0.02 and status == 0: #reach next to the cube
                status = 1

            displacement = np.multiply(displacement, 5)
            
#             if status in (0, 3, 4): #add randomness to non-sensitive positions 
            displacement = [component + (np.random.rand() - 0.5) * 0.3 for component in displacement] #adding randomness 

            displacement = [component if -1 <= component <= 1 else -1 if component < -1 else 1 for component in displacement]


            if not(liftStatus):
                action = np.append(displacement, gripperStatus)
            else:
                action = np.array([(np.random.rand() - 0.5) * 0.5, (np.random.rand() - 0.5) * 0.5, 1, gripperStatus])
                
            assert max(map(abs, action)) <= 1 #making sure we are within our action space 
            
            raw_dict, next_lowdim, next_obs, reward, done, info = self.env.step(action)
            
            if record:
                self.video_recorder.simple_record(self.env.render_highdim_list(256, 256, ["agentview", "sideview"]))
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            buffer_list.append((lowdim, obs, action, reward, 
                                (1.0 if self.step > cfg.sparseProp * cfg.episodeLength else 0.0), next_lowdim, next_obs, done, done_no_max))
            
            if self.step % 10 == 0:
                print(reward)
#                 print(raw_dict["robot0_gripper_joint_force"])
                print(status_dict[status])
#                 print(status, " : ", np.linalg.norm(raw_dict["robot0_gripper_joint_force"]))
            if reward > 0.99:
                success = 1 
            obs = next_obs
            lowdim = next_lowdim
            episode_step += 1
            self.step += 1
            
        if record:
            self.video_recorder.clean_up()
        if success == 1:
            self.replay_buffer.add(buffer_list)
#             for buf in buffer_list:
#                 self.replay_buffer.add(*buf)
            print("****** ADDED ****** and we are at ", self.replay_buffer.idx)
        return success 
    
    def single_demo_indicator_box(self, iteration, cfg, record = False):
        assert cfg.environmentName == "IndicatorBoxBlock"
        SCALINGFACTOR = 25 #put this in config later
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        raw_dict, lowdim, obs = self.env.reset()
        
        done = False
        episode_reward = 0
        episode_step = 0
        episode += 1  
        if record:
            self.video_recorder.new_recorder_init(f'demo_' + str(iteration) + '.gif')

        status = 0 
        
        gripperMagnitude = 1
        gripperStatus = -gripperMagnitude
        liftStatus = False 
        destination = np.zeros(3)
        success = 0
        
        buffer_list = list()
        raw_dict, next_lowdim, next_obs, reward, done, info = self.env.step([0, 0, 0, gripperMagnitude])
        initial_cube = raw_dict["cube_pos"]
#         input(raw_dict.keys())
        
        #bad programming style, but this is the easiest way 
        if initial_cube[1] < -0.1:
            target_y = -0.3
        elif initial_cube[1] < 0:
            target_y = -0.2
        elif initial_cube[1] < 0.1:
            target_y = -0.1
        else: 
            target_y = 0
                    
        
        while self.step < cfg.episodeLength: 
            cube_pos = raw_dict["cube_pos"]
            claw_pos = raw_dict["robot0_eef_pos"]
#             print("\t", raw_dict["object_sound"])
            gripper_pos = raw_dict["robot0_gripper_qpos"]
            status_dict = {0 : "sidereach", 1 : "sidestep", 2 : "why am I seeing this", 3: "moveup", 4 : "positioning", 
                          5 : "blockreach", 6 : "grabbing", 7 : "lifting", 8: "HALT"}
            if status == 0: #reach for cube
                destination = cube_pos
                destination[1] = target_y
              
            elif status == 1: #move to the side
                destination = cube_pos
                destination[1] += 0.11 #change back to 0.01
                
            elif status == 3: #move up 
                destination = claw_pos.copy()
                destination[2] = cube_pos[2] + 0.03
                gripperStatus = -gripperMagnitude
                
            elif status == 4: #move over cube
                destination = cube_pos
                destination[2] = cube_pos[2] + 0.03
                gripperStatus = -gripperMagnitude
                
            elif status == 5: #reach
                gripperStatus = -gripperMagnitude
                destination = cube_pos
                
            elif status == 6: #grab 
                gripperStatus = gripperMagnitude
                
            elif status == 7: #lift
                liftStatus = True 
                gripperStatus = gripperMagnitude
            
            elif status == 8:
                destination = claw_pos
                liftStatus = False
            
            else:
                raise Exception("whoops!")
  
            displacement = destination - claw_pos

#             print("\t", np.linalg.norm(raw_dict["robot0_gripper_joint_force"]))
            if claw_pos[2] > 0.95 and status == 7:
                status = 8
            if np.linalg.norm(gripper_pos) < 0.045 and status == 6: #used to be 0.04
                status = 7
            if np.linalg.norm(claw_pos - cube_pos) > 0.1 and status == 7: #close to lift
                print("regrasping!")
                status = 0 
            if np.linalg.norm(displacement) < 0.02 and status == 5: #plunge to close #used to be 0.01
                status = 6
            if np.linalg.norm(displacement) < 0.02 and status == 4: #approach to plunge
                status = 5 
            if np.linalg.norm(displacement) < 0.02 and status == 3: #raise to approach
                status = 4
#             if np.linalg.norm(raw_dict["robot0_gripper_joint_force"]) > 0.35 and status == 1: #remove to raise formerly 1.3 
#                 print("CONTACT")
#                 status = 3
            if np.linalg.norm(displacement) < 0.02 and status == 0: #reach next to the cube
                status = 1

            displacement = np.multiply(displacement, 5)
            
#             if status in (0, 3, 4): #add randomness to non-sensitive positions 
            displacement = [component + (np.random.rand() - 0.5) * 0.3 for component in displacement] #adding randomness 

            displacement = [component if -1 <= component <= 1 else -1 if component < -1 else 1 for component in displacement]

            if not(liftStatus):
                action = np.append(displacement, gripperStatus)
            else:
                action = np.array([(np.random.rand() - 0.5) * 0.5, (np.random.rand() - 0.5) * 0.5, 1, gripperStatus])
                
            assert max(map(abs, action)) <= 1 #making sure we are within our action space 
            
            raw_dict, next_lowdim, next_obs, reward, done, info = self.env.step(action)
            
            if record:
                self.video_recorder.simple_record(self.env.render_highdim_list(256, 256, ["agentview", "sideview"]))
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            buffer_list.append((lowdim, obs, action, reward, 
                                (1.0 if self.step > cfg.sparseProp * cfg.episodeLength else 0.0), next_lowdim, next_obs, done, done_no_max))
            
            if self.step % 10 == 0:
                print(reward)
#                 print(raw_dict["robot0_gripper_joint_force"])
                print(status_dict[status])
#                 print(status, " : ", np.linalg.norm(raw_dict["robot0_gripper_joint_force"]))
            if reward > 0.99:
                success = 1 
            obs = next_obs
            lowdim = next_lowdim
            episode_step += 1
            self.step += 1
            
        if record:
            self.video_recorder.clean_up()
        if success == 1:
            self.replay_buffer.add(buffer_list)
#             for buf in buffer_list:
#                 self.replay_buffer.add(*buf)
            print("****** ADDED ****** and we are at ", self.replay_buffer.idx)
        return success 
    
    def single_demo_box_searching(self, iteration, cfg, record = False):
        assert cfg.environmentName == "BoxBlock"
        SCALINGFACTOR = 25 #put this in config later
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        raw_dict, lowdim, obs = self.env.reset()
        done = False
        episode_reward = 0
        episode_step = 0
        episode += 1  
        if record:
            self.video_recorder.new_recorder_init(f'demo_' + str(iteration) + '.gif')

        status = 0 
        
        gripperMagnitude = 1
        gripperStatus = -gripperMagnitude
        liftStatus = False 
        destination = np.zeros(3)
        success = 0
        
        buffer_list = list()
        raw_dict, next_lowdim, next_obs, reward, done, info = self.env.step([0, 0, 0, gripperMagnitude])
        initial_cube = raw_dict["cube_pos"]
        target = initial_cube
        target[0] = -0.02
        
        
        while self.step < cfg.episodeLength: 
            cube_pos = raw_dict["cube_pos"]
            claw_pos = raw_dict["robot0_eef_pos"]
#             print("\t", raw_dict["object_sound"])
            gripper_pos = raw_dict["robot0_gripper_qpos"]
            status_dict = {0 : "sidereach", 1 : "sidestep", 2 : "asdfasdad", 3: "moveup", 4 : "positioning", 
                          5 : "blockreach", 6 : "grabbing", 7 : "lifting", 8: "HALT"}
            if status == 0: #reach for table
                destination = target.copy()
                destination[1] -= 0.2
              
            elif status == 1: #move to the side
                destination = target.copy()
                destination[1] += 0.1
                
            elif status == 3: #move up 
                destination = claw_pos.copy()
                destination[2] = cube_pos[2] + 0.1
                gripperStatus = -gripperMagnitude
                
            elif status == 4: #move over cube
                destination = cube_pos
                destination[2] = cube_pos[2] + 0.1
                gripperStatus = -gripperMagnitude
                
            elif status == 5: #reach
                gripperStatus = -gripperMagnitude
                destination = cube_pos
                
            elif status == 6: #grab 
                gripperStatus = gripperMagnitude
                
            elif status == 7: #lift
                liftStatus = True 
                gripperStatus = gripperMagnitude
            
            elif status == 8:
                destination = claw_pos
                liftStatus = False
            
            else:
                raise Exception("whoops!")
#             print(target)
            displacement = destination - claw_pos
            
            if claw_pos[2] > 0.95 and status == 7:
                status = 8
            if np.linalg.norm(gripper_pos) < 0.045 and status == 6: #used to be 0.04
                status = 7
            if np.linalg.norm(claw_pos - cube_pos) > 0.1 and status == 7: #close to lift
                print("regrasping!")
                status = 0 
            if np.linalg.norm(displacement) < 0.02 and status == 5: #plunge to close #used to be 0.01
                status = 6
            if np.linalg.norm(displacement) < 0.02 and status == 4: #approach to plunge
                status = 5 
            if np.linalg.norm(displacement) < 0.02 and status == 3: #raise to approach
                status = 4
#             print(np.linalg.norm(raw_dict["robot0_gripper_joint_force"]))
            if np.linalg.norm(raw_dict["robot0_gripper_joint_force"]) > 0.8 and (status == 1 or status == 0): #remove to raise
                print("CONTACT")
                status = 3
            
            if claw_pos[1] > 0.05 and status == 1:
                print("SWIPE FAILED")
                status = 0
                target[0] += 0.01
#                 target[1] = cube_pos[1] #making sure that the y location hasn't 
                
            if np.linalg.norm(displacement) < 0.02 and status == 0: #reach next to the cube
                status = 1

            displacement = np.multiply(displacement, 5)
            
#             if status in (0, 3, 4): #add randomness to non-sensitive positions 
            displacement = [component + (np.random.rand() - 0.5) * 0.2 for component in displacement] #adding randomness 

            displacement = [component if -1 <= component <= 1 else -1 if component < -1 else 1 for component in displacement]

            if not(liftStatus):
                action = np.append(displacement, gripperStatus)
            else:
                action = np.array([(np.random.rand() - 0.5) * 0.5, (np.random.rand() - 0.5) * 0.5, 1, gripperStatus])
                
            assert max(map(abs, action)) <= 1 #making sure we are within our action space 
            
            raw_dict, next_lowdim, next_obs, reward, done, info = self.env.step(action)
            
            if record:
                self.video_recorder.simple_record(self.env.render_highdim_list(256, 256, ["agentview", "sideview"]))
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            buffer_list.append((lowdim[-1], obs[-1], action, reward, 
                                (1.0 if self.step > cfg.sparseProp * cfg.episodeLength else 0.0), next_lowdim[-1], next_obs[-1], done, done_no_max))
            #essentially, we only select the most present one to save, because when we sample, we will be sampling across an episode 
            
            if self.step % 10 == 0:
                print(reward)
#                 print(raw_dict["robot0_gripper_joint_force"])
                print(status_dict[status])
#                 print(status, " : ", np.linalg.norm(raw_dict["robot0_gripper_joint_force"]))
            if reward > 0.99:
                success = 1 
            obs = next_obs
            lowdim = next_lowdim
            episode_step += 1
            self.step += 1
            
        if record:
            self.video_recorder.clean_up()
        if success == 1:
            self.replay_buffer.add(buffer_list)
#             for buf in buffer_list:
#                 self.replay_buffer.add(*buf)
            print("****** ADDED ****** and we are at ", self.replay_buffer.idx)
        return success 
    
    
    
    def run(self, cfg):
        counter = 0
        successes = 0
        func_dict = {"BoxBlock": self.single_demo_box_searching, "Lift": self.single_demo_lift, "MMLift": self.single_demo_lift, 
                    "ClothCube": self.single_demo_cloth, "IndicatorBoxBlock": self.single_demo_indicator_box, "BlockedPickPlace": self.single_demo_pick_place}
        task = func_dict[cfg.environmentName]
        print(task)
        while successes < math.ceil(cfg.episodes):
            isSuccessful = task(counter, cfg, record = (counter % cfg.recordFrq == 0)) #change to cloth if needed
            counter += 1 
            self.step = 0
            successes += isSuccessful
            print("\t", successes, " out of ", counter) 
        pkl.dump(self.replay_buffer, open( "demos.pkl", "wb" ), protocol=4 )

        

@hydra.main(config_path='writeDemos_episodes.yaml', strict=True)
def main(cfg):
    from writeDemos_episodes import Workspace as W
    workspace = W(cfg)
    workspace.run(cfg)
        
if __name__ == '__main__':
    main()

    