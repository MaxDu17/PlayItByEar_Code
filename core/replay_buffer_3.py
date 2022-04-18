# replay buffer for simulation experiments

import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F

import core.utils as utils
from collections import deque
from torch.utils.data import IterableDataset

import random

class SingleEpisode():
    def __init__(self, episode_length, lowdim_shape, obs_shape, action_shape, image_pad, device):
        self.episode_length = episode_length

        self.lowdim = np.empty((episode_length, *lowdim_shape), dtype=np.float32)
        self.next_lowdim = np.empty((episode_length, *lowdim_shape), dtype=np.float32)

        self.obses = np.empty((episode_length, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((episode_length, *obs_shape), dtype=np.uint8)

        self.actions = np.empty((episode_length, *action_shape), dtype=np.float32)

        self.shaped_rewards = np.empty((episode_length, 1), dtype=np.float32)
        self.sparse_rewards = np.empty((episode_length, 1), dtype=np.float32)

        self.not_dones = np.empty((episode_length, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((episode_length, 1), dtype=np.float32)

        self.device = device
        self.idx = 0
        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

        self.priorities = None #which indexes to select the most from

    def _add(self, lowdim, obs, action, shaped_reward, sparse_reward, next_lowdim, next_obs, done, done_no_max):
        assert self.idx < self.episode_length, "something's wrong! You added too much data "
        np.copyto(self.lowdim[self.idx], lowdim)
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.shaped_rewards[self.idx], shaped_reward)
        np.copyto(self.sparse_rewards[self.idx], sparse_reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.next_lowdim[self.idx], next_lowdim)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        self.idx = self.idx + 1

    def add(self, episode_list): #so this requires a very minimal change to the current demo-collecting software
        for step in episode_list: #step is a dictionary
            self._add(*step)
        assert self.idx % self.episode_length == 0

    def setPriority(self, priority):
        self.priorities = priority

    def indexed_rollout(self, length, index): #for residual stuff
        startIdx = index
        #padding allows for the sampling of initial actions

        padded_lowdim = np.pad(self.lowdim, ((length - 1, 0), (0, 0), (0, 0)), "edge")
        padded_next_lowdim = np.pad(self.next_lowdim, ((length - 1, 0), (0, 0), (0, 0)), "edge")
        padded_obses = np.pad(self.obses, ((length - 1, 0), (0, 0), (0, 0), (0, 0)), "edge")
        padded_next_obses = np.pad(self.next_obses, ((length - 1, 0), (0, 0), (0, 0), (0, 0)), "edge")

        lowdim = padded_lowdim[startIdx:startIdx + length]
        next_lowdim = padded_next_lowdim[startIdx:startIdx + length]
        obses = padded_obses[startIdx:startIdx + length]
        next_obses = padded_next_obses[startIdx:startIdx + length]

        padded_actions = np.pad(self.actions, ((length - 1, 0), (0, 0)), "edge")
        actions = padded_actions[startIdx:startIdx + length]

        shaped_buffer = self.shaped_rewards
        sparse_buffer = self.sparse_rewards

        padded_shaped_buffer = np.pad(shaped_buffer, ((length - 1, 0), (0, 0)), "edge")
        padded_sparse_buffer = np.pad(sparse_buffer, ((length - 1, 0), (0, 0)), "edge")

        shaped_rewards = padded_shaped_buffer[startIdx:startIdx + length]
        sparse_rewards = padded_sparse_buffer[startIdx:startIdx + length]

        padded_not_dones_no_max = np.pad(self.not_dones_no_max, ((length - 1, 0), (0, 0)), "edge")
        not_dones_no_max = padded_not_dones_no_max[startIdx:startIdx + length]

        padded_not_dones = np.pad(self.not_dones, ((length - 1, 0), (0, 0)), "edge")
        not_dones = padded_not_dones[startIdx:startIdx + length]

        return lowdim, obses, actions, shaped_rewards, sparse_rewards, next_lowdim, next_obses, not_dones, not_dones_no_max

    def sample_rollout_episode(self, length, shaped_rewards = True, correctionsOnly = False):
        if correctionsOnly:
            assert hasattr(self, 'priorities'), "whoops! You're trying to select priorities without having a priorities list"
            assert self.priorities is not None, "whoops! You've forgotten to fill ths priorities list!"
            if len(self.priorities) == 0:
                #this is a catch-all; normally we shouldn't be running this
                print("no corrections found! Selecting randomly")
                startIdx = np.random.randint(0, self.episode_length)
            else:
                startIdx = random.choice(self.priorities)
            #now, technically we are doing this:  We step back to account for the starting position, and we step forward to account for the padding.
            #as a net result, nothing changes.
#             startIdx -= (length - 1)
#             startIdx += (length - 1)
        else:
            startIdx = np.random.randint(0, self.episode_length)


        #padding allows for the sampling of initial actions
        padded_lowdim = np.pad(self.lowdim, ((length - 1, 0), (0, 0), (0, 0)), "edge")
        padded_next_lowdim = np.pad(self.next_lowdim, ((length - 1, 0), (0, 0), (0, 0)), "edge")
        padded_obses = np.pad(self.obses, ((length - 1, 0), (0, 0), (0, 0), (0, 0)), "edge")
        padded_next_obses = np.pad(self.next_obses, ((length - 1, 0), (0, 0), (0, 0), (0, 0)), "edge")

        lowdim = padded_lowdim[startIdx:startIdx + length]
        next_lowdim = padded_next_lowdim[startIdx:startIdx + length]
        obses = padded_obses[startIdx:startIdx + length]
        next_obses = padded_next_obses[startIdx:startIdx + length]


        padded_actions = np.pad(self.actions, ((length - 1, 0), (0, 0)), "edge")
        actions = padded_actions[startIdx:startIdx + length]

        reward_buffer = self.shaped_rewards if shaped_rewards else self.sparse_rewards
        padded_reward_buffer = np.pad(reward_buffer, ((length - 1, 0), (0, 0)), "edge")
        rewards = padded_reward_buffer[startIdx:startIdx + length]

        padded_not_dones_no_max = np.pad(self.not_dones_no_max, ((length - 1, 0), (0, 0)), "edge")
        not_dones_no_max = padded_not_dones_no_max[startIdx:startIdx + length]

        return lowdim, obses, actions, rewards, next_lowdim, next_obses, not_dones_no_max

class ReplayBufferDoubleRewardEpisodes(IterableDataset): #object,
    """Buffer to store environment transitions."""
    def __init__(self, lowdim_shape, obs_shape, action_shape, episodes, episode_length, image_pad, device):
        self.numEpisodes = episodes
        self.device = device
        self.episodeLength = episode_length
        self.lowdim_shape = lowdim_shape
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.image_pad = image_pad

        self.allEpisodes = deque([], maxlen = int(self.numEpisodes))
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.numEpisodes if self.full else self.idx

    def set_sample_settings(self, length =10, shaped_rewards = False, trainprop = 1, train = True, correctionsOnly = False):
        self.length = length
        self.shaped_rewards = shaped_rewards
        self.trainprop = trainprop
        self.train = train
        self.correctionsOnly = correctionsOnly


    def add(self, episode_list, priority = None):
        episode_object = SingleEpisode(self.episodeLength, self.lowdim_shape, self.obs_shape, self.action_shape, self.image_pad, self.device)
        episode_object.add(episode_list)
        if priority is not None:
            episode_object.setPriority(priority)

        self.allEpisodes.append(episode_object)

        self.idx = (self.idx + 1) % self.numEpisodes
        self.full = self.full or self.idx == 0


    def sampleRollout(self): #defaulting for now
        assert len(self.allEpisodes) <= self.numEpisodes #making sure that we are indeed rolling the buffer

        if self.train:
            index = np.random.randint(0, self.trainprop * (self.numEpisodes if self.full else self.idx))
        else:
            print("TEST TEST TEST")
            index = np.random.randint(self.trainprop * (self.numEpisodes if self.full else self.idx),
                                     self.numEpisodes if self.full else self.idx)
        if self.correctionsOnly:
        #not good style, but this works
            while len(self.allEpisodes[index].priorities) == 0:
                index = np.random.randint(0,
                                     self.trainprop * (self.numEpisodes if self.full else self.idx))

        lowdim, obses, actions, rewards, next_lowdim, next_obses, not_dones_no_max = self.allEpisodes[index].sample_rollout_episode(self.length, self.shaped_rewards, correctionsOnly = self.correctionsOnly)

        return lowdim, obses, actions, rewards, next_lowdim, next_obses, not_dones_no_max


    def __iter__(self):
        while True:
            yield self.sampleRollout()

