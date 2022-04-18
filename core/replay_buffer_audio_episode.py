# replay buffer for robot experiments

import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import core.utils as utils
from collections import deque
from torch.utils.data import IterableDataset
import random
import copy

class SingleEpisode():
    def __init__(self, episode_length, lowdim_shape, obs_shape, audio_shape, action_shape, image_pad, device):
        self.episode_length = episode_length

        self.lowdim = np.empty((episode_length, *lowdim_shape), dtype=np.float32)
        self.next_lowdim = np.empty((episode_length, *lowdim_shape), dtype=np.float32)

        self.audio = np.empty((episode_length, *audio_shape), dtype=np.float32)
        self.next_audio = np.empty((episode_length, *audio_shape), dtype=np.float32)

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

    def _add(self, lowdim, obs, audio,action, shaped_reward, sparse_reward, next_lowdim, next_obs, next_audio, done, done_no_max):
        assert self.idx < self.episode_length, "something's wrong! You added too much data "
        np.copyto(self.lowdim[self.idx], lowdim)
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.audio[self.idx], audio)
        np.copyto(self.shaped_rewards[self.idx], shaped_reward)
        np.copyto(self.sparse_rewards[self.idx], sparse_reward)
        np.copyto(self.next_audio[self.idx], next_audio)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.next_lowdim[self.idx], next_lowdim)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        self.idx = self.idx + 1

    def add(self, episode_list): #so this requires a very minimal change to the current demo-collecting software
        for step in episode_list: #step is a dictionary
            self._add(*step)
        print(self.idx)
        print(self.obses.shape)
        assert self.idx % self.episode_length == 0

    def setPriority(self, priority):
        self.priorities = priority

    def indexed_rollout(self, length, index):
        startIdx = index

        #padding allows for the sampling of initial actions
        padded_lowdim = np.pad(self.lowdim, ((length - 1, 0), (0, 0), (0, 0)), "edge")
        padded_next_lowdim = np.pad(self.next_lowdim, ((length - 1, 0), (0, 0), (0, 0)), "edge")
        lowdim = padded_lowdim[startIdx:startIdx + length]
        next_lowdim = padded_next_lowdim[startIdx:startIdx + length]

        padded_obses = np.pad(self.obses, ((length - 1, 0), (0, 0), (0, 0), (0, 0)), "edge")
        padded_next_obses = np.pad(self.next_obses, ((length - 1, 0), (0, 0), (0, 0), (0, 0)), "edge")
        obses = padded_obses[startIdx:startIdx + length]
        next_obses = padded_next_obses[startIdx:startIdx + length]

        padded_audio = np.pad(self.audio, ((length - 1, 0), (0, 0), (0, 0)), "edge")
        padded_next_audio = np.pad(self.next_audio, ((length - 1, 0), (0, 0), (0, 0)), "edge")
        audio = padded_audio[startIdx:startIdx + length]
        next_audio = padded_next_audio[startIdx:startIdx + length]

        padded_actions = np.pad(self.actions, ((length - 1, 0), (0, 0)), "edge")
        actions = padded_actions[startIdx:startIdx + length]

        padded_shaped_buffer = np.pad(self.shaped_rewards, ((length - 1, 0), (0, 0)), "edge")
        padded_sparse_buffer = np.pad(self.sparse_rewards , ((length - 1, 0), (0, 0)), "edge")
        shaped_rewards = padded_shaped_buffer[startIdx:startIdx + length]
        sparse_rewards = padded_sparse_buffer[startIdx:startIdx + length]

        padded_not_dones_no_max = np.pad(self.not_dones_no_max, ((length - 1, 0), (0, 0)), "edge")
        not_dones_no_max = padded_not_dones_no_max[startIdx:startIdx + length]
        padded_not_dones = np.pad(self.not_dones, ((length - 1, 0), (0, 0)), "edge")
        not_dones = padded_not_dones[startIdx:startIdx + length]

        return lowdim, obses, audio, actions, shaped_rewards, sparse_rewards, next_lowdim, next_obses, next_audio, not_dones_no_max

    def sample_rollout(self, length, shaped_rewards = True, correctionsOnly = False):
        if correctionsOnly:
            assert hasattr(self, 'priorities'), "whoops! You're trying to select priorities without having a priorities list"
            assert self.priorities is not None, "whoops! You've forgotten to fill ths priorities list!"
            if len(self.priorities) == 0:
                #this is a catch-all; normally we shouldn't be running this
                print("no corrections found! Selecting randomly")
                startIdx = np.random.randint(0, self.episode_length)
            else:
                startIdx = random.choice(self.priorities)
            #now, technically we are doing this. We step back to account for the starting position, and we step forward to account for the padding.
            #as a net result, nothing changes.
#             startIdx -= (length - 1)
#             startIdx += (length - 1)

        startIdx = np.random.randint(0, self.episode_length)
        padded_lowdim = np.pad(self.lowdim, ((length - 1, 0), (0, 0), (0, 0)), "edge")
        padded_next_lowdim = np.pad(self.next_lowdim, ((length - 1, 0), (0, 0), (0, 0)), "edge")
        lowdim = padded_lowdim[startIdx:startIdx + length]
        next_lowdim = padded_next_lowdim[startIdx:startIdx + length]

        padded_obses = np.pad(self.obses, ((length - 1, 0), (0, 0), (0, 0), (0, 0)), "edge")
        padded_next_obses = np.pad(self.next_obses, ((length - 1, 0), (0, 0), (0, 0), (0, 0)), "edge")
        obses = padded_obses[startIdx:startIdx + length]
        next_obses = padded_next_obses[startIdx:startIdx + length]

        padded_audio = np.pad(self.audio, ((length - 1, 0), (0, 0), (0, 0)), "edge")
        padded_next_audio = np.pad(self.next_audio, ((length - 1, 0), (0, 0), (0, 0)), "edge")
        audio = padded_audio[startIdx:startIdx + length]
        next_audio = padded_next_audio[startIdx:startIdx + length]

        padded_actions = np.pad(self.actions, ((length - 1, 0), (0, 0)), "edge")
        actions = padded_actions[startIdx:startIdx + length]

        reward_buffer = self.shaped_rewards if shaped_rewards else self.sparse_rewards
        padded_reward_buffer = np.pad(reward_buffer, ((length - 1, 0), (0, 0)), "edge")
        rewards = padded_reward_buffer[startIdx:startIdx + length]

        padded_not_dones_no_max = np.pad(self.not_dones_no_max, ((length - 1, 0), (0, 0)), "edge")
        not_dones_no_max = padded_not_dones_no_max[startIdx:startIdx + length]
        padded_not_dones = np.pad(self.not_dones, ((length - 1, 0), (0, 0)), "edge")
        not_dones = padded_not_dones[startIdx:startIdx + length]

        return lowdim, obses, audio, actions, rewards, next_lowdim, next_obses, next_audio, not_dones_no_max

class ReplayBufferAudioEpisodes(IterableDataset):
    """Buffer to store environment transitions."""
    def __init__(self, lowdim_shape, obs_shape, audio_shape, action_shape, episodes, episode_length, image_pad, device):
        self.numEpisodes = episodes
        self.device = device
        self.episodeLength = episode_length
        self.lowdim_shape = lowdim_shape
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.image_pad = image_pad
        self.audio_shape = audio_shape

        self.allEpisodes = deque([], maxlen = int(self.numEpisodes))

        self.idx = 0
        self.full = False

    def update(self, buffer):
        assert type(buffer) == ReplayBufferAudioEpisodes, "oops looks like the types don't match"
        additional_eps = len(buffer.allEpisodes)
        self.numEpisodes += additional_eps
        self.newAllEpisodes = deque([], maxlen = int(self.numEpisodes))

        for episode_object in self.allEpisodes:
            self.newAllEpisodes.append(copy.deepcopy(episode_object))

        for episode_object in buffer.allEpisodes:
            self.newAllEpisodes.append(copy.deepcopy(episode_object))

        del self.allEpisodes
        self.allEpisodes = self.newAllEpisodes

    def __len__(self):
        return self.numEpisodes if self.full else self.idx

    def set_sample_settings(self, length = 10, shaped_rewards = False, trainprop = 1, train = True, correctionsOnly = False):
        self.length = length
        self.shaped_rewards = shaped_rewards
        self.trainprop = trainprop
        self.train = train
        self.correctionsOnly = correctionsOnly
        print(f"sample length: {length}")


    def add(self, episode_list, priority = None):
        episode_object = SingleEpisode(len(episode_list), self.lowdim_shape, self.obs_shape, self.audio_shape, self.action_shape, self.image_pad, self.device)
        print("this demo is length ", len(episode_list))
        episode_object.add(episode_list)
        if priority is not None:
            episode_object.setPriority(priority)
        self.allEpisodes.append(episode_object)

        self.idx = (self.idx + 1) % self.numEpisodes
        self.full = self.full or self.idx == 0


    def sampleRollout(self):
        if self.train:
            index = np.random.randint(0, self.trainprop * (self.numEpisodes if self.full else self.idx))
        else:
            print("TEST TEST TEST")
            index = np.random.randint(self.trainprop * (self.numEpisodes if self.full else self.idx),
                                     self.numEpisodes if self.full else self.idx)

        #really stupid hack but it works for now
        while self.correctionsOnly and len(self.allEpisodes[index].priorities) == 0:
            index = np.random.randint(0,
                                 self.trainprop * (self.numEpisodes if self.full else self.idx))

        lowdim, obses, audio, actions, rewards, next_lowdim, \
            next_obses, next_audio, not_dones_no_max = self.allEpisodes[index].sample_rollout(self.length,
                                                                                              self.shaped_rewards, correctionsOnly = self.correctionsOnly)

        return lowdim, obses, audio, actions, rewards, next_lowdim, next_obses, next_audio, not_dones_no_max

    def __iter__(self):
        while True:
            yield self.sampleRollout()

