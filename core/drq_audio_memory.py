# This is the code behind the imitation learning for the robot

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import time
import math

import torchvision.transforms.functional as T_f
import torchvision.transforms as T

import torchvision.models as models


import core.utils as utils
import hydra

from torch import distributions as pyd

class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, feature_dim, lowdim_dim, audio_steps, audio_bins, audio_feature_dim, num_layers, num_filters, output_dim, output_logits):
        super().__init__()
        assert len(obs_shape) == 3
        self.audio_steps = audio_steps #how many timesteps
        self.audio_bins = audio_bins #how many bins
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.output_dim = output_dim
        self.output_logits = output_logits
        self.lowdim = lowdim_dim
        self.feature_dim = feature_dim

        #------ MEMORY CELL HERE -----------------
        self.audioConvs = nn.ModuleList([
            nn.Conv1d(self.audio_bins, 64, kernel_size = 7),
            nn.Conv1d(64, 32, kernel_size = 7),
            nn.Conv1d(32, 16, kernel_size = 7),
            nn.Conv1d(16, 8, kernel_size = 7)
        ])
        #-----------------------------------------

        resnet = models.resnet18(pretrained=False)
        resnet.conv1 =  nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet.fc = nn.Identity()
        self.convs = resnet
        for param in self.convs.parameters():
            param.requires_grad = True
        self.head = nn.Sequential(
            nn.Linear(512 + self.lowdim + 264, self.feature_dim),
            nn.LayerNorm(self.feature_dim)) #a bit hacky with the 10

        self.outputs = dict()

    def forward_conv(self, obs):
        assert self.encoder_type == "resnet", "resize shouldn't be here"
        obs = T_f.resize(obs, 224)
        self.outputs['obs'] = obs
        conv = self.convs(obs)
        h = conv.reshape(conv.size(0), -1)
        return h

    def audio_forward_conv(self, audio):
        audio = audio / 20 # reducing magnitude
        conv = torch.relu(self.audioConvs[0](audio))
        self.outputs['audioconv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.audioConvs[i](conv))
            self.outputs['audioconv%s' % (i + 1)] = conv
        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, lowdim, obs, audio, detach=False):
        if len(audio.shape) < 3: #expanding to fit the batch requirement
            torch.unsqueeze(audio, 0)
        #preliminary checks
        assert audio.shape[1] == self.audio_steps #batch x rollout x features
        assert audio.shape[2] == self.audio_bins #checking that we feed in the right format
        audio = audio.transpose(1, 2) #flip to batch x features x rollout

        #forward prop
        h = self.forward_conv(obs)
        h_aud = self.audio_forward_conv(audio)

        if detach:
            h = h.detach()
            h_aud = h_aud.detach()

        #a slightly hacky solutoin to solve dimensionality problem
        if(np.shape(lowdim)[0] > 1):
            lowdim = torch.squeeze(lowdim)

        combined_states = torch.cat([h, lowdim, h_aud], dim=-1) #add lowdims here

        out = self.head(combined_states)
        if not self.output_logits:
            out = torch.tanh(out)

        self.outputs['out'] = out
        return out

    def copy_conv_weights_from(self, source):
        pass #remnant from the drq codebase

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()
        self.encoder = hydra.utils.instantiate(encoder_cfg)
        self.log_std_bounds = log_std_bounds
        self.outputs = dict()

        self.trunk = utils.mlp(self.encoder.feature_dim, hidden_dim,
                   2 * action_shape[0], hidden_depth)
        self.memory_cells = nn.LSTM(self.encoder.feature_dim,self.encoder.feature_dim, batch_first = True )

        self.apply(utils.weight_init)

    def forward(self, lowdim, obs, audio, detach_encoder = False, squashed = True):
        # squashed is a remant from AWAC experiment; you can ignore
        # preliminary checks
        assert lowdim.shape[0] == obs.shape[0] #batch size
        assert lowdim.shape[0] == audio.shape[0] #batch size
        assert lowdim.shape[1] == obs.shape[1] #sequence leng
        assert lowdim.shape[1] == audio.shape[1] #sequence lengh

        batch_size = lowdim.shape[0]
        sequence_length = lowdim.shape[1]

        combined = lowdim.shape[0] * lowdim.shape[1]
        obs = self.encoder(lowdim.view(combined, lowdim.shape[2], lowdim.shape[3]), obs.view(combined, obs.shape[2],  obs.shape[3],  obs.shape[4]), audio.view(combined, audio.shape[2], audio.shape[3]), detach=detach_encoder)
        encoded = obs.reshape(batch_size, sequence_length, -1) #reshaping back into batches of runs


        assert encoded.shape[0] == batch_size
        assert encoded.shape[1] == sequence_length
        self.outputs['before_encoding'] = encoded
        _, (encoded, c) = self.memory_cells(encoded)
        self.outputs['encoded'] = encoded
        encoded = encoded.reshape(batch_size, -1)

        mu, log_std = self.trunk(encoded).chunk(2, dim=-1)
#             constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std
        dist = utils.SquashedNormal(mu, std)
        return dist


    def log(self, logger, step):
        self.encoder.log(logger, step)
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)

    def save(self, step, work_dir):
        torch.save(self.trunk.state_dict(), work_dir + "/" + str(step)+ "_actor_trunk.pt")
        torch.save(self.encoder.state_dict(), work_dir + "/" + str(step)+ "_actor_encoder.pt")
        torch.save(self.memory_cells.state_dict(), work_dir + "/" + str(step) + "_actor_memory_cells.pt")

    def load(self, cwd):
        print("loading the trunk")
        self.trunk.load_state_dict(torch.load(cwd + "actor_trunk.pt"))
        print("\tloaded {0}".format(cwd + "actor_trunk.pt"))
        print("loading the encoder")
        self.encoder.load_state_dict(torch.load(cwd + "actor_encoder.pt"))
        print("\tloaded {0}".format(cwd + "actor_encoder.pt"))
        print("loading the memory cells")
        self.memory_cells.load_state_dict(torch.load(cwd + "actor_memory_cells.pt"))
        print("\tloaded {0}".format(cwd + "actor_memory_cells.pt"))

class DRQAgent(object):
    """Data regularized Q: actor-critic method for learning from pixels."""
    def __init__(self, obs_shape, action_shape, action_range, device,
                 encoder_cfg, critic_cfg, actor_cfg, discount,
                 init_temperature, lr, actor_update_frequency, critic_tau,
                 critic_target_update_frequency, batch_size, lowdim_dim, log_frequency):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.actor_update_frequency = actor_update_frequency
        self.batch_size = batch_size
        self.lowdim_dim = lowdim_dim
        self.log_frequency = log_frequency

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)


        # tie conv layers between actor and critic
        print("I am tying the encoders together!!!")
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)



        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.train()

                #the 4 is just hardcoded
        self.aug_trans = nn.Sequential(
            T.Resize(224),
            nn.ReplicationPad2d(4),
            T.RandomCrop((obs_shape[-1], obs_shape[-1])),
            T.ColorJitter(brightness=.1, hue=.1),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.95, 1.05))
        )

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

    def save(self, step, work_dir):
        self.actor.save(step, work_dir)


    def load(self, cwd, load_critic = False, prefix = None):
        # load_critic is a remnant from drq codebase
        print("loading models from saved files!")
        if prefix is not None:
            cwd = prefix + cwd
        self.actor.load(cwd)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # helper function
    def to_tensor(self, lowdim, obs, audio, action, reward = None, done_no_max = None):
        lowdim = torch.as_tensor(lowdim, device=self.device).float() #loading to memory
        obs = torch.as_tensor(obs, device=self.device).float()
        action = torch.as_tensor(action, device=self.device).float()
        audio = torch.as_tensor(audio, device=self.device).float()
        if reward is not None:
            reward = torch.as_tensor(reward, device=self.device).float()
        if done_no_max is not None:
            done_no_max = torch.as_tensor(done_no_max, device=self.device).float()
        return lowdim, obs, audio, action, reward, done_no_max

    # helper function
    def augment_observation(self, obs):
        assert len(obs.shape) == 5, "invalid function call"
        batch = obs.shape[0]
        rollout = obs.shape[1]
        new_obs = obs.view(batch * rollout, obs.shape[2], obs.shape[3], obs.shape[4])
        new_obs = torch.split(new_obs, 3, dim = 1)
        obs_list = list()
        for split in new_obs:
            obs_list.append(self.aug_trans(split))
        new_obs = torch.cat(obs_list, dim = 1)
        new_obs = new_obs.view(batch, rollout, obs.shape[2], obs.shape[3], obs.shape[4]) #reshapes after aug
        return new_obs

    def act(self, lowdim, obs, audio, sample=False, squash = True):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)

        lowdim = torch.FloatTensor(lowdim).to(self.device)
        lowdim = lowdim.unsqueeze(0)

        audio = torch.FloatTensor(audio).to(self.device)
        audio = audio.unsqueeze(0)

        dist = self.actor(lowdim, obs, audio, squashed = squash)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_actor_bc(self, lowdim, obs, audio, logger, step, action, squash = True): #only for pure BC
        beg = time.time()
        dist = self.actor(lowdim, obs, audio, detach_encoder=False, squashed = squash)
        agent_action = dist.mean
        loss = nn.MSELoss()

        action = action[:, -1, :] #pick last action to compare to

        actor_loss = loss(agent_action, action)
        logger.log('train_actor/loss', actor_loss, step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        beg = time.time()
        self.actor.log(logger, step)

    def update_bc(self, replay_buffer, logger, step, squash = True):
        lowdim, obs, audio, action, reward, next_lowdim, next_obses, next_audio, not_dones_no_max = next(replay_buffer)
        lowdim, obs, audio, action, _, _ = self.to_tensor(lowdim, obs, audio, action)
        obs /= 255
        obs = self.augment_observation(obs)
        self.update_actor_bc(lowdim, obs, audio, logger, step, action, squash)

    # for intervention training
    def update_bc_balanced(self, replay_buffer1, replay_buffer2, logger, step, squash = True):
        lowdim1, obs1, audio1, action1, reward, next_lowdim, next_obses, next_audio, not_dones_no_max = next(replay_buffer1)
        lowdim1, obs1, audio1, action1, _, _ = self.to_tensor(lowdim1, obs1, audio1, action1)
        lowdim2, obs2, audio2, action2, reward, next_lowdim, next_obses, next_audio, not_dones_no_max = next(replay_buffer2)
        lowdim2, obs2, audio2, action2, _, _ = self.to_tensor(lowdim2, obs2, audio2, action2)

        obs1 /= 255
        obs2 /= 255

        obs1 = self.augment_observation(obs1)
        obs2 = self.augment_observation(obs2)

        lowdim = torch.cat([lowdim1, lowdim2], axis = 0)
        obs = torch.cat([obs1, obs2], axis = 0)
        audio = torch.cat([audio1, audio2], axis = 0)
        action = torch.cat([action1, action2], axis = 0)

        self.update_actor_bc(lowdim, obs, audio, logger, step, action, squash)
