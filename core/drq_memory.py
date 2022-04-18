# this code is for the simulation experiments

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import time
import kornia
import math

import core.utils as utils
import hydra

from torch import distributions as pyd
import torchvision.transforms as T
import torchvision.transforms.functional as T_f

import torchvision.models as models


class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, feature_dim, lowdim_dim, num_layers, num_filters, output_dim, output_logits):
        super().__init__()
        assert len(obs_shape) == 3
        self.output_dim = output_dim
        self.output_logits = output_logits
        self.lowdim = lowdim_dim
        self.feature_dim = feature_dim

        #resnet structure
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 =  nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet.fc = nn.Identity()
        self.convs = resnet
        for param in self.convs.parameters():
            param.requires_grad = True
        self.head = nn.Sequential(
            nn.Linear(512 + self.lowdim, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.outputs = dict()


    def forward_conv(self, obs):
        obs = T_f.resize(obs, 224)
        self.outputs['obs'] = obs
        conv = self.convs(obs)
        h = conv.reshape(conv.size(0), -1)
        return h

    def train(self, train = True):
        if train:
            self.convs.train()
            self.head.train()
        else:
            self.convs.eval()
            self.head.eval()

    def forward(self, lowdim, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()

        combined_states = torch.cat([h, lowdim], dim=-1) #add lowdims here
        out = self.head(combined_states)
        if not self.output_logits:
            out = torch.tanh(out)

        self.outputs['out'] = out

        return out

    def copy_conv_weights_from(self, source):
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])

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

        self.trunk = utils.mlp(self.encoder.feature_dim, hidden_dim,
                   2 * action_shape[0], hidden_depth)
        self.memory_cells = nn.LSTM(self.encoder.feature_dim,self.encoder.feature_dim, batch_first = True )

        #LSTM expects batch X length x features
        self.outputs = dict()
        self.apply(utils.weight_init)

    def train(self, train = True):
        self.memory_cells.train(train)
        self.trunk.train(train)
        self.encoder.train(train)

    def forward(self, lowdim, obs, detach_encoder = False, squashed = True):
        # squashed is a remnant from AWAC; ignore this
        assert lowdim.shape[0] == obs.shape[0] #batch size
        assert lowdim.shape[1] == obs.shape[1] #sequence leng
        batch_size = lowdim.shape[0]
        sequence_length = lowdim.shape[1]
        combined = lowdim.shape[0] * lowdim.shape[1]
        obs = self.encoder(lowdim.reshape(combined, lowdim.shape[2], lowdim.shape[3]), obs.reshape(combined, obs.shape[2],  obs.shape[3],  obs.shape[4]), detach=detach_encoder)
        encoded = obs.reshape(batch_size, sequence_length, -1) #reshaping back into batches of runs


        assert encoded.shape[0] == batch_size
        assert encoded.shape[1] == sequence_length
        self.outputs['before_encoding'] = encoded
        input(encoded.shape)
        _, (encoded, c) = self.memory_cells(encoded)
        input(encoded.shape)
        input("here!")
        self.outputs['encoded'] = encoded
        encoded = encoded.reshape(batch_size, -1)
        #################################

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

#### THE FOLLOWING IS FOR THE RESIDUAL BASELINE ######
class ActorResidual(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()
        self.encoder = hydra.utils.instantiate(encoder_cfg)

        self.log_std_bounds = log_std_bounds
        #the 6 comes from the reduced length after convolving, and the 16 comes from the CNN stack
        self.trunk = utils.mlp(self.encoder.feature_dim + action_shape[0], hidden_dim,
                       2 * action_shape[0], hidden_depth)
        self.memory_cells = nn.LSTM(self.encoder.feature_dim,self.encoder.feature_dim, batch_first = True )

        #1d conv take in batch X features X length
        #LSTM expects batch X length x features

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, lowdim, obs, action, detach_encoder = False, squashed = True):
        assert squashed == True, "resid is not equipped for AWAC"
        assert lowdim.shape[0] == obs.shape[0] #batch size
        assert lowdim.shape[1] == obs.shape[1] #sequence leng
        batch_size = lowdim.shape[0]
        sequence_length = lowdim.shape[1]
        combined = lowdim.shape[0] * lowdim.shape[1]
        obs = self.encoder(lowdim.reshape(combined, lowdim.shape[2], lowdim.shape[3]), obs.reshape(combined, obs.shape[2],  obs.shape[3],  obs.shape[4]), detach=detach_encoder)
        encoded = obs.reshape(batch_size, sequence_length, -1) #reshaping back into batches of runs
        assert encoded.shape[0] == batch_size
        assert encoded.shape[1] == sequence_length
        self.outputs['before_encoding'] = encoded
        _, (encoded, c) = self.memory_cells(encoded)
        self.outputs['encoded'] = encoded
        encoded = encoded.reshape(batch_size, -1)

        if action.shape[0] != batch_size:#really janky like all of this residual thing
            action = action.unsqueeze(0)
            #add a "batch" of 1`if we are running real-time

        encoded = torch.cat([encoded, action], dim = -1)

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

class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth):
        super().__init__()
        self.encoder = hydra.utils.instantiate(encoder_cfg)
        self.outputs = dict()
        self.Q1 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                hidden_dim, 1, hidden_depth)

        self.memory_cells = nn.LSTM(self.encoder.feature_dim,self.encoder.feature_dim, batch_first = True )
        self.apply(utils.weight_init)


    def forward(self, lowdim, obs, action, detach_encoder=False):
        assert lowdim.shape[0] == obs.shape[0] #batch size
        assert lowdim.shape[1] == obs.shape[1] #sequence leng
        batch_size = lowdim.shape[0]
        sequence_length = lowdim.shape[1]
        combined = lowdim.shape[0] * lowdim.shape[1]
        obs = self.encoder(lowdim.reshape(combined, lowdim.shape[2], lowdim.shape[3]), obs.reshape(combined, obs.shape[2],  obs.shape[3],  obs.shape[4]), detach=detach_encoder)
        encoded = obs.reshape(batch_size, sequence_length, -1) #reshaping back into batches of runs
        assert encoded.shape[0] == batch_size
        assert encoded.shape[1] == sequence_length
        self.outputs['before_encoding'] = encoded
        _, (encoded, c) = self.memory_cells(encoded)
        self.outputs['encoded'] = encoded
        encoded = encoded.reshape(batch_size, -1)
        obs_action = torch.cat([encoded, action], dim=-1) #add lowdims here
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        self.outputs['q1'] = q1
        self.outputs['q2'] = q2
        return q1, q2

    def log(self, logger, step):
        self.encoder.log(logger, step)
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)
        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)

    def save(self, step, work_dir):
        torch.save(self.Q1.state_dict(), work_dir + "/" + str(step)+ "_Q1.pt")
        torch.save(self.Q1.state_dict(), work_dir + "/" + str(step)+ "_Q2.pt")
        torch.save(self.encoder.state_dict(), work_dir + "/" + str(step)+ "_critic_encoder.pt")
        torch.save(self.memory_cells.state_dict(), work_dir + "/" + str(step) + "_critic_memory_cells.pt")

    def load(self, cwd):
        self.Q1.load_state_dict(torch.load(cwd+"Q1.pt"))
        print("loaded {0}".format(cwd + "Q1.pt"))
        self.Q2.load_state_dict(torch.load(cwd+"Q2.pt"))
        print("loaded {0}".format(cwd + "Q2.pt"))
        self.encoder.load_state_dict(torch.load(cwd+"critic_encoder.pt"))
        print("loaded {0}".format(cwd+"critic_encoder.pt"))
        self.memory_cells.load_state_dict(torch.load(cwd + "critic_memory_cells.pt"))
        print("\tloaded {0}".format(cwd + "critic_memory_cells.pt"))

class DRQAgent(object):
    """Data regularized Q: actor-critic method for learning from pixels."""
    def __init__(self, obs_shape, action_shape, action_range, device,
                 encoder_cfg, critic_cfg, actor_cfg, discount,
                 init_temperature, lr, actor_update_frequency, critic_tau,
                 critic_target_update_frequency, batch_size, lowdim_dim, log_frequency):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.lowdim_dim = lowdim_dim
        self.log_frequency = log_frequency

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)
        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie conv layers between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.train()
        self.critic_target.train()

        #the 4 is just hardcoded
        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(4),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])) ,
            kornia.augmentation.ColorJitter(brightness=.05, hue=.05),
            kornia.augmentation.RandomAffine(degrees=5, translate = (0.05, 0.05), scale = (0.95, 1.05))
        )

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def save(self, step, work_dir):
        self.critic.save(step, work_dir)
        self.actor.save(step, work_dir)

    def load(self, cwd, load_critic = False, prefix = None):
        print("loading models from saved files!")
        if prefix is not None:
            cwd = prefix + cwd
        if load_critic:
            self.critic.load(cwd)
        self.actor.load(cwd)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, lowdim, obs, sample=False, squash = True, base_action = None, batch = False):
        #if we are in batch, this is done for us
        if not batch:
            obs = torch.FloatTensor(obs).to(self.device)
            lowdim = torch.FloatTensor(lowdim).to(self.device)
            obs = obs.unsqueeze(0)
            lowdim = lowdim.unsqueeze(0)

        if base_action is not None: #residual
            base_action = torch.as_tensor(base_action, device = self.device)
            dist = self.actor(lowdim, obs, base_action, squashed = squash) #happens only when we are in residual mode; weird polymorphism
        else:
            dist = self.actor(lowdim, obs, squashed = squash)

        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        if not batch:
            assert action.ndim == 2 and action.shape[0] == 1
            return utils.to_np(action[0])
        else:
            return action #keep in torch tensor if we're doing batch training

    def update_critic(self, lowdim, lowdim_aug, obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, next_lowdim, next_lowdim_aug, not_done, logger, step, detach_encoder = False, squash = True, base_agent = None):

        with torch.no_grad():
            if base_agent is not None:
                next_base_action = base_agent.act(next_lowdim, next_obs, sample = False, batch = True)
                dist = self.actor(next_lowdim, next_obs, next_base_action, squashed = squash)
                next_base_action_aug = base_agent.act(next_lowdim_aug, next_obs_aug, sample = False, batch = True)
                dist_aug = self.actor(next_lowdim_aug, next_obs_aug, next_base_action_aug, squashed = squash) #do the same thing on the augmented things
            else:
                dist_aug = self.actor(next_lowdim_aug, next_obs_aug, squashed = squash) #do the same thing on the augmented things
                dist = self.actor(next_lowdim, next_obs, squashed = squash)

            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True) #calculates the negative entropy
            target_Q1, target_Q2 = self.critic_target(next_lowdim, next_obs, next_action) #run Q target on one step in the future
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob #entropy-termed value function
            target_Q = reward + (not_done * self.discount * target_V) #bellman backup
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1,
                                                                  keepdim=True)

            target_Q1, target_Q2 = self.critic_target(next_lowdim_aug, next_obs_aug,
                                                      next_action_aug)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + (not_done * self.discount * target_V)
            target_Q = (target_Q + target_Q_aug) / 2 #average the two bellmans

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(lowdim, obs, action, detach_encoder)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        Q1_aug, Q2_aug = self.critic(lowdim_aug, obs_aug, action, detach_encoder)
        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(Q2_aug, target_Q)

        if step % self.log_frequency == 0:
            logger.log('train_critic/loss', critic_loss, step)
            self.critic.log(logger, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    # for residual training
    def update_actor_and_alpha_bc_resid(self, lowdim, obs, base_actions, logger, step, expert_action,
                                  usebc, scale_BC, squash = True):  # combined BC and RL backprop
        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(lowdim, obs, base_actions, detach_encoder=False, squashed = squash)
        agent_action = dist.mean

        log_prob = dist.log_prob(agent_action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(lowdim, obs, agent_action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean() #smaller (more neg) is better

        if usebc: #add behavior cloning loss
            agent_mean = dist.mean #this is changed
            bc_loss = nn.MSELoss()
            scaled_bc = scale_BC * bc_loss(agent_mean, expert_action)
            logger.log('train_actor/scaled_bc', scaled_bc, step)
            logger.log('train_actor/rl_loss', actor_loss, step)
            actor_loss = scaled_bc + actor_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step() #only optimize actor steps

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if step % self.log_frequency == 0:
            logger.log('train_actor/summed_loss', actor_loss, step)
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            logger.log('train_actor/variances', dist.scale.mean(), step)
            logger.log('train_actor/target_entropy', self.target_entropy, step)
            logger.log('train_actor/entropy', -log_prob.mean(), step)

    def update_actor_bc(self, lowdim, obs, logger, step, action, squash = True): #only for pure BC
        dist = self.actor(lowdim, obs, detach_encoder=False, squashed = squash)
        agent_action = dist.mean
        loss = nn.MSELoss()
        action = action[:, -1, :] #pick last action to compare to
        actor_loss = loss(agent_action, action)

        with torch.no_grad():
            old_loss = loss(agent_action[0 : self.batch_size], action[0 : self.batch_size])
            new_loss = loss(agent_action[self.batch_size : ], action[self.batch_size : ])
            #will print "NAN" for new_loss if not on balanced batches; that's fine because this is only for logging purposes

        if step % 10 == 0:
            print("loss: ", actor_loss.item())
            print("old loss: ", old_loss.item())
            print("new loss: ", new_loss.item())
            print("step ----" , step, " ----step")

        if step % self.log_frequency == 0:
            logger.log('train_actor/loss', actor_loss, step)
            logger.log('train_actor/old_loss', old_loss, step)
            logger.log('train_actor/new_loss', new_loss, step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor.log(logger, step)


    def to_tensor(self, lowdim, obs, action, reward = None, done_no_max = None):
        lowdim = lowdim.to(self.device)
        obs = obs.to(self.device)
        action = action.to(self.device)
        if reward is not None:
            reward = reward.to(self.device) #torch.as_tensor(reward, device=self.device).float()
        if done_no_max is not None:
            done_no_max = done_no_max.to(self.device) #torch.as_tensor(done_no_max, device=self.device).float()
        return lowdim, obs, action, reward, done_no_max

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
#         input(new_obs.shape)
        new_obs = new_obs.view(batch, rollout, obs.shape[2], obs.shape[3], obs.shape[4]) #reshapes after aug
        return new_obs

    def update_bc(self, replay_buffer, logger, step, squash = True):
        t0 = time.time()
        lowdim, obs, action, reward, next_lowdim, next_obses, not_dones_no_max= next(replay_buffer)
        t1 = time.time()
        lowdim, obs, action, _, _ = self.to_tensor(lowdim, obs, action)
        t2 = time.time()
        obs = obs / 255. #we must scale
        obs = self.augment_observation(obs)
        t3 = time.time()
        self.update_actor_bc(lowdim, obs, logger, step, action, squash)
        t4 = time.time()
#         print(f"Total time {t4-t0}, Sample time {t1-t0}, Tensor Time {t2-t1}, Aug Time {t3-t2}, Update Time {t4-t3}")

    #if balanced batches are used
    def update_bc_balanced(self, base_buffer, intervention_buffer, logger, step, squash = True):
        lowdim1, obs1, action1, reward, next_lowdim, next_obses, not_dones_no_max = next(base_buffer)
        lowdim2, obs2, action2, reward, next_lowdim, next_obses, not_dones_no_max = next(intervention_buffer)
        lowdim1, obs1, action1, _, _ = self.to_tensor(lowdim1, obs1, action1)
        lowdim2, obs2, action2, _, _ = self.to_tensor(lowdim2, obs2, action2)
        obs1 = obs1 / 255.
        obs2 = obs2 / 255.
        obs1 = self.augment_observation(obs1)
        obs2 = self.augment_observation(obs2)
        lowdim = torch.cat([lowdim1, lowdim2], axis = 0)
        obs = torch.cat([obs1, obs2], axis = 0)
        action = torch.cat([action1, action2], axis = 0)
        self.update_actor_bc(lowdim, obs, logger, step, action, squash)

    def update_resid(self, replay_buffer, logger, step, cfg, base_policy, expert_buffer = None, freeze_all_encoder = False):
        assert expert_buffer is not None, "I need expert demonstrations!"
        buf = replay_buffer
        usebc = False

        if expert_buffer is not None and np.random.rand() > 0.5: #select from expert half of the time
            buf = expert_buffer
            usebc = True

        lowdim, obs, action, reward, next_lowdim, next_obs, not_done = next(buf)
        lowdim, obs, action, reward, not_done = self.to_tensor(lowdim, obs, action, reward, not_done)
        next_obs = next_obs.to(self.device)
        next_lowdim = next_lowdim.to(self.device)
        obs = obs / 255.
        next_obs = next_obs / 255.
        obs = self.augment_observation(obs)
        obs_aug = self.augment_observation(obs.clone())
        next_obs_aug = self.augment_observation(next_obs)

        #take last frame
        action = action[:, -1, :]
        reward = reward[:, -1, :]
        not_done = not_done[:, -1, :]

        if step % self.log_frequency == 0:
            logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(lowdim, lowdim.clone(), obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, next_lowdim, next_lowdim.clone(), not_done, logger, step, freeze_all_encoder, squash = cfg.use_squashed, base_agent = base_policy)


        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        if step % self.actor_update_frequency == 0:
            #BC must be enabeled and we must be sampling from the correct buffer
            base_actions = base_policy.act(lowdim, obs, sample=False, batch = True)
            base_actions = torch.as_tensor(base_actions, device = self.device)
            self.update_actor_and_alpha_bc_resid(lowdim, obs, base_actions, logger, step, action,
                                  usebc, cfg.scale_BC, squash = cfg.use_squashed)


