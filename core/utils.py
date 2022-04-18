import math
import os
import random
from collections import deque

import numpy as np
import scipy.linalg as sp_la

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.util.shape import view_as_windows
from torch import distributions as pyd


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

#extracting data from vector 
def obsOnly(obs, cfg): #only works with non-object observation
    return obs[0:cfg.agent.params.lowdim_dim]

def obsAndImage(obs, cfg, non_stacked_size = None):
    if non_stacked_size == None:
        image = np.reshape(obs[cfg.agent.params.lowdim_dim:], (cfg.image_size, cfg.image_size, 3))
        ending = cfg.agent.params.lowdim_dim
    else:
        non_stacked_size = int(non_stacked_size)#cast to prevent indexing error 
        image = np.reshape(obs[non_stacked_size:], (cfg.image_size, cfg.image_size, 3))
        ending = non_stacked_size
    
    assert np.shape(image) == (cfg.image_size, cfg.image_size, 3)
    image = image.astype(np.uint8)
    image = np.flipud(image) - np.zeros_like(image)
    
    return obs[0:ending], image

#this is for the inputs without stacking 
class FrameStack_Lowdim(gym.Wrapper):
    def __init__(self, env, cfg, k, l_k, frameMode = 'cat', demo = False, audio = False):
        self.frameMode = frameMode
        if frameMode == 'cat':
            #in concatenation mode, lowdim_dim should be the dim * stacks
            #in stack mode, lowdim_dim is just the lowdim
            assert cfg.agent.params.lowdim_dim % l_k == 0
            self.non_stacked_space = cfg.agent.params.lowdim_dim / l_k
            shp = (3, cfg.image_size, cfg.image_size)
            self.lowdim_space = [1, cfg.agent.params.lowdim_dim] #again, remember that lowdim is k * lowdim
        elif frameMode == 'stack':
            assert l_k == k, "in this mode, the stacks must be of equal size"
            shp = (1, 3, cfg.image_size, cfg.image_size)
            self.lowdim_space = [k, 1, cfg.agent.params.lowdim_dim] #remember that lowdim is unmodified 
#             input(cfg.agent.params.lowdim_dim)
        else:
            raise Exception("invalid mode! Must be either 'cat' or 'stack'")
            
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._l_k = l_k
        self._frames = deque([], maxlen=k)
        self._ldframes = deque([], maxlen=l_k)
        self._audframes = deque([], maxlen=l_k)
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype)
        
        self._max_episode_steps = cfg.horizon
        self.cfg = cfg
        self.mode = self.cfg.system
        assert self.mode == "sim" or self.mode == "real"
        
        self.demo = demo
        self.audio = audio
        assert not(demo) or self.mode == 'sim', "demo reset only works in sim"
        
        #self._max_episode_steps = env._max_episode_steps
    def reset(self):
        if self.demo:
            print("mode: demo")
            ob_dict, obs_raw = self.env.demo_reset()
            if self.frameMode == 'cat': 
                lowdim, image = obsAndImage(obs_raw, self.cfg, self.non_stacked_space)
            else:
                lowdim, image = obsAndImage(obs_raw, self.cfg)
            image = np.transpose(image, (2, 0, 1))
            for _ in range(self._k):
                self._frames.append(image)
            for _ in range(self._l_k):
                self._ldframes.append(lowdim)
            return ob_dict, self._get_low(), self._get_obs()
        
        if self.audio and self.frameMode == 'cat':
            print("mode: audio and concatenate (single audio)")
            assert self.mode == 'real', "audio doesn't exist in sim!"
            obs_raw = self.env.reset()
            lowdim, image, audio = obs_raw
    #         print(np.shape(image))
            image = np.transpose(image, (2, 0, 1))
            for _ in range(self._k):
                self._frames.append(image.copy())
            for _ in range(self._l_k):
                self._ldframes.append(lowdim.copy())
            return  self._get_low(), self._get_obs(), audio #quirk with the audio; automatically assume that we only have one copy 

    def step(self, action):
        if self.demo:
#             print("mode: demo")
            ob_dict, obs_raw, reward, done, info  = self.env.demo_step(action)
            if self.frameMode == 'cat': 
                lowdim, image = obsAndImage(obs_raw, self.cfg, self.non_stacked_space)
            else:
                lowdim, image = obsAndImage(obs_raw, self.cfg)
#             lowdim, image = obsAndImage(obs_raw, self.cfg)
            image = np.transpose(image, (2, 0, 1))
            self._frames.append(image.copy())
            self._ldframes.append(lowdim.copy())
            return ob_dict, self._get_low(), self._get_obs(), reward, done, info
        
        if self.audio and self.frameMode == 'cat':
#             print("mode: audio and concatenate (single audio)")
            assert self.mode == 'real', "audio doesn't exist in sim!"
            obs_raw, reward, done, info = self.env.step(action)

            lowdim, image, audio = obs_raw
            image = np.transpose(image, (2, 0, 1)) 
            self._frames.append(image.copy())
            self._ldframes.append(lowdim.copy())
            return self._get_low(), self._get_obs(), audio, reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        if self.frameMode == 'cat':
            return np.concatenate(list(self._frames), axis=0)
        return np.stack(list(self._frames), axis=0)
    
    def _get_low(self):
        assert len(self._ldframes) == self._l_k
        if self.frameMode == 'cat':
            return np.concatenate(list(self._ldframes), axis=0)
        return np.expand_dims(np.stack(list(self._ldframes), axis=0), axis = 1)
    
    def _get_audio(self):
        assert len(self._audframes) == self._k
        assert self.frameMode == 'stack', "in cat mode, audio is singular!"
        aud_frames = np.stack(list(self._audframes), axis=0)  
#         print("ABLATE AUDIO")
#         aud_frames = np.zeros_like(aud_frames)
        return aud_frames 

#this is a special class that allows history AND stacking
class FrameStack_StackCat(gym.Wrapper):
    def __init__(self, env, cfg, k, l_k, stack_depth, demo = False, audio = False):
        shp = (1, 3 * k, cfg.image_size, cfg.image_size)
        self.lowdim_space = [stack_depth, 1, l_k * cfg.agent.params.lowdim_dim] #remember that lowdim is unmodified 
#             input(cfg.agent.params.lowdim_dim)
            
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._l_k = l_k
        self._auxframes = deque([], maxlen = k) #the idea is that we keep two RB: one that deals with the new information, and another that stores that RB
        self._frames = deque([], maxlen=stack_depth)
        self._ldframes = deque([], maxlen=stack_depth)
        self._auxldframes = deque([], maxlen=l_k)
        
        self._audframes = deque([], maxlen=stack_depth) #no cat stack is done for audio , used to be l_k
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * stack_depth,) + shp[1:]),
            dtype=env.observation_space.dtype)
        
        self._max_episode_steps = cfg.horizon
        self.cfg = cfg
        self.mode = self.cfg.system
        assert self.mode == "sim" or self.mode == "real"
        
        self.demo = demo
        self.audio = audio
        self.stack_depth = stack_depth 
        assert not(demo) or self.mode == 'sim', "demo reset only works in sim"
        self.non_stacked_space = cfg.agent.params.lowdim_dim / l_k
        #self._max_episode_steps = env._max_episode_steps
        
    def reset(self):
        if self.demo:
            print("mode: demo")
            ob_dict, obs_raw = self.env.demo_reset() 
            lowdim, image = obsAndImage(obs_raw, self.cfg, self.non_stacked_space)

            image = np.transpose(image, (2, 0, 1))
            for _ in range(self._k): #filling up the auxilary buffers 
                self._auxframes.append(image)
            for _ in range(self._l_k):
                self._auxldframes.append(lowdim)
            for _ in range(self.stack_depth): 
                self._frames.append(np.concatenate(list(self._auxframes.copy()))) #using these buffers to fill up the main stack buffer 
                self._ldframes.append(np.concatenate(list(self._auxldframes.copy())))
                
            return ob_dict, self._get_low(), self._get_obs()
       
        elif self.audio: 
            assert self.mode == 'real', "audio doesn't exist in sim!"
            print("mode: audio")
            obs_raw = self.env.reset()
            lowdim, image, audio = obs_raw    
            image = np.transpose(image, (2, 0, 1))
            for _ in range(self._k): #filling up the auxilary buffers 
                self._auxframes.append(image)
            for _ in range(self._l_k):
                self._auxldframes.append(lowdim)
            for _ in range(self.stack_depth): 
                self._frames.append(np.concatenate(list(self._auxframes.copy()))) #using these buffers to fill up the main stack buffer 
                self._ldframes.append(np.concatenate(list(self._auxldframes.copy())))
                self._audframes.append(audio.copy())
            return  self._get_low(), self._get_obs(), self._get_audio()
        else:
            print("mode: vanilla mode")
            obs_raw = self.env.reset()
            if self.mode == "sim":
                lowdim, image = obsAndImage(obs_raw, self.cfg, self.non_stacked_space)
            else:
                lowdim, image, audio = obs_raw
            image = np.transpose(image, (2, 0, 1))
            for _ in range(self._k): #filling up the auxilary buffers 
                self._auxframes.append(image)
            for _ in range(self._l_k):
                self._auxldframes.append(lowdim)
            for _ in range(self.stack_depth): 
                self._frames.append(np.concatenate(list(self._auxframes.copy()))) #using these buffers to fill up the main stack buffer 
                self._ldframes.append(np.concatenate(list(self._auxldframes.copy())))
            return self._get_low(), self._get_obs()

    def step(self, action):
        if self.demo:
            ob_dict, obs_raw, reward, done, info  = self.env.demo_step(action) 
            lowdim, image = obsAndImage(obs_raw, self.cfg, self.non_stacked_space)

#             lowdim, image = obsAndImage(obs_raw, self.cfg)
            image = np.transpose(image, (2, 0, 1))
            self._auxframes.append(image.copy())
            self._auxldframes.append(lowdim.copy())
            self._frames.append(np.concatenate(list(self._auxframes.copy()))) #using these buffers to fill up the main stack buffer 
            self._ldframes.append(np.concatenate(list(self._auxldframes.copy())))
            return ob_dict, self._get_low(), self._get_obs(), reward, done, info

        elif self.audio:
            assert self.mode == 'real', "audio doesn't exist in sim!"
#             print("mode: audio and stack")
            obs_raw, reward, done, info = self.env.step(action)
            lowdim, image, audio = obs_raw

            image = np.transpose(image, (2, 0, 1)) 
            self._auxframes.append(image.copy())
            self._auxldframes.append(lowdim.copy())
            self._frames.append(np.concatenate(list(self._auxframes.copy()))) #using these buffers to fill up the main stack buffer 
            self._ldframes.append(np.concatenate(list(self._auxldframes.copy())))
            
            self._audframes.append(audio.copy())
            return self._get_low(), self._get_obs(), self._get_audio(), reward, done, info
        else:
#             print("vanilla step")
            obs_raw, reward, done, info = self.env.step(action)
            if self.mode == "sim":
                lowdim, image = obsAndImage(obs_raw, self.cfg, self.non_stacked_space)
            else:
                lowdim, image, audio = obs_raw

            image = np.transpose(image, (2, 0, 1)) 
            
            
            self._auxframes.append(image.copy())
            self._auxldframes.append(lowdim.copy())
            self._frames.append(np.concatenate(list(self._auxframes.copy()))) #using these buffers to fill up the main stack buffer 
            self._ldframes.append(np.concatenate(list(self._auxldframes.copy())))
            return self._get_low(), self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self.stack_depth 
        frames = np.stack(list(self._frames), axis=0)
        
        ####### ablating history #######
#         print("ABLATE HISTORY")
#         frames[:, : -3, :, :] = 0


        ####### ablating video #######
#         print("ABLATE VIDEO")
#         frames = np.zeros_like(frames)
        
        return frames
    
    def _get_low(self):
        assert len(self._ldframes) == self.stack_depth 
        lowdim_frames = np.expand_dims(np.stack(list(self._ldframes), axis=0), axis = 1)
        #### ablating audio ###
#         print("ABLATE AUDIO")
#         lowdim_frames = np.zeros_like(lowdim_frames)
#         for j in range(10):
#             for i in range(10):
#                 lowdim_frames[j, :, 13 * i : 13 * i + 6] = 0
        #### ABLATE AUDIO ##### 
        
        ####### ablating history #######
#         print("ABLATE HISTORY")
#         lowdim_frames[:, :, :-7] = 0
        
        return lowdim_frames 
    
    def _get_audio(self):
        assert len(self._audframes) == self.stack_depth 
        aud_frames = np.stack(list(self._audframes), axis=0)  
#         print("ABLATE AUDIO")
#         aud_frames = np.zeros_like(aud_frames)
        return aud_frames
    
class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)
        self.transforms = [TanhTransform()]
        super().__init__(self.base_dist, self.transforms)
    
    def log_prob(self, values): #THIS IS PROBLEMATIC
        l_prob = super().log_prob(values)
        if torch.isnan(l_prob).any():
            return super().log_prob(torch.tanh(values))
        return l_prob
#         return super().log_prob(torch.tanh(values))
    
    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu