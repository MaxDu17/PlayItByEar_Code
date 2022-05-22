# Play It ByEar Code
 Code used for our 2022 RSS paper, *Play it by Ear: Learning Skills amidst Occlusion through Audio-Visual Imitation Learning*

# Installing Dependencies
1. Run `conda env create -f conda_env.yml`. 
2. Install system-compatible versions of cudatoolkit, pytorch, torchvision, and torchaudio

# Simulation Demo Collection
The demo collection program is `writeDemos_episodes.py` with a `.yaml` configuration under the same name. 

The `environmentName` parameter sets the environment. Pick between `BlockedPickPlace` (occluded pick-place task) and `IndicatorBoxBlock` (occluded lift task).

Set the `demo_root` parameter to demo directory. Other parameters to change include `episodeLength` and `episodes`. After running, the code will save the demos to a `.pkl` file that will be loaded by training code.g

# Simulation Training
The imitation learning code is `imitationtrain_memory.py` with a `.yaml` configuration under the same name. 

Like in the demo collection, the `environmentName` parameter sets the environment. Pick between `BlockedPickPlace` (occluded pick-place task) and `IndicatorBoxBlock` (occluded lift task).

Set both `demo_root` and `actor_root` to be the demo directory and the results directory, respectively. They should be outer directories, i.e. if your demo is saved in `demos/IndicatorBoxBlock_50/demo.pkl`, the `demo_root` should be `demos/`.  

The same code can also accomodate balanced batch training. Use the `balanced_batches` flag and modify the `corrections_file` to reference the replay buffer of corrections. If `priority` is enabled, we will only sample active corrections from the corrections buffer. Otherwise, we will sample uniformly from the whole episode.
# Simulation Intervention
The intervention code is `sim_intervention_episodes.py` with a `.yaml` configuration under the same name. 

Like in simulation, set both `demo_root` and `actor_root` to be the demo directory and the results directory, respectively. Set the desired environment. 

Set the `orig_seed` to the seed of the imitation-trained model, and set the `checkpoint` to the saved model version. The locations of the files should be automatically found. 


# Structure of this repository
The `core/` directory contains all the code for the model we used, as well as some critical utility functions and replay buffer implementations. 

The `custom_environments/` directory contains the code and `.xml` models for the Robosuite environments we used.

All the code that can be run directly is in the main folder. 

# Acknowledgements
This codebase is adapted from *Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels* (https://github.com/denisyarats/drq)

