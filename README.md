# Play It ByEar Code
 Code used for our 2022 RSS paper, *Play it by Ear: Learning Skills amidst Occlusion through Audio-Visual Imitation Learning*

# Structure of this repository
The `core/` directory contains all the code for the model we used, as well as some critical utility functions and replay buffer implementations. 

The `custom_environments/` directory contains the code and `.xml` models for the Robosuite environments we used. 

The `robot_experiments/` directory contains the code we used to run imitation learning and corrections on the Franka-Emika Panda robot. It does not contain the Franka Server or the Oculus code. 

The `simulation_experiments/` directory contains the code we used to run imitaiton learning and corrections on the Robosuite simulator. This also includes a residual training baseline. 

The `utilities/` directory contains the code that we used to collect demonstrations. 

# Acknowledgements
This codebase is adapted from *Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels* (https://github.com/denisyarats/drq)

