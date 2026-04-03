# -*- coding: utf-8 -*-
"""
This file defines defaults for the parameters of SAC.
"""

# ---- SAC HYPERPARAMETERS ----
GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_SIZE = 300000
LEARNING_STARTS = 2000  # random exploration before training begins
TRAIN_FREQ = 1  # env steps between gradient updates
GRADIENT_STEPS = 1  # gradient updates per train_freq
LEARNING_RATE = 3e-4
ENTROPY_COEF = "auto"

NET_ARCH = dict(pi=[512, 256, 128], qf=[512, 256, 128])

# For debugging -> set to 1 environment
N_ENVS = 1

# ---- EXPERT CONFIG ----
INITIAL_BLEND = 0.8  # expert weight at start
FINAL_BLEND = 0.0  # pure RL at end
BLEND_DECAY_FRAC = 0.20  # fraction of total timesteps to decay over # TODO I MADE THIS LESS BUT DEPENDS ON CURRICULUM SO ONCE THAT IS SET, I CAN FIGURE OUT WHAT THIS WILL BE
