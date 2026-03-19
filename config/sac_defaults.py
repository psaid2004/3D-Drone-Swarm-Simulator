# -*- utf-8 -*-

# ---- SAC HYPERPARAMETERS ----
GAMMA           = 0.99
BATCH_SIZE      = 256
BUFFER_SIZE     = 300_000
LEARNING_STARTS = 100         # random exploration before training begins
TRAIN_FREQ      = 1           # env steps between gradient updates
GRADIENT_STEPS  = 1           # gradient updates per train_freq
LEARNING_RATE   = 3e-4
ENTROPY_COEF    = "auto"

NET_ARCH = dict(pi=[512, 256, 128], qf=[512, 256, 128])

# For debugging -> set to 1 environment
N_ENVS = 1