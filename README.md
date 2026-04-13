# Multi-Agent Collision Avoidance Module in Custom 3D Environment using SAC 

This repository contains the implementation of a decentralised Deep Multi-Agent Reinforcement Learning (MARL) system 
for autonomous drone swarm navigation. The project uses Soft-Actor Critic (SAC) to train agents to navigate a custom 3D
environment in order to get from a starting point to a pre-defined target, while avoiding both dynamic and static 
obstacles, the boundaries of the environment and other drones using only local observations. These 
observation are also enhanced using high fidelity sensors, namely LiDAR perception. 

## 🚀 Overview

The goal is to enable a swarm of drones (maximum 10 in a swarm, and possibly multiple swarms) to reach assigned targets
in a shared environment. Unlike centralised systems, each agent here makes decisions independently based on a 
float observation vector (including the optional LiDAR data).

### Key Features:

* **Decentralized Control:** Each drone runs its own policy instance.
* **Curriculum Learning:** Training is broken into phases (increasing drone/obstacle density/world scale) to stabilize learning.
* **Expert Blend Policy:** Training is initially sped-up through the use of an expert policy.
* **LiDAR Integration:** Support for simulated LiDAR for complex obstacle avoidance.
* **Built with:** `VelocityAviary`,`gym-pybullet-drones`,`Stable-Baselines3`, `PettingZoo`, `Gymnasium`, and `PyBullet`.

---

## 🛠 Installation

> ⚠️ Recommended: Python 3.10–3.11

1. **Clone the repository:**
```bash
git clone https://gitlab.utwente.nl/<your_username>/3D_Drone_Swarm_Simulator.git
cd 3D_Drone_Swarm_Simulator
```

2. **Create Virtual Environment:**
```bash
python -m venv venv
```

3. **Activate the virtual environment:**

**macOS / Linux**
```bash
source venv/bin/activate
```
**Windows (PowerShell)**
```bash
venv\Scripts\Activate.ps1
```
**Windows (Command Prompt)**
```bash
venv\Scripts\activate.bat
```

4. **Upgrade pip:**
```bash
python -m pip install --upgrade pip
```

5. **Install Dependencies:**
```bash
pip install -r requirements.txt
```
This will install `gym-pybullet-drones` from the GitHub repository specified in requirements.txt.


6. **(Optional) Set up Weights & Biases (wandb) for logging:**

If you want to track training metrics using Weights & Biases (only available in `/train/sac_training.py`):

```bash
pip install wandb
```
Then log in to your account:
```bash
wandb login
```

---
## ⚡ Quick Start 🏋️ Training

After installation, you can start training with:

```bash
python train/sac_training_with_expert.py
````

Or, if you prefer training without the use of the expert: 
```bash
python train/sac_training_with_expert.py
```

This class also contains additional features such as the use of an early-stopping callback, and
the integration of Weights and Biases to plot statistics to assist during training. 

Training needs to undergo further hyperparameter tuning and parameter selection. These can be found in 
`/config/sac_defualts.py`. Additionally, the reward function also needs to be altered for the 
sake of training an optimal model, which can be done from the `/config/defaults.py`. This 
class contains the coefficients that make up the reward function and determine the 
strength of each term.

*The model checkpoints will be saved in the `models/` directory.*

### Training Phases 

For an overview of the different training stages: `/config/curriculum.py`
These increase in difficulty by increasing the number of drones within 
a swarm, increasing the number of dynamic/static obstacles and increasing the
size of the world.

---
## 📂 Project Structure

```
DroneSwarmSimulator/
│
├── config/ # Parameter configs and curriculum definitions
├── envs/ # Environment creation and testing scripts
├── interface/ # UI / interaction layer 
├── models/ # Saved models
├── train/ # Training scripts and pipelines
│ ├── callbacks/ # Custom training callbacks
│ ├── evaluation/ # Evaluation scripts
│ ├── wandb/ # Logging (Weights & Biases)
│
├── wrappers/ # Environment wrappers and reward/step logic
│
├── gym-pybullet-drones/ # External dependency
└── requirements.txt

```
---





---

## 📊 Evaluation & Results

To evaluate a trained model against the performance metrics (Collision-free rate, Goal completion rate):

1. **Run Evaluation Script:**
```bash
python train/evaluation/eval.py
```


2. **Visualize a Test Run:**
```bash
python train/evaluation/testmodel.py
```



### Metrics Tracked:

* **Success Rate:** Percentage of agents reaching the goal.
* **CFR (Collision-Free Rate):** Percentage of episodes with zero collisions.
* **Wall-Hit Episode Rate:** Count of wall collisions over all episodes.

