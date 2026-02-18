# Multi-Agent Collision Avoidance using PPO

This repository contains the implementation of a decentralized Deep Reinforcement Learning (DRL) system for autonomous drone navigation. Based on the research paper **"Multi-Agent Collision Avoidance Using PPO In Decentralized Reinforcement Learning"**, the project uses Proximal Policy Optimization (PPO) to train agents to navigate a 2D environment while avoiding obstacles and other drones using only local observations.

## 🚀 Overview

The goal is to enable multiple drones to reach assigned targets in a shared airspace. Unlike centralized systems, each agent here makes decisions independently based on a 16-40 float observation vector (including optional LiDAR data).

### Key Features:

* **Decentralized Control:** Each drone runs its own policy instance.
* **Curriculum Learning:** Training is broken into phases (increasing drone/obstacle density) to stabilize learning.
* **LiDAR Integration:** Support for 24-ray simulated LiDAR for complex obstacle avoidance.
* **Built with:** `Stable-Baselines3`, `PettingZoo`, `Gymnasium`, and `Pygame`.

---

## 🛠 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/drones_rl.git
cd drones_rl

```


2. **Install dependencies:**
```bash
pip install stable-baselines3 supersuit pettingzoo pygame imageio

```



---

## 📖 Theory to Implementation Mapping

The following table maps the mathematical and conceptual elements described in the research paper to the specific files and variables in this codebase.

| Research Concept | Implementation in Code | Description |
| --- | --- | --- |
| **PPO Algorithm** | `train/sb_ppo_training.py` | Configures `N_STEPS`, `GAMMA`, and `ENTROPY`. |
| **Policy Network Architecture** | `FIXED_NET_ARCH` in `sb_ppo_training.py` | Defines the MLP layers: `[512, 256, 128]`. |
| **Observation Space** | `DroneWorldEnv` & `pz_parallel.py` | Implementation of the 16-40 float state vector. |
| **LiDAR Perception** | `lidar=True` in `make_env()` | Adds 24 rays of 20m range to the observation. |
| **Reward Function** | `DroneWorldEnv.step()` | Logic for goal reaching (+), collisions (-), and time penalty. |
| **Curriculum Stages** | `staged_curriculums.py` | Defines the transitions between Phase 1 and Phase 2. |

---

## 🏋️ Training

The project uses a **Staged Curriculum** to train drones effectively.

To start training from Stage 1:

```bash
python train/sb3_sac_training.py

```

*The model checkpoints will be saved in the `models/` directory.*

### Training Phases (as per Paper Section 4)

* **Phase 1:** Single agent, increasing obstacles (0 to 10).
* **Phase 2:** Multi-agent (2+ drones), increasing obstacles and shrinking spawn areas.

---

## 📊 Evaluation & Results

To evaluate a trained model against the paper's metrics (Collision-free rate, Goal completion):

1. **Run Evaluation Script:**
```bash
python train/evaluation/eval.py

```


2. **Visualize a Test Run:**
```bash
python train/evaluation/test_run.py

```



### Metrics Tracked:

* **Success Rate:** Percentage of agents reaching the goal.
* **CFR (Collision-Free Rate):** Percentage of episodes with zero collisions.
* **Safety Violations:** Count of wall and agent-to-agent hits.

---

## 📂 Repository Structure

* `/train`: Core training logic and PPO hyperparameters.
* `/wrappers`: PettingZoo/Supersuit wrappers for multi-agent parallelization.
* `/evaluation`: Scripts for calculating metrics and generating playback videos.
* `staged_curriculums.py`: The "brain" of the training schedule.
* `repo_content.txt`: Bundled source code for analysis.

---
