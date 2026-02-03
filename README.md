# Gym_Wheelrunner

A custom OpenAI Gym environment for simulating a step-wheel machine running task. This environment implements a reinforcement learning challenge where an agent learns to control a wheeled robot navigating over pegs on a rotating wheel pattern.

## Overview

This project provides two custom Gym environments that simulate a step-wheel locomotion task:
- **CustomEnv**: Standard environment with torque-based control
- **CustomEnv_leaky**: Environment with velocity decay, requiring continuous energy input

The goal is to train an agent to successfully run 200 steps by controlling the angular velocity of left and right wheels, timing their contact with pegs on a rotating pattern.

## Features

- **Custom Gym Environment**: Fully compatible with OpenAI Gym interface
- **Reinforcement Learning**: Uses Recurrent PPO (LSTM-based policy) from Stable Baselines3
- **Visual Rendering**: Real-time visualization of the wheel positions and peg locations
- **Complex Peg Patterns**: Configurable peg patterns for varying difficulty
- **Dual Control**: Independent control of left and right wheels
- **Anticipation System**: Whisker-based detection system that detects pegs before arrival

## Environment Details

### Observation Space

The observation space consists of 9 continuous values:
- Left and right wheel angles (normalized)
- Left and right angular velocities (normalized)
- Left and right upcoming peg timing (normalized, -1 if no peg detected)
- Left and right peg detection flags (0 or 1)
- Turn time within the pattern cycle (normalized)

### Action Space

The action space is discrete with 9 possible actions (3×3 grid):
- Each wheel can choose from: **TORQUE DOWN**, **NONE**, **TORQUE UP**
- Actions are applied independently to left and right wheels

### Reward Structure

- **Success**: +100 reward for completing 200 steps
- **Step Reward**: Incremental rewards for each successful step (10 + step_number × 5)
- **Speed Bonus**: Continuous small rewards for maintaining speed
- **Failure**: -5 reward for falling below minimum speed or missing pegs

### Success Criteria

- Complete 200 steps without:
  - Dropping below minimum angular velocity (2*pi rad/s)
  - Missing peg timing by more than +/-50ms
  - Attempting to step when no peg is detected

## Installation

### Prerequisites

- Python 3.7+
- OpenAI Gym
- Stable Baselines3
- SB3 Contrib (for RecurrentPPO)
- NumPy
- Matplotlib
- OpenCV (cv2)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/hirokame/Gym_Wheelrunner.git
cd Gym_Wheelrunner
```

2. Install dependencies:
```bash
pip install gym numpy matplotlib opencv-python
pip install stable-baselines3
pip install sb3-contrib
```

## Usage

### Training a Model

To train a new model using Recurrent PPO:

```bash
python train.py
```

Training parameters:
- Total timesteps: 300,000
- Evaluation frequency: Every 1,000 steps
- Model saves to: `./save_weights/`
- Logs save to: `./log/`

The training script will:
- Initialize the CustomEnv environment
- Create a RecurrentPPO model with LSTM policy
- Train with periodic evaluations
- Save the best model to `./save_weights/best_model.zip`

### Testing a Trained Model

To test a trained model:

```bash
python test.py
```

This will:
- Load the best model from `./save_weights/best_model.zip`
- Run 10 test episodes
- Display visual rendering of each episode
- Print the time taken for each attempt

### Environment Variants

#### Standard Environment (CustomEnv)
```python
from environment import CustomEnv

env = CustomEnv()
obs = env.reset()

# Run one step
action = env.action_space.sample()  # Random action
obs, reward, done, info = env.step(action)
```

#### Leaky Environment (CustomEnv_leaky)
```python
from leaky_environment import CustomEnv_leaky

env = CustomEnv_leaky()
obs = env.reset()

# This environment has velocity decay (0.98 multiplier per step)
# Requires active torque application to maintain speed
```

### Rendering

To visualize the environment during training or testing:

```python
env.popup()  # Create visualization window
env.render()  # Update the visualization
```

The visualization shows:
- Left and right wheels (circles)
- Current wheel angles (lines from center)
- Upcoming pegs (horizontal lines)
- Peg positions relative to wheel contact point

## Project Structure

```
Gym_Wheelrunner/
├── environment.py         # Standard gym environment implementation
├── leaky_environment.py   # Environment variant with velocity decay
├── train.py              # Training script with RecurrentPPO
├── test.py               # Testing script for trained models
├── env_check.ipynb       # Jupyter notebook for environment testing
├── log_check.ipynb       # Jupyter notebook for analyzing training logs
├── save_weights/         # Directory for saved models
├── log/                  # Directory for training logs
└── README.md            # This file
```

## Technical Details

### Peg Pattern

The default "Complex" pattern has 12 pegs per cycle (4000ms):
- **Left pegs** at: 0, 160, 400, 600, 900, 1300, 1660, 2060, 2360, 2760, 3200, 3600 ms
- **Right pegs** at: 100, 260, 500, 860, 1160, 1460, 1860, 2260, 2400, 2800, 3260, 3660 ms

### Control Loop

- **Time step**: 20ms (0.02 seconds)
- **Peg detection advance**: 500ms before arrival
- **Timing tolerance**: +/-50ms for successful peg contact

### Model Architecture

The trained model uses:
- **Algorithm**: Recurrent PPO (Proximal Policy Optimization)
- **Policy**: MlpLstmPolicy (LSTM-based policy network)
- **Framework**: Stable Baselines3 with SB3 Contrib

## License

See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

This environment simulates a bio-inspired locomotion task, inspired by wheel-running behaviors in biological systems.
