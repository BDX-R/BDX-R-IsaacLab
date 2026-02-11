# BDX-R: A Journey in Bipedal Robotic Locomotion

This repository chronicles the development of BDX-R, a personal endeavor to create a bipedal robot inspired by Disney's BDX droids. The primary goal is to achieve stable walking and successfully bridge the simulation-to-reality (Sim2Real) gap using reinforcement learning.

| BDX-R in Isaac Lab Simulation | BDX-R Physical Prototype|
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/7b92c5b6-71ba-4746-a2d3-77d880e18014" width="500" /> | <img src="https://github.com/user-attachments/assets/4f65d9e9-85ad-497f-b687-10c54377d0f2" width="302" /> |

---

## 🎯 Current Focus: Walking and Sim2Real

The project is currently in its initial phase, with the core focus on mastering bipedal locomotion. The immediate objectives are:

-   **Achieve Stable Walking:** Train a robust walking policy using reinforcement learning.
-   **Cross the Sim2Real Gap:** Successfully transfer the policy trained in a simulated environment to the physical robot.

*At this stage, the project is concentrated on the fundamental mechanics of the body's movement. Expressiveness and the integration of a head are future goals to be explored after mastering stable locomotion.*

---

## 🛠️ Hardware

The BDX-R is built with a focus on high-performance components that are accessible to the robotics community. The entire build is being developed with a target budget under **$3,000**.

-   **Robstride Motors:** These motors provide the necessary torque and precision for dynamic and controlled leg movements.
-   **NVIDIA Jetson Orin Nano:** Serving as the onboard computer, the Jetson Orin Nano has the computational power required to run the trained RL policy in real-time.

---

## 🤖 Software and Training: Reinforcement Learning with Isaac Lab

The robot's ability to walk is being developed through reinforcement learning within the **NVIDIA Isaac Lab** simulation environment.

A policy is trained in this virtual space, allowing the BDX-R to learn and adapt its movements to maintain balance and achieve forward motion. This process is critical for developing a robust control system before deploying it on the physical hardware.

---

## 🚀 Installation

To install the necessary packages for this project, after cloning the repo, run the following command:

```bash
python -m pip install -e source/BDXR
```

```bash
cd BDX-R-Isaaclab/scripts/rsl_rl/
```

```bash
python train.py --task=Bdxr-Velocity-Flat-v0 --headless

```

```bash
python play.py --task=Bdxr-Velocity-Flat-Play-v0 --num_envs 100

```

## 🙏 Community and Acknowledgements

This project is a personal learning journey and would not have been possible without the guidance and inspiration from the wider robotics community. A special thank you to:

-   [louislelay](https://github.com/louislelay)
-   [skelmir](https://github.com/rimim)
-   [Kscalelabs](https://github.com/Kscalelabs)

Their expertise and willingness to share knowledge have been invaluable.
