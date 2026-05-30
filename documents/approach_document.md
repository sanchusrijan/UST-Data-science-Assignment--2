# Approach Document: Reinforcement Learning-Based Human Driver Behaviour Modelling

This document outlines our engineering approach, system architecture, and validation strategy for building human-like traffic agent models deployable in Eclipse OpenPASS via an FMI 2.0 co-simulation FMU.

---

## 1. Problem Understanding

Autonomous driving validation requires high-fidelity, reactive simulations. Traditional rule-based traffic controllers (like pure IDM) produce highly predictable, low-variance behavior, leading to unrealistic testing conditions. 

This project trains **Reinforcement Learning (RL) agents** that behave like human drivers (e.g. following, overtaking, or yielding to a rule-based ego vehicle) and packages them into **FMI 2.0 Functional Mock-up Units (FMUs)**. This allows the trained policies to act as traffic participants in OpenPASS, running co-simulations via standard OpenSimulationInterface (OSI) messages.

---

## 2. OSMP Bridge & System Architecture

The **OSMP (Open Simulation Integration) Bridge** coordinates bidirectional communication between the OpenPASS simulation loop and the RL model.

```
       OSMP BRIDGE (FMU 2.0 Co-Simulation Wrapper)
+------------------------------------------------------+
|                                                      |
|  [OSI Message Receiver]                              |
|           │                                          |
|           ▼ (Parses OSI 3.x message)                 |
|  [Environment State Translator]                       |
|           │                                          |
|           ▼ (Transforms frames & builds 259-dim vector)
|  [RL Model Interface (PPO Policy)]                   |
|           │                                          |
|           ▼ (Invokes CPU-deterministic inference)    |
|  [Action Translator]                                 |
|           │                                          |
|           ▼ (Maps action to target yaw & speed)       |
|  [OSI Actuator Message Interface]                    |
|                                                      |
+------------------------------------------------------+
```

### 2.1 Bidirectional Data Flow
1. **Forward Pass (MetaDrive/OpenPASS to RL):** At each tick, the environment state (positions, heading, lanes, lidar points) is parsed from OSI messages. The bridge transforms coordinates from the OpenPASS world frame to an ego-centric frame centered on the NPC, constructing the exact **259-dimensional observation space vector**.
2. **Reverse Pass (RL to OpenPASS/MetaDrive):** The RL policy processes the observation vector and returns continuous steering and acceleration actions. The Action Translator converts steering in $[-1, 1]$ into a target yaw rate, and acceleration in $[-1, 1]$ into desired velocity using a bicycle kinematic model, which is sent back to OpenPASS via OSI.

---

## 3. Observation Space (259 Features)

The 259-dimensional observation vector ensures full environmental awareness:
- **Ego State (7 features):** Position $(x,y)$, heading, velocity $(vx,vy)$, acceleration, steering angle.
- **Nearby NPC States (60 features):** Relative position $(dx,dy)$, velocity $(dvx,dvy)$, heading difference, and Euclidean distance for the top 10 nearest vehicles.
- **Waypoint / Path Features (20 features):** Relative coordinates $(dx,dy)$ for 10 upcoming route points along the lane center line.
- **Lidar Features (128 features):** Normalized distances from a 128-beam ray-casting sensor.
- **Road / Map Features (44 features):** Lane width, speed limits, road width, distance to left/right boundaries, and padded metadata.

---

## 4. Model Architecture & Training

### 4.1 Neural Network Setup
We use the **PPO (Proximal Policy Optimization)** algorithm implemented via **Stable-Baselines3**.
- **Architecture:** 2-layer Multi-Layer Perceptron (MLP) with 256 hidden units per layer.
- **Topology:** Separate policy (actor) and value (critic) heads, preventing value-update noise from destabilizing the action distributions.
- **Action Space:** Continuous action space: `[steering_angle, acceleration]` bounded in $[-1, 1]$.

### 4.2 Behavioral Profile Training
We train two distinct behavioral profiles by configuring parameters in the custom reward function:
- **Aggressive Profile:** Smaller safety headway ($0.6\text{ s}$), smaller TTC threshold ($1.0\text{ s}$), higher target speed ($100\text{ km/h}$).
- **Cautious Profile:** Larger safety headway ($1.8\text{ s}$), larger TTC threshold ($2.0\text{ s}$), lower target speed ($70\text{ km/h}$).

---

## 5. KPI Fulfillment Strategy

- **Safety (Collision Rate < 5%):** Ensured by placing high weights on safety headway and TTC violations, backed by a large collision penalty ($R_{collision} = -20.0$).
- **Human-Likeness (KS-test p > 0.05):** We run Kolmogorov-Smirnov tests comparing the agent's headway distributions against NGSIM highway driving datasets. The reward's comfort jerk penalties prevent robotic, high-frequency oscillations.
- **Reaction Time (N(250ms, 50ms)):** In S2 braking scenarios, the model learns to react to deceleration triggers. By structuring the PPO policy's observation window, we calibrate step delays to align with human response times (~2-3 simulation ticks at 10Hz).
- **Latency (< 75ms):** PPO model inference is run in CPU-only mode with a thread limit of 1. Because the model is a small 2-layer MLP, CPU inference latency is extremely low (typically $< 3\text{ ms}$), far exceeding the 75ms constraint.
- **FMU 2.0 Portability:** We use `pythonfmu` to build the FMU co-simulation container. The zip file contains the trained neural network model weights and `modelDescription.xml` specifying inputs and outputs, allowing seamless execution.
