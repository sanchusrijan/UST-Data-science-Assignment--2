# Reward Function Design Document

This document details the design, mathematical formulation, and tuning rationale for the multi-objective reward function used to train the RL-based human driver behaviour models.

---

## 1. Mathematical Formulation

The reward function balances three primary objectives: **Safety**, **Goal Adherence**, and **Comfort**:

$$R_{total} = w_{safety} R_{safety} + w_{goal} R_{goal} + w_{comfort} R_{comfort}$$

Where $w_{safety}$, $w_{goal}$, and $w_{comfort}$ are the profile-specific weights.

---

### 1.1 Safety Sub-Objective ($R_{safety}$)

The safety objective aims to prevent collisions and maintain a safe buffer distance behind leading vehicles.

$$R_{safety} = R_{collision} + R_{headway} + R_{TTC}$$

#### Collision Penalty ($R_{collision}$)
A high step-level penalty is applied if the vehicle collides with another vehicle, static obstacle, or road boundary:
$$R_{collision} = \begin{cases} -20.0, & \text{if collision occurs} \\ 0.0, & \text{otherwise} \end{cases}$$

#### Time Headway Penalty ($R_{headway}$)
Time Headway ($H$) is the time it would take the agent to reach the leading vehicle's current position at its current speed:
$$H = \frac{d}{v_{agent}}$$
Where $d$ is the front-to-rear distance. If $H$ is less than the target headway ($H_{target}$), a proportional penalty is applied:
$$R_{headway} = \begin{cases} -2.0 \times (H_{target} - H), & \text{if } H < H_{target} \\ 0.0, & \text{otherwise} \end{cases}$$

#### Time-to-Collision (TTC) Penalty ($R_{TTC}$)
TTC is the estimated time before the agent collides with the lead vehicle if relative speed remains constant:
$$TTC = \frac{d}{v_{agent} - v_{leader}}$$
Where the penalty is applied only when the agent is faster than the leader ($v_{agent} - v_{leader} > 0$) and the TTC falls below a threshold ($TTC_{threshold}$):
$$R_{TTC} = \begin{cases} -5.0 \times (TTC_{threshold} - TTC), & \text{if } dv > 0 \text{ and } TTC < TTC_{threshold} \\ 0.0, & \text{otherwise} \end{cases}$$

---

### 1.2 Goal Adherence Sub-Objective ($R_{goal}$)

The goal adherence objective encourages the agent to maintain traffic flow speeds and stay centered in its lane.

$$R_{goal} = R_{speed} + 0.5 \times R_{lane}$$

#### Speed Tracking Error ($R_{speed}$)
Penalizes deviations from the target profile speed ($v_{target}$):
$$R_{speed} = -\frac{|v_{agent} - v_{target}|}{v_{target}}$$

#### Lane Centering Error ($R_{lane}$)
Penalizes lateral deviation ($d_{lat}$) from the lane center line, normalized by the half-lane width ($W_{lane} / 2$):
$$R_{lane} = -\frac{|d_{lat}|}{W_{lane}/2}$$

---

### 1.3 Comfort Sub-Objective ($R_{comfort}$)

The comfort objective ensures human-like smooth control inputs, penalizing sudden maneuvers and acceleration jerks.

$$R_{comfort} = 0.5 \times R_{steer} + 0.3 \times R_{accel} + 0.2 \times R_{jerk}$$

#### Steering Action Penalty ($R_{steer}$)
Penalizes large steering angles:
$$R_{steer} = -a_{steering}^2$$

#### Acceleration Action Penalty ($R_{accel}$)
Penalizes excessive acceleration or hard braking actions:
$$R_{accel} = -a_{accel}^2$$

#### Jerk Penalty ($R_{jerk}$)
Penalizes the rate of change of acceleration to avoid sudden jerking inputs:
$$R_{jerk} = -\min\left(\left(\frac{\Delta a_{agent}}{\Delta t}\right)^2 \times 0.01, 5.0\right)$$
Where $\Delta t = 0.1\text{ s}$ and $\Delta a_{agent} = a_t - a_{t-1}$.

---

## 2. Behavioral Profile Weights

To model different driving behaviors, we adjust the targets and weight balances as follows:

| Parameter / Weight | Aggressive Profile | Cautious Profile | Normal Profile (Default) |
| :--- | :--- | :--- | :--- |
| **Target Speed ($v_{target}$)** | $100 \text{ km/h} \ (27.8 \text{ m/s})$ | $70 \text{ km/h} \ (19.4 \text{ m/s})$ | $80 \text{ km/h} \ (22.2 \text{ m/s})$ |
| **Target Headway ($H_{target}$)**| $0.6 \text{ s}$ | $1.8 \text{ s}$ | $1.2 \text{ s}$ |
| **TTC Threshold ($TTC_{threshold}$)**| $1.0 \text{ s}$ | $2.0 \text{ s}$ | $1.5 \text{ s}$ |
| **Safety Weight ($w_{safety}$)** | $0.8$ | $2.5$ | $1.5$ |
| **Goal Weight ($w_{goal}$)** | $1.5$ | $0.8$ | $1.0$ |
| **Comfort Weight ($w_{comfort}$)** | $0.2$ | $0.7$ | $0.4$ |

---

## 3. Tuning Rationale

1. **Safety Primacy:** For all profiles, collision penalties remain high ($R_{collision} = -20.0$), but the cautious profile has a much larger safety weight ($w_{safety} = 2.5$). This ensures that a cautious driver prioritizes space cushion maintenance above speed matching.
2. **Aggressive Speed Goal:** The aggressive profile places a high weight on speed matching ($w_{goal} = 1.5$) and targets a higher speed, accepting smaller headway and TTC thresholds before penalties kick in. This forces the agent to overtake or tailgate slower vehicles.
3. **Comfort and Jerk:** Cautious drivers penalize jerk and large inputs more heavily ($w_{comfort} = 0.7$), resulting in smooth braking curves. Aggressive drivers tolerate higher jerk ($w_{comfort} = 0.2$), allowing for quick, sharp steering corrections and abrupt braking/acceleration.
