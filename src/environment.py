import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from typing import Dict, Tuple, Any, List, Optional
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.vehicle.vehicle_type import DefaultVehicle

class USTDriverEnv(MetaDriveEnv):
    """
    Custom MetaDrive environment for training and evaluating reinforcement learning-based
    human driver behaviour modelling.
    
    The learning agent acts as the 'NPC vehicle' reacting to a rule-based 'Ego vehicle'
    simulating various traffic interaction scenarios (S1-S5).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Default environment config
        default_config = {
            "map": "SSSSSSSSSS",  # Straight highway (10 blocks = ~1000m)
            "traffic_density": 0.0,  # No random traffic, we spawn our own scenario vehicles
            "use_render": False,
            "decision_repeat": 5,   # 10Hz decision frequency (0.02s * 5)
            "vehicle_config": {
                "show_lidar": False,
                "lidar": {
                    "num_lasers": 128,
                    "distance": 50.0,
                    "num_others": 0
                }
            }
        }
        
        # Scenario and behaviour profile are custom parameters.
        # We extract them first to prevent MetaDrive KeyError during config parsing.
        self.scenario: str = "S1"
        self.behaviour_profile: str = "normal"
        
        if config is not None:
            config_copy = config.copy()
            self.scenario = config_copy.pop("scenario", "S1")
            self.behaviour_profile = config_copy.pop("behaviour_profile", "normal")
            default_config.update(config_copy)
            
        super().__init__(default_config)
        
        # Track parameters for reward comfort terms
        self.prev_speed: float = 0.0
        self.prev_accel: float = 0.0
        self.step_count: int = 0
        self.ego_vehicle: Optional[DefaultVehicle] = None
        self.ego_lane_index: int = 1  # 0: left, 1: center/right
        
    @property
    def observation_space(self) -> gym.Space:
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(259,),
            dtype=np.float32
        )
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Clean up existing custom ego vehicle if any BEFORE resetting the base engine
        if hasattr(self, "ego_vehicle") and self.ego_vehicle is not None:
            try:
                if hasattr(self, "engine") and self.engine is not None:
                    ego_ref = self.ego_vehicle
                    self.engine.clear_objects(lambda x: x == ego_ref or x.name == ego_ref.name or getattr(x, "id", None) == getattr(ego_ref, "id", None))
            except Exception:
                pass
            self.ego_vehicle = None
            
        # Reset base environment (MetaDrive's BaseEnv.reset does not accept options argument)
        obs, info = super().reset(seed=seed)
        
        self.step_count = 0
        self.prev_speed = self.vehicle.speed
        self.prev_accel = 0.0
            
        # Spawn rule-based ego vehicle relative to agent's vehicle
        agent_vehicle = self.vehicle
        agent_pos = agent_vehicle.position
        
        current_lane = agent_vehicle.navigation.current_lane
        lane_heading = current_lane.heading_theta_at(agent_pos[0])
        
        # Configure starting states based on scenario
        # In MetaDrive, the highway lanes are separated laterally by 3.5m (negative left, positive right)
        self.ego_lane_index = 1
        if self.scenario == "S1" or self.scenario == "S2":
            # 50m ahead in same lane
            spawn_pos = agent_pos + np.array([50.0 * np.cos(lane_heading), 50.0 * np.sin(lane_heading)])
            spawn_heading = lane_heading
        elif self.scenario == "S3":
            # Cut-in: starts 35m ahead in the left lane (lateral offset -3.5m)
            self.ego_lane_index = 0
            offset = np.array([-3.5 * np.sin(lane_heading), 3.5 * np.cos(lane_heading)])
            spawn_pos = agent_pos + np.array([35.0 * np.cos(lane_heading), 35.0 * np.sin(lane_heading)]) + offset
            spawn_heading = lane_heading
        elif self.scenario == "S4":
            # 40m ahead in same lane
            spawn_pos = agent_pos + np.array([40.0 * np.cos(lane_heading), 40.0 * np.sin(lane_heading)])
            spawn_heading = lane_heading
        elif self.scenario == "S5":
            # Lane clearing: starts 45m ahead in same lane, moves to left lane later
            spawn_pos = agent_pos + np.array([45.0 * np.cos(lane_heading), 45.0 * np.sin(lane_heading)])
            spawn_heading = lane_heading
        else:
            spawn_pos = agent_pos + np.array([50.0 * np.cos(lane_heading), 50.0 * np.sin(lane_heading)])
            spawn_heading = lane_heading
            
        # Spawn via simulator engine
        ego_config = self.config["vehicle_config"].copy()
        ego_config["use_special_color"] = True  # Visually distinct
        
        self.ego_vehicle = self.engine.spawn_object(
            DefaultVehicle,
            vehicle_config=ego_config,
            position=spawn_pos,
            heading=spawn_heading
        )
        
        # Set initial velocities
        initial_speed = 22.2 if self.scenario != "S4" else 8.3  # 80 km/h vs 30 km/h
        self.ego_vehicle.set_velocity(np.array([initial_speed * np.cos(lane_heading), initial_speed * np.sin(lane_heading)]))
        
        return self._get_observation(), info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        
        # 1. Update rule-based ego vehicle behavior
        if self.ego_vehicle is not None:
            ego_steering, ego_accel = self._get_ego_rule_based_control()
            self.ego_vehicle.before_step([ego_steering, ego_accel])
            
        # 2. Step base environment using learning agent action
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 3. Calculate 259-dimensional observation space
        obs_vector = self._get_observation()
        
        # 4. Compute custom reward
        custom_reward = self._get_custom_reward(action)
        
        # 5. Handle custom collision termination conditions
        if self.vehicle.crash_vehicle or self.vehicle.crash_object or self.vehicle.crash_sidewalk:
            terminated = True
            info["collision"] = True
        else:
            info["collision"] = False
            
        # Keep track of speed for acceleration calculation
        self.prev_speed = self.vehicle.speed
        
        return obs_vector, custom_reward, terminated, truncated, info
        
    def _get_ego_rule_based_control(self) -> Tuple[float, float]:
        """
        Calculates rule-based steering and acceleration commands for the simulated Ego Vehicle.
        """
        ego_pos = self.ego_vehicle.position
        ego_vel = self.ego_vehicle.velocity
        ego_speed = self.ego_vehicle.speed
        
        agent_pos = self.vehicle.position
        current_lane = self.vehicle.navigation.current_lane
        
        # 10Hz decision rate implies 10 steps = 1 second
        dt = 0.1
        
        ego_steering = 0.0
        ego_accel = 0.0
        
        if self.scenario == "S1":
            # Maintain constant speed of 80 km/h (22.2 m/s)
            target_speed = 22.2
            ego_accel = 0.5 * (target_speed - ego_speed)
            
        elif self.scenario == "S2":
            # Constant speed for 3 seconds, then emergency braking
            if self.step_count < 30:
                target_speed = 22.2
                ego_accel = 0.5 * (target_speed - ego_speed)
            else:
                # Brake hard (~8 m/s^2, represented as full brake -1.0)
                ego_accel = -1.0
                
        elif self.scenario == "S3":
            # Cut-in: starts in left lane, cuts in front of agent at step 20 (2.0s)
            if self.step_count < 20:
                # Maintain speed of 70 km/h (19.4 m/s) and stay in left lane (index 0)
                target_speed = 19.4
                ego_accel = 0.5 * (target_speed - ego_speed)
                # Keep steering straight along the lane heading
                ego_steering = 0.0
            else:
                # Cut-in: steer into the agent's lane (right lane, lateral offset +3.5m relative to start)
                # target lateral position is agent's lane center
                target_speed = 19.4
                ego_accel = 0.5 * (target_speed - ego_speed)
                
                # Check current lateral position
                lateral_pos = current_lane.local_coordinates(ego_pos)[1]
                # Lateral target is 0.0 (agent's lane center)
                # Simple proportional steering to perform lane change
                error = 0.0 - lateral_pos
                ego_steering = np.clip(0.3 * error, -0.4, 0.4)
                
        elif self.scenario == "S4":
            # Slow leader: maintains 30 km/h (8.3 m/s)
            target_speed = 8.3
            ego_accel = 0.5 * (target_speed - ego_speed)
            
        elif self.scenario == "S5":
            # Lane clearing: drives at 60 km/h (16.7 m/s), changes to left lane at step 30
            if self.step_count < 30:
                target_speed = 16.7
                ego_accel = 0.5 * (target_speed - ego_speed)
                ego_steering = 0.0
            else:
                # Move to left lane (index 0, lateral offset -3.5m)
                target_speed = 16.7
                ego_accel = 0.5 * (target_speed - ego_speed)
                
                lateral_pos = current_lane.local_coordinates(ego_pos)[1]
                error = -3.5 - lateral_pos
                ego_steering = np.clip(0.3 * error, -0.4, 0.4)
                
        # Clip control inputs
        ego_steering = float(np.clip(ego_steering, -1.0, 1.0))
        ego_accel = float(np.clip(ego_accel, -1.0, 1.0))
        
        return ego_steering, ego_accel
        
    def _get_observation(self) -> np.ndarray:
        """
        Extracts and formats the 259-dimensional observation space vector.
        """
        agent_vehicle = self.vehicle
        ego_pos = agent_vehicle.position
        ego_heading = agent_vehicle.heading_theta
        ego_vel = agent_vehicle.velocity
        ego_speed = agent_vehicle.speed
        
        # 1. Ego State (7 features)
        dt = 0.1
        ego_accel = (ego_speed - self.prev_speed) / dt
        ego_steering = agent_vehicle.steering
        
        try:
            lane_index = float(agent_vehicle.navigation.current_lane.index[2])
        except Exception:
            lane_index = 1.0
            
        ego_state = np.array([
            ego_pos[0],
            ego_pos[1],
            ego_heading,
            ego_vel[0],
            ego_vel[1],
            ego_accel,
            ego_steering
        ], dtype=np.float32)
        
        # 2. Nearby NPC States (10 NPCs x 6 features = 60 features)
        # Find other vehicles (excluding the learning agent itself)
        npc_features = []
        all_vehicles = self.engine.get_objects(lambda x: isinstance(x, DefaultVehicle) and x != agent_vehicle)
        sorted_npcs = sorted(all_vehicles.values(), key=lambda v: np.linalg.norm(v.position - ego_pos))
        
        for i in range(10):
            if i < len(sorted_npcs):
                npc = sorted_npcs[i]
                npc_pos = npc.position
                npc_vel = npc.velocity
                npc_heading = npc.heading_theta
                
                # Compute relative position in ego-centric frame
                dx_w = npc_pos[0] - ego_pos[0]
                dy_w = npc_pos[1] - ego_pos[1]
                dx_ego = dx_w * np.cos(ego_heading) + dy_w * np.sin(ego_heading)
                dy_ego = -dx_w * np.sin(ego_heading) + dy_w * np.cos(ego_heading)
                
                # Compute relative velocity in ego-centric frame
                dvx_w = npc_vel[0] - ego_vel[0]
                dvy_w = npc_vel[1] - ego_vel[1]
                dvx_ego = dvx_w * np.cos(ego_heading) + dvy_w * np.sin(ego_heading)
                dvy_ego = -dvx_w * np.sin(ego_heading) + dvy_w * np.cos(ego_heading)
                
                # Heading difference
                heading_diff = npc_heading - ego_heading
                heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
                
                # Euclidean distance
                dist = np.linalg.norm(npc_pos - ego_pos)
                
                npc_features.extend([dx_ego, dy_ego, dvx_ego, dvy_ego, heading_diff, dist])
            else:
                npc_features.extend([0.0] * 6)
                
        npc_state = np.array(npc_features, dtype=np.float32)
        
        # 3. Waypoint / Path Features (20 features)
        waypoint_features = []
        current_lane = agent_vehicle.navigation.current_lane
        long_pos = current_lane.local_coordinates(ego_pos)[0]
        
        for step in range(1, 11):
            sample_dist = long_pos + step * 10.0  # 10 waypoints spaced 10m apart
            sample_dist = min(sample_dist, current_lane.length)
            waypoint_w = current_lane.position(sample_dist, 0.0)
            
            dx_w = waypoint_w[0] - ego_pos[0]
            dy_w = waypoint_w[1] - ego_pos[1]
            dx_ego = dx_w * np.cos(ego_heading) + dy_w * np.sin(ego_heading)
            dy_ego = -dx_w * np.sin(ego_heading) + dy_w * np.cos(ego_heading)
            
            waypoint_features.extend([dx_ego, dy_ego])
            
        waypoint_state = np.array(waypoint_features, dtype=np.float32)
        
        # 4. Lidar / Depth Features (128 features)
        try:
            lidar_sensor = self.engine.get_sensor("lidar")
            res = lidar_sensor.perceive(agent_vehicle, self.engine.physics_world, 128, 50.0)
            if isinstance(res, np.ndarray):
                lidar_data = res.flatten()
            else:
                lidar_data = np.array(lidar_sensor.get_cloud_points(), dtype=np.float32)
        except Exception:
            lidar_data = np.ones(128, dtype=np.float32)
            
        # Ensure exact shape of 128
        if len(lidar_data) > 128:
            lidar_data = lidar_data[:128]
        elif len(lidar_data) < 128:
            lidar_data = np.pad(lidar_data, (0, 128 - len(lidar_data)), 'constant', constant_values=1.0)
            
        # 5. Road / Map Features (44 features)
        lat = current_lane.local_coordinates(ego_pos)[1]
        road_features = [
            current_lane.width,
            22.2,  # Speed limit (80 km/h)
            current_lane.width * 3.0,  # 3-lane road width
            current_lane.width / 2.0 - lat,  # Dist to left lane boundary
            current_lane.width / 2.0 + lat,  # Dist to right lane boundary
            0.0,  # Road curvature (0.0 for straight highway)
            0.0,  # Intersection proximity
            0.0   # Traffic light state
        ]
        road_features.extend([0.0] * 36)  # Pad to make exactly 44 features
        road_state = np.array(road_features, dtype=np.float32)
        
        # Combine all features into the 259-dimensional vector
        obs_vector = np.concatenate([
            ego_state,       # 7
            npc_state,       # 60
            waypoint_state,  # 20
            lidar_data,      # 128
            road_state       # 44
        ])
        
        return obs_vector
        
    def _get_custom_reward(self, action: np.ndarray) -> float:
        """
        Computes a multi-objective reward balancing Safety, Goal Adherence, and Comfort.
        """
        agent_vehicle = self.vehicle
        ego_pos = agent_vehicle.position
        ego_speed = agent_vehicle.speed
        current_lane = agent_vehicle.navigation.current_lane
        
        # Find leading vehicle and distance to calculate headway/TTC
        leader = None
        min_dist_ahead = float('inf')
        
        all_vehicles = self.engine.get_objects(lambda x: isinstance(x, DefaultVehicle) and x != agent_vehicle)
        for npc in all_vehicles.values():
            # Check relative positions in ego coordinates
            dx_w = npc.position[0] - ego_pos[0]
            dy_w = npc.position[1] - ego_pos[1]
            dx_ego = dx_w * np.cos(agent_vehicle.heading_theta) + dy_w * np.sin(agent_vehicle.heading_theta)
            dy_ego = -dx_w * np.sin(agent_vehicle.heading_theta) + dy_w * np.cos(agent_vehicle.heading_theta)
            
            # Check if NPC is ahead and in the same lane (lateral limit of 1.8m covers standard lane width)
            if dx_ego > 0 and abs(dy_ego) < 1.8:
                if dx_ego < min_dist_ahead:
                    min_dist_ahead = dx_ego
                    leader = npc
                    
        # Determine behavior profile parameters
        if self.behaviour_profile == "aggressive":
            target_speed = 27.8      # 100 km/h
            target_headway = 0.6     # 0.6 seconds
            ttc_threshold = 1.0      # 1.0 seconds
            w_safety = 0.8
            w_goal = 1.5
            w_comfort = 0.2
        elif self.behaviour_profile == "cautious":
            target_speed = 19.4      # 70 km/h
            target_headway = 1.8     # 1.8 seconds
            ttc_threshold = 2.0      # 2.0 seconds
            w_safety = 2.5
            w_goal = 0.8
            w_comfort = 0.7
        else:  # normal
            target_speed = 22.2      # 80 km/h
            target_headway = 1.2     # 1.2 seconds
            ttc_threshold = 1.5      # 1.5 seconds
            w_safety = 1.5
            w_goal = 1.0
            w_comfort = 0.4
            
        # 1. Safety Sub-Objective
        r_safety = 0.0
        # Collision penalty
        if agent_vehicle.crash_vehicle or agent_vehicle.crash_object or agent_vehicle.crash_sidewalk:
            r_safety -= 20.0
            
        if leader is not None:
            # Time Headway
            headway = min_dist_ahead / max(ego_speed, 1.0)
            if headway < target_headway:
                r_safety -= 2.0 * (target_headway - headway)
                
            # Time-to-Collision (TTC)
            dv = ego_speed - leader.speed
            if dv > 0:
                ttc = min_dist_ahead / dv
                if ttc < ttc_threshold:
                    r_safety -= 5.0 * (ttc_threshold - ttc)
                    
        # 2. Goal Adherence Sub-Objective
        # Speed tracking error
        speed_err = abs(ego_speed - target_speed) / target_speed
        r_speed = -speed_err
        
        # Lane centering error
        lat_pos = current_lane.local_coordinates(ego_pos)[1]
        r_lane = -abs(lat_pos) / (current_lane.width / 2.0)
        
        r_goal = r_speed + 0.5 * r_lane
        
        # 3. Comfort Sub-Objective
        # Steering and throttle penalties
        r_steer = - (action[0] ** 2)
        r_accel = - (action[1] ** 2)
        
        # Jerk estimation
        dt = 0.1
        ego_accel = (ego_speed - self.prev_speed) / dt
        jerk = (ego_accel - self.prev_accel) / dt
        self.prev_accel = ego_accel
        r_jerk = -min((jerk / 10.0) ** 2, 5.0)  # normalized jerk penalty
        
        r_comfort = 0.5 * r_steer + 0.3 * r_accel + 0.2 * r_jerk
        
        # Final weighted reward
        total_reward = float(w_safety * r_safety + w_goal * r_goal + w_comfort * r_comfort)
        return total_reward
