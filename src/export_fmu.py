import os
import numpy as np
from pythonfmu.fmi2slave import Fmi2Slave, Fmi2Causality, Real

class DriverBehaviourFMU(Fmi2Slave):
    """
    FMI 2.0-compliant co-simulation FMU wrapping the trained PPO driving policy.
    
    Inputs:
        obs_0 to obs_258: 259-dimensional observation space vector.
    Outputs:
        steering_command: Continuous steering action in [-1.0, 1.0].
        acceleration_command: Continuous acceleration/braking action in [-1.0, 1.0].
    """
    
    author = "UST Intern"
    description = "RL-Based Driver Behaviour Policy Model"
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        # 1. Register 259 input observation variables
        self.obs_vals = [0.0] * 259
        for i in range(259):
            name = f"obs_{i}"
            setattr(self, name, 0.0)
            self.register_variable(Real(name, causality=Fmi2Causality.input))
            
        # 2. Register 2 output action variables
        self.steering_command: float = 0.0
        self.acceleration_command: float = 0.0
        self.register_variable(Real("steering_command", causality=Fmi2Causality.output))
        self.register_variable(Real("acceleration_command", causality=Fmi2Causality.output))
        
        # 3. Model reference and loading state
        self.model = None
        self.model_loaded: bool = False
        
    def do_step(self, current_time: float, step_size: float) -> bool:
        # Load the model on the first step to avoid overhead in construction
        if not self.model_loaded:
            self._load_model()
            
        # Reconstruct 259-dimensional observation vector
        obs_vec = np.zeros(259, dtype=np.float32)
        for i in range(259):
            obs_vec[i] = getattr(self, f"obs_{i}")
            
        # Run model inference
        if self.model_loaded and self.model is not None:
            try:
                # deterministic=True ensures identical actions for identical observations
                action, _ = self.model.predict(obs_vec, deterministic=True)
                self.steering_command = float(action[0])
                self.acceleration_command = float(action[1])
            except Exception:
                # Safety fallback
                self.steering_command = 0.0
                self.acceleration_command = 0.0
        else:
            # Fallback to passive/safe control if model is not loaded
            self.steering_command = 0.0
            self.acceleration_command = 0.0
            
        return True
        
    def _load_model(self) -> None:
        """
        Loads the trained PPO model weights.
        """
        try:
            import stable_baselines3
            import torch
            
            # Make PyTorch CPU-inference deterministic
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(False)  # False to avoid PyTorch errors on CPU unsupported operations, but we keep seed
            
            # Search for trained model files
            # Look for aggressive or cautious models in common directories
            model_filenames = [
                "cautious_S1_ppo.zip",
                "aggressive_S1_ppo.zip",
                "normal_S1_ppo.zip",
                "models/cautious_S1_ppo.zip",
                "models/aggressive_S1_ppo.zip"
            ]
            
            # We also check the resources directory of the FMU
            # When the FMU is instantiated, resources are located in resources/ folder
            resources_dir = getattr(self, "resources", "")
            if resources_dir:
                for fn in ["cautious_S1_ppo.zip", "aggressive_S1_ppo.zip"]:
                    model_filenames.append(os.path.join(resources_dir, fn))
                    
            model_path = None
            for path in model_filenames:
                if os.path.exists(path):
                    model_path = path
                    break
                    
            if model_path is not None:
                # Remove .zip extension if present since SB3 loads by prefix
                load_path = model_path
                if load_path.endswith(".zip"):
                    load_path = load_path[:-4]
                    
                self.model = stable_baselines3.PPO.load(load_path, device="cpu")
                self.model_loaded = True
            else:
                self.model_loaded = False
        except Exception:
            self.model_loaded = False
