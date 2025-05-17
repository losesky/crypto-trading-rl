import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium


class TimeSeriesCNN(BaseFeaturesExtractor):
    """
    Input  : (batch, 9, 100)   # we permute the obs for channels_first
    Output : flat 256-dim vector fed to actor/critic MLPs
    """

    def __init__(self, observation_space: gymnasium.spaces.Box):
        super().__init__(observation_space, features_dim=256)

        # The CNN expects (batch, num_features, seq_len)
        # observation_space.shape is (seq_len, num_features), e.g. (L, 9)
        # num_features is assumed to be 9 based on original Conv1d in_channels
        # and sample th.zeros(1, 9, 100)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Infer conv output size to plug a fully-connected layer
        # Use a sample from the actual observation_space
        with th.no_grad():
            # observation_space.sample() has shape (seq_len, num_features)
            # Add batch dimension: (1, seq_len, num_features)
            # Permute to (1, num_features, seq_len) for cnn input
            # e.g., if obs_space.shape is (99, 9), dummy_input_for_cnn is (1, 9, 99)
            dummy_input_for_cnn = th.as_tensor(observation_space.sample()[None]).float().permute(0, 2, 1)
            
            # Ensure the channel dimension of this sample matches the Conv1d in_channels
            # This check is more for robustness, assuming num_features is consistently 9 from data pipeline
            if self.cnn[0].in_channels != dummy_input_for_cnn.shape[1]:
                raise ValueError(
                    f"Observation space feature dimension ({dummy_input_for_cnn.shape[1]}) "
                    f"does not match expected CNN in_channels ({self.cnn[0].in_channels})."
                )
            
            n_flat = self.cnn(dummy_input_for_cnn).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flat, 256),
            nn.ReLU(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # obs arrives as (batch, 100, 9) → transpose to (batch, 9, 100)
        x = obs.permute(0, 2, 1)
        x = self.cnn(x)
        return self.fc(x)