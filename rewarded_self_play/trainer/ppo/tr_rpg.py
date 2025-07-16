"""
Task-Relative Regularized Policy Gradients (TR-RPG) implementation for CRSP.

This module implements the TR-RPG algorithm as specified in the CRSP paper,
providing superior theoretical guarantees compared to TRR++ through KL regularization.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class TRRPGConfig:
    """Configuration for TR-RPG algorithm."""
    policies: List[str] = None  # ['propose', 'solve', 'critique']
    beta_coefficients: Dict[str, float] = None  # Policy-specific KL regularization coefficients
    
    def __post_init__(self):
        if self.policies is None:
            self.policies = ['propose', 'solve', 'critique']
        
        if self.beta_coefficients is None:
            self.beta_coefficients = {
                'propose': 0.01,   # Encourage task diversity
                'solve': 0.05,     # Balance exploration/correctness
                'critique': 0.1    # Stable evaluation
            }


class TaskRelativeRPG:
    """
    Task-Relative Regularized Policy Gradients implementation.
    
    Extends the RPG framework to handle CRSP's three-policy architecture
    with policy-specific KL regularization and importance weighting.
    """
    
    def __init__(self, config: TRRPGConfig):
        self.config = config
        self.policies = config.policies
        self.beta_coefficients = config.beta_coefficients
        
        # Reference policies for KL regularization
        self.reference_policies = {}
        
        # Running statistics for reward normalization
        self.reward_stats = {
            policy: {'mean': 0.0, 'std': 1.0, 'count': 0}
            for policy in self.policies
        }
    
    def update_reference_policy(self, policy_name: str, policy_state_dict: Dict):
        """
        Update reference policy for KL regularization.
        
        Args:
            policy_name: Name of the policy ('propose', 'solve', 'critique')
            policy_state_dict: State dictionary of the policy
        """
        self.reference_policies[policy_name] = policy_state_dict.copy()
    
    def compute_importance_weights(self, 
                                 policy_name: str,
                                 new_log_probs: torch.Tensor,
                                 old_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute importance weights for off-policy correction.
        
        Args:
            policy_name: Name of the policy
            new_log_probs: Log probabilities from current policy
            old_log_probs: Log probabilities from reference policy
            
        Returns:
            Importance weights tensor
        """
        # Compute importance weights: w = π_new / π_old
        log_ratio = new_log_probs - old_log_probs
        importance_weights = torch.exp(log_ratio)
        
        # Clip importance weights for stability
        importance_weights = torch.clamp(importance_weights, 0.1, 10.0)
        
        return importance_weights
    
    def compute_kl_regularized_gradient(self,
                                      policy_name: str,
                                      rewards: torch.Tensor,
                                      importance_weights: torch.Tensor,
                                      log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute KL-regularized policy gradient.
        
        Args:
            policy_name: Name of the policy
            rewards: Reward tensor
            importance_weights: Importance weights
            log_probs: Log probabilities
            
        Returns:
            KL-regularized gradient tensor
        """
        beta = self.beta_coefficients[policy_name]
        
        # KL regularization term
        kl_term = beta * (torch.log(importance_weights) + 1)
        
        # Regularized advantage
        regularized_advantage = rewards - kl_term
        
        # Policy gradient with importance weighting
        gradient = importance_weights * regularized_advantage * log_probs
        
        return gradient
    
    def compute_policy_loss(self,
                           policy_name: str,
                           rewards: torch.Tensor,
                           new_log_probs: torch.Tensor,
                           old_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute TR-RPG loss for a specific policy.
        
        Args:
            policy_name: Name of the policy
            rewards: Reward tensor
            new_log_probs: Log probabilities from current policy
            old_log_probs: Log probabilities from reference policy
            
        Returns:
            TR-RPG loss tensor
        """
        # Compute importance weights
        importance_weights = self.compute_importance_weights(
            policy_name, new_log_probs, old_log_probs
        )
        
        # Get KL regularization coefficient
        beta = self.beta_coefficients[policy_name]
        
        # Compute KL-regularized loss
        kl_term = beta * torch.log(importance_weights)
        loss = importance_weights * (-rewards + kl_term)
        
        return loss.mean()
    
    def update_reward_statistics(self, policy_name: str, rewards: torch.Tensor):
        """
        Update running statistics for reward normalization.
        
        Args:
            policy_name: Name of the policy
            rewards: Reward tensor
        """
        rewards_np = rewards.detach().cpu().numpy()
        
        # Update running statistics
        stats = self.reward_stats[policy_name]
        old_count = stats['count']
        new_count = old_count + len(rewards_np)
        
        # Update mean
        old_mean = stats['mean']
        new_mean = (old_mean * old_count + rewards_np.sum()) / new_count
        
        # Update std (using Welford's online algorithm)
        if new_count > 1:
            old_var = stats['std'] ** 2
            new_var = ((old_count - 1) * old_var + 
                      np.sum((rewards_np - new_mean) ** 2)) / (new_count - 1)
            new_std = np.sqrt(new_var)
        else:
            new_std = 1.0
        
        # Update statistics
        stats['mean'] = new_mean
        stats['std'] = max(new_std, 1e-8)  # Prevent division by zero
        stats['count'] = new_count
    
    def normalize_rewards(self, policy_name: str, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize rewards using running statistics.
        
        Args:
            policy_name: Name of the policy
            rewards: Reward tensor
            
        Returns:
            Normalized reward tensor
        """
        stats = self.reward_stats[policy_name]
        normalized_rewards = (rewards - stats['mean']) / stats['std']
        return normalized_rewards
    
    def compute_kl_divergence(self,
                             new_log_probs: torch.Tensor,
                             old_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between new and old policies.
        
        Args:
            new_log_probs: Log probabilities from current policy
            old_log_probs: Log probabilities from reference policy
            
        Returns:
            KL divergence tensor
        """
        # KL(π_new || π_old) = E[log(π_new) - log(π_old)]
        kl_div = new_log_probs - old_log_probs
        return kl_div.mean()
    
    def get_policy_specific_config(self, policy_name: str) -> Dict:
        """
        Get policy-specific configuration.
        
        Args:
            policy_name: Name of the policy
            
        Returns:
            Policy-specific configuration dictionary
        """
        return {
            'beta': self.beta_coefficients[policy_name],
            'reward_stats': self.reward_stats[policy_name].copy()
        }


class TRRPGTrainer:
    """
    TR-RPG trainer for CRSP three-policy architecture.
    """
    
    def __init__(self, config: TRRPGConfig):
        self.tr_rpg = TaskRelativeRPG(config)
        self.policies = config.policies
        
        # Training state
        self.global_step = 0
        self.policy_update_frequency = 1  # Update reference policies every N steps
    
    def train_step(self,
                   policy_outputs: Dict[str, Dict],
                   rewards: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform one TR-RPG training step.
        
        Args:
            policy_outputs: Dictionary containing outputs for each policy
            rewards: Dictionary containing rewards for each policy
            
        Returns:
            Dictionary of losses for each policy
        """
        losses = {}
        
        for policy_name in self.policies:
            if policy_name not in policy_outputs or policy_name not in rewards:
                continue
            
            outputs = policy_outputs[policy_name]
            policy_rewards = rewards[policy_name]
            
            # Update reward statistics
            self.tr_rpg.update_reward_statistics(policy_name, policy_rewards)
            
            # Normalize rewards
            normalized_rewards = self.tr_rpg.normalize_rewards(policy_name, policy_rewards)
            
            # Compute TR-RPG loss
            loss = self.tr_rpg.compute_policy_loss(
                policy_name,
                normalized_rewards,
                outputs['new_log_probs'],
                outputs['old_log_probs']
            )
            
            losses[policy_name] = loss
        
        self.global_step += 1
        return losses
    
    def should_update_reference_policies(self) -> bool:
        """Check if reference policies should be updated."""
        return self.global_step % self.policy_update_frequency == 0
    
    def get_training_metrics(self) -> Dict[str, float]:
        """Get training metrics for logging."""
        metrics = {}
        
        for policy_name in self.policies:
            stats = self.tr_rpg.reward_stats[policy_name]
            metrics[f'{policy_name}/reward_mean'] = stats['mean']
            metrics[f'{policy_name}/reward_std'] = stats['std']
            metrics[f'{policy_name}/beta'] = self.tr_rpg.beta_coefficients[policy_name]
        
        return metrics