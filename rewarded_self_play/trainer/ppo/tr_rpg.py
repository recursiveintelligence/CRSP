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
import gc


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
        Compute importance weights for off-policy correction with memory optimization.
        
        Args:
            policy_name: Name of the policy
            new_log_probs: Log probabilities from current policy
            old_log_probs: Log probabilities from reference policy
            
        Returns:
            Importance weights tensor
        """
        # Compute log importance weights directly (avoid exp/log cycle)
        log_importance_weights = new_log_probs - old_log_probs
        
        # Clamp in log space for numerical stability (log(10) ≈ 2.3)
        log_importance_weights = torch.clamp(log_importance_weights, min=-2.3, max=2.3)
        
        # Compute importance weights with in-place operations
        importance_weights = torch.exp(log_importance_weights)
        
        return importance_weights
    
    def compute_importance_weights_chunked(self, 
                                         policy_name: str,
                                         new_log_probs: torch.Tensor,
                                         old_log_probs: torch.Tensor,
                                         chunk_size: int = 32) -> torch.Tensor:
        """
        Compute importance weights in chunks to minimize memory usage.
        
        Args:
            policy_name: Name of the policy
            new_log_probs: Log probabilities from current policy
            old_log_probs: Log probabilities from reference policy
            chunk_size: Size of chunks for processing
            
        Returns:
            Importance weights tensor
        """
        batch_size = new_log_probs.size(0)
        if batch_size <= chunk_size:
            return self.compute_importance_weights(policy_name, new_log_probs, old_log_probs)
        
        # Pre-allocate output tensor
        device = new_log_probs.device
        importance_weights = torch.empty_like(new_log_probs)
        
        # Process in chunks
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            
            # Extract chunk
            new_chunk = new_log_probs[i:end_idx]
            old_chunk = old_log_probs[i:end_idx]
            
            # Compute importance weights for chunk
            chunk_weights = self.compute_importance_weights(policy_name, new_chunk, old_chunk)
            
            # Store in pre-allocated tensor
            importance_weights[i:end_idx] = chunk_weights
            
            # Clear intermediate tensors
            del new_chunk, old_chunk, chunk_weights
        
        return importance_weights
    
    def compute_importance_weights_streaming(self,
                                           policy_name: str,
                                           new_log_probs: torch.Tensor,
                                           old_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute importance weights using streaming processing for minimal memory footprint.
        
        Args:
            policy_name: Name of the policy
            new_log_probs: Log probabilities from current policy
            old_log_probs: Log probabilities from reference policy
            
        Returns:
            Importance weights tensor
        """
        batch_size = new_log_probs.size(0)
        device = new_log_probs.device
        
        # Pre-allocate output tensor
        importance_weights = torch.empty_like(new_log_probs)
        
        # Process samples individually to minimize memory usage
        for i in range(batch_size):
            # Extract single sample
            new_prob = new_log_probs[i:i+1]
            old_prob = old_log_probs[i:i+1]
            
            # Compute log importance weight
            log_weight = new_prob - old_prob
            log_weight = torch.clamp(log_weight, min=-2.3, max=2.3)
            
            # Compute importance weight
            weight = torch.exp(log_weight)
            
            # Store result
            importance_weights[i:i+1] = weight
            
            # Clear intermediate tensors (not strictly necessary for single elements)
            del new_prob, old_prob, log_weight, weight
        
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
        Compute TR-RPG loss for a specific policy with memory optimization.
        
        Args:
            policy_name: Name of the policy
            rewards: Reward tensor
            new_log_probs: Log probabilities from current policy
            old_log_probs: Log probabilities from reference policy
            
        Returns:
            TR-RPG loss tensor
        """
        # Fused computation to minimize memory allocations
        beta = self.beta_coefficients[policy_name]
        
        # Compute log importance weights directly
        log_importance_weights = new_log_probs - old_log_probs
        log_importance_weights = torch.clamp(log_importance_weights, min=-2.3, max=2.3)
        
        # Fused computation: w * (-R + β*log(w))
        # = exp(log_w) * (-R + β*log_w)
        kl_penalty = beta * log_importance_weights
        loss_coeff = torch.exp(log_importance_weights) * (-rewards + kl_penalty)
        
        return loss_coeff.mean()
    
    def compute_policy_loss_chunked(self,
                                   policy_name: str,
                                   rewards: torch.Tensor,
                                   new_log_probs: torch.Tensor,
                                   old_log_probs: torch.Tensor,
                                   chunk_size: int = 32) -> torch.Tensor:
        """
        Compute TR-RPG loss with chunked processing for memory efficiency.
        
        Args:
            policy_name: Name of the policy
            rewards: Reward tensor
            new_log_probs: Log probabilities from current policy
            old_log_probs: Log probabilities from reference policy
            chunk_size: Size of chunks for processing
            
        Returns:
            TR-RPG loss tensor
        """
        batch_size = rewards.size(0)
        if batch_size <= chunk_size:
            return self.compute_policy_loss(policy_name, rewards, new_log_probs, old_log_probs)
        
        total_loss = 0.0
        num_chunks = 0
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            
            # Process chunk
            chunk_rewards = rewards[i:end_idx]
            chunk_new_probs = new_log_probs[i:end_idx]
            chunk_old_probs = old_log_probs[i:end_idx]
            
            chunk_loss = self.compute_policy_loss(
                policy_name, chunk_rewards, chunk_new_probs, chunk_old_probs
            )
            
            total_loss += chunk_loss.item()
            num_chunks += 1
            
            # Clear intermediate tensors
            del chunk_rewards, chunk_new_probs, chunk_old_probs, chunk_loss
        
        return torch.tensor(total_loss / num_chunks, device=rewards.device, requires_grad=True)
    
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


class SequentialPolicyProcessor:
    """
    Sequential policy processor for memory-efficient three-policy training.
    Processes policies one at a time to reduce memory pressure.
    """
    
    def __init__(self, policies: List[str] = None):
        self.policies = policies or ['propose', 'solve', 'critique']
        self.policy_order = self.policies.copy()  # Can be reordered for optimization
    
    def process_policies_sequentially(self, 
                                    policy_data: Dict[str, Dict],
                                    processing_fn,
                                    clear_cache: bool = True) -> Dict[str, any]:
        """
        Process each policy sequentially to reduce memory pressure.
        
        Args:
            policy_data: Dictionary containing data for each policy
            processing_fn: Function to process each policy's data
            clear_cache: Whether to clear GPU cache between policies
            
        Returns:
            Dictionary of processing results for each policy
        """
        results = {}
        
        for policy_name in self.policy_order:
            if policy_name not in policy_data:
                continue
            
            # Clear GPU cache before processing each policy
            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Process this policy
            policy_result = processing_fn(policy_name, policy_data[policy_name])
            results[policy_name] = policy_result
            
            # Force garbage collection periodically
            if policy_name != self.policy_order[-1]:  # Not the last policy
                gc.collect()
        
        return results
    
    def extract_policy_data(self, batch_data: Dict, policy_name: str) -> Dict:
        """
        Extract policy-specific data from batch.
        
        Args:
            batch_data: Full batch data
            policy_name: Name of the policy to extract data for
            
        Returns:
            Policy-specific data dictionary
        """
        # This is a placeholder - actual implementation would depend on data structure
        # In practice, this would extract the relevant portions of the batch for each policy
        policy_data = {}
        
        # Extract based on policy type
        if policy_name == 'propose':
            # Extract data relevant to task proposal
            policy_data = {
                'inputs': batch_data.get('task_inputs', []),
                'outputs': batch_data.get('task_outputs', []),
                'metadata': batch_data.get('task_metadata', {})
            }
        elif policy_name == 'solve':
            # Extract data relevant to problem solving
            policy_data = {
                'problems': batch_data.get('problems', []),
                'solutions': batch_data.get('solutions', []),
                'rewards': batch_data.get('solve_rewards', [])
            }
        elif policy_name == 'critique':
            # Extract data relevant to critique
            policy_data = {
                'solutions': batch_data.get('solutions', []),
                'reasoning': batch_data.get('reasoning_traces', []),
                'creativity_scores': batch_data.get('creativity_scores', [])
            }
        
        return policy_data
    
    def optimize_policy_order(self, memory_usage_stats: Dict[str, float]):
        """
        Optimize the order of policy processing based on memory usage statistics.
        
        Args:
            memory_usage_stats: Dictionary mapping policy names to memory usage
        """
        # Sort policies by memory usage (process memory-intensive ones first)
        sorted_policies = sorted(
            self.policies, 
            key=lambda p: memory_usage_stats.get(p, 0), 
            reverse=True
        )
        self.policy_order = sorted_policies


class GradientAccumulator:
    """
    Memory-efficient gradient accumulation manager for TR-RPG training.
    """
    
    def __init__(self, accumulation_steps: int, policies: List[str]):
        self.accumulation_steps = accumulation_steps
        self.policies = policies
        self.accumulated_gradients = {policy: {} for policy in policies}
        self.accumulation_count = 0
        
    def accumulate_gradients(self, policy_name: str, gradients: Dict[str, torch.Tensor], weight: float = 1.0):
        """
        Accumulate gradients for a specific policy with in-place operations.
        
        Args:
            policy_name: Name of the policy
            gradients: Dictionary of parameter gradients
            weight: Weight for this gradient contribution
        """
        if policy_name not in self.accumulated_gradients:
            self.accumulated_gradients[policy_name] = {}
        
        policy_grads = self.accumulated_gradients[policy_name]
        
        for param_name, grad in gradients.items():
            if param_name in policy_grads:
                # In-place accumulation to save memory
                policy_grads[param_name].add_(grad, alpha=weight)
            else:
                # Clone only when necessary
                policy_grads[param_name] = grad.clone() * weight
    
    def get_accumulated_gradients(self, policy_name: str, normalize: bool = True) -> Dict[str, torch.Tensor]:
        """
        Get accumulated gradients for a policy.
        
        Args:
            policy_name: Name of the policy
            normalize: Whether to normalize by accumulation steps
            
        Returns:
            Dictionary of accumulated gradients
        """
        if policy_name not in self.accumulated_gradients:
            return {}
        
        gradients = self.accumulated_gradients[policy_name]
        
        if normalize and self.accumulation_count > 0:
            # Normalize gradients by accumulation count
            normalized_grads = {}
            for param_name, grad in gradients.items():
                normalized_grads[param_name] = grad / self.accumulation_count
            return normalized_grads
        
        return gradients
    
    def clear_gradients(self, policy_name: str = None):
        """
        Clear accumulated gradients.
        
        Args:
            policy_name: Specific policy to clear, or None to clear all
        """
        if policy_name is not None:
            if policy_name in self.accumulated_gradients:
                self.accumulated_gradients[policy_name].clear()
        else:
            for policy in self.policies:
                self.accumulated_gradients[policy].clear()
            self.accumulation_count = 0
    
    def step(self):
        """Increment accumulation count."""
        self.accumulation_count += 1
    
    def should_update(self) -> bool:
        """Check if gradients should be applied."""
        return self.accumulation_count >= self.accumulation_steps
    
    def reset(self):
        """Reset accumulation state."""
        self.clear_gradients()
        self.accumulation_count = 0


class TRRPGTrainer:
    """
    TR-RPG trainer for CRSP three-policy architecture with memory optimization.
    """
    
    def __init__(self, config: TRRPGConfig, gradient_accumulation_steps: int = 4, max_batch_size: int = 32):
        self.tr_rpg = TaskRelativeRPG(config)
        self.policies = config.policies
        
        # Training state
        self.global_step = 0
        self.policy_update_frequency = 1  # Update reference policies every N steps
        
        # Memory optimization settings
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_batch_size = max_batch_size
        
        # Pre-allocated buffers for memory efficiency
        self._loss_buffers = {}
        self._gradient_buffers = {}
        
        # Initialize gradient accumulation manager
        self.gradient_accumulator = GradientAccumulator(
            accumulation_steps=gradient_accumulation_steps,
            policies=self.policies
        )
    
    def train_step(self,
                   policy_outputs: Dict[str, Dict],
                   rewards: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform one TR-RPG training step with memory optimization.
        
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
            
            # Use chunked processing for large batches
            batch_size = normalized_rewards.size(0)
            if batch_size > self.max_batch_size:
                loss = self.tr_rpg.compute_policy_loss_chunked(
                    policy_name,
                    normalized_rewards,
                    outputs['new_log_probs'],
                    outputs['old_log_probs'],
                    chunk_size=self.max_batch_size
                )
            else:
                loss = self.tr_rpg.compute_policy_loss(
                    policy_name,
                    normalized_rewards,
                    outputs['new_log_probs'],
                    outputs['old_log_probs']
                )
            
            losses[policy_name] = loss
            
            # Clear intermediate tensors to free memory
            del normalized_rewards
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.global_step += 1
        return losses
    
    def train_step_with_accumulation(self,
                                   policy_outputs: Dict[str, Dict],
                                   rewards: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform TR-RPG training step with gradient accumulation for memory efficiency.
        
        Args:
            policy_outputs: Dictionary containing outputs for each policy
            rewards: Dictionary containing rewards for each policy
            
        Returns:
            Dictionary of accumulated losses for each policy
        """
        accumulated_losses = {}
        
        # Process each policy sequentially to reduce memory pressure
        for policy_name in self.policies:
            if policy_name not in policy_outputs or policy_name not in rewards:
                continue
            
            outputs = policy_outputs[policy_name]
            policy_rewards = rewards[policy_name]
            batch_size = policy_rewards.size(0)
            
            # Calculate sub-batch size for gradient accumulation
            sub_batch_size = max(1, batch_size // self.gradient_accumulation_steps)
            total_loss = 0.0
            
            # Process in sub-batches
            for step in range(self.gradient_accumulation_steps):
                start_idx = step * sub_batch_size
                end_idx = min((step + 1) * sub_batch_size, batch_size)
                
                if start_idx >= batch_size:
                    break
                
                # Extract sub-batch
                sub_rewards = policy_rewards[start_idx:end_idx]
                sub_new_probs = outputs['new_log_probs'][start_idx:end_idx]
                sub_old_probs = outputs['old_log_probs'][start_idx:end_idx]
                
                # Update statistics and normalize
                self.tr_rpg.update_reward_statistics(policy_name, sub_rewards)
                normalized_sub_rewards = self.tr_rpg.normalize_rewards(policy_name, sub_rewards)
                
                # Compute loss for sub-batch
                sub_loss = self.tr_rpg.compute_policy_loss(
                    policy_name,
                    normalized_sub_rewards,
                    sub_new_probs,
                    sub_old_probs
                )
                
                total_loss += sub_loss.item()
                
                # Clear sub-batch tensors
                del sub_rewards, sub_new_probs, sub_old_probs, normalized_sub_rewards, sub_loss
                
                # Clear GPU cache periodically
                if step % 2 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Average loss across accumulation steps
            accumulated_losses[policy_name] = torch.tensor(
                total_loss / self.gradient_accumulation_steps,
                device=policy_rewards.device,
                requires_grad=True
            )
        
        self.global_step += 1
        return accumulated_losses
    
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