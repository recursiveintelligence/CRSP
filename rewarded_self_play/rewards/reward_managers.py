import os
import logging
import json
from functools import partial
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import re
import uuid
from functools import partial

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from verl import DataProto
from verl.protocol import DataProtoItem
from verl.utils.dataset.rl_dataset import collate_fn
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

import rewarded_self_play.rewards.custom_evaluate as custom_evaluate
from rewarded_self_play.rewards.code_reward import (
    parse_code_input_output,
    parse_inputs_message,
    parse_code_function,
    ast_edit_distance,
    get_code_complexity_reward,
    get_halstead_reward,
    get_type_counts_reward,
)
from rewarded_self_play.rewards.custom_evaluate import get_format_reward, extract_answer, extract_thought
from rewarded_self_play.data_construction.process_data import boxed_instruction, instruction_following
from rewarded_self_play.data_construction.constructor import get_code_problem_predictor_prompt
from rewarded_self_play.utils.dataset.rl_dataset import RLHFDataset
from rewarded_self_play.utils.logging_utils.stdout import PrettyPrinter
from rewarded_self_play.utils.code_utils.checks import check_composite_function, check_no_definitions
from rewarded_self_play.data_construction.prompts import get_creativity_grading_prompt


class CritiqueManager:
    """Manager for evaluating reasoning quality and creativity."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
    
    @staticmethod
    def extract_think_tags(solution_text: str) -> str:
        """
        Extract content within <think> tags from solution text.
        
        Args:
            solution_text: The solution text containing <think> tags
            
        Returns:
            The reasoning content within <think> tags, or empty string if not found
        """
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, solution_text, re.DOTALL)
        
        if think_matches:
            return think_matches[0].strip()
        return ""
    
    def evaluate_reasoning(self, solution_text: str, model=None) -> Tuple[float, str]:
        """
        Evaluate the reasoning trajectory within <think> tags.
        
        Args:
            solution_text: The solution text containing <think> tags
            model: The model to use for evaluation (if None, returns default score)
            
        Returns:
            Tuple of (creativity_score, rationale)
        """
        think_content = self.extract_think_tags(solution_text)
        
        if not think_content:
            return 0.0, "No reasoning content found in <think> tags"
        
        if model is None:
            # Return a default score based on length and complexity heuristics
            tokens = self.tokenizer.encode(think_content)
            token_count = len(tokens)
            
            # Simple heuristic: longer reasoning gets higher creativity score
            if token_count < 10:
                return 0.1, "Very short reasoning"
            elif token_count < 50:
                return 0.3, "Short reasoning with some detail"
            elif token_count < 200:
                return 0.6, "Moderate reasoning with good detail"
            else:
                return 0.8, "Extensive reasoning with thorough exploration"
        
        # If model is provided, use it for evaluation
        try:
            prompt = get_creativity_grading_prompt(think_content)
            response = model.generate(prompt)
            
            # Parse JSON response
            import json
            evaluation = json.loads(response)
            return evaluation.get('grade', 0.0), evaluation.get('rationale', 'No rationale provided')
        except Exception as e:
            # Fallback to heuristic if model evaluation fails
            return 0.5, f"Model evaluation failed: {str(e)}"
    
    def compute_creativity_reward(self, solution_text: str, model=None) -> float:
        """
        Compute creativity reward based on reasoning evaluation.
        
        Args:
            solution_text: The solution text to evaluate
            model: Optional model for evaluation
            
        Returns:
            Creativity reward score between 0 and 1
        """
        creativity_score, _ = self.evaluate_reasoning(solution_text, model)
        return creativity_score


class AlphaDecayScheduler:
    """RLSP-inspired alpha decay scheduler for CRSP."""
    
    @staticmethod
    def compute_alpha_decay(step: int, total_steps: int) -> Tuple[float, float]:
        """
        Compute alpha values for solver and critique rewards using RLSP-inspired coupled decay.
        
        Args:
            step: Current training step
            total_steps: Total number of training steps
            
        Returns:
            Tuple of (alpha_s, alpha_c) values
        """
        progress = step / total_steps
        
        if progress < 0.2:
            # Early exploration phase
            alpha_s, alpha_c = 0.3, 0.1
        elif progress < 0.6:
            # Interpolation phase
            phase_progress = (progress - 0.2) / 0.4
            alpha_s = 0.3 + 0.5 * phase_progress  # 0.3 → 0.8
            alpha_c = 0.1 + 0.5 * phase_progress  # 0.1 → 0.6
        else:
            # Final convergence phase
            phase_progress = (progress - 0.6) / 0.4
            alpha_s = 0.8 + 0.15 * phase_progress  # 0.8 → 0.95
            alpha_c = 0.6 + 0.2 * phase_progress   # 0.6 → 0.8
        
        return alpha_s, alpha_c


class CRSPLogger:
    """Comprehensive logging for CRSP training metrics."""
    
    def __init__(self, output_dir: str, log_level: str = "INFO"):
        self.output_dir = output_dir
        self.logger = logging.getLogger("CRSP")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create file handler for detailed logs
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(output_dir, "crsp_training.log"))
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
        # Initialize metrics storage
        self.metrics_history = {
            'rewards': [],
            'alpha_values': [],
            'creativity_scores': [],
            'length_rewards': [],
            'tr_rpg_metrics': [],
            'trajectory_seeding': []
        }
    
    def log_reward_breakdown(self, step: int, rewards: Dict[str, float], alpha_s: float, alpha_c: float):
        """Log detailed reward breakdown for debugging and monitoring."""
        self.logger.info(f"Step {step} - Reward Breakdown:")
        self.logger.info(f"  Alpha values: α_s={alpha_s:.3f}, α_c={alpha_c:.3f}")
        
        for reward_type, value in rewards.items():
            self.logger.info(f"  {reward_type}: {value:.4f}")
        
        # Store for history
        self.metrics_history['rewards'].append({
            'step': step,
            'rewards': rewards.copy(),
            'timestamp': pd.Timestamp.now()
        })
        self.metrics_history['alpha_values'].append({
            'step': step,
            'alpha_s': alpha_s,
            'alpha_c': alpha_c,
            'timestamp': pd.Timestamp.now()
        })
    
    def log_creativity_evaluation(self, step: int, creativity_scores: List[float], rationales: List[str]):
        """Log creativity evaluation results."""
        avg_creativity = np.mean(creativity_scores)
        std_creativity = np.std(creativity_scores)
        
        self.logger.info(f"Step {step} - Creativity Evaluation:")
        self.logger.info(f"  Average creativity score: {avg_creativity:.4f} ± {std_creativity:.4f}")
        self.logger.info(f"  Score distribution: min={min(creativity_scores):.3f}, "
                        f"max={max(creativity_scores):.3f}, median={np.median(creativity_scores):.3f}")
        
        # Log sample rationales
        if rationales:
            self.logger.debug("Sample creativity rationales:")
            for i, rationale in enumerate(rationales[:3]):  # Log first 3 rationales
                self.logger.debug(f"  Sample {i+1}: {rationale}")
        
        # Store for history
        self.metrics_history['creativity_scores'].append({
            'step': step,
            'scores': creativity_scores.copy(),
            'avg_score': avg_creativity,
            'std_score': std_creativity,
            'timestamp': pd.Timestamp.now()
        })
    
    def log_length_rewards(self, step: int, length_rewards: List[float], token_counts: List[int]):
        """Log length reward statistics."""
        avg_length_reward = np.mean(length_rewards)
        avg_token_count = np.mean(token_counts)
        
        self.logger.info(f"Step {step} - Length Rewards:")
        self.logger.info(f"  Average length reward: {avg_length_reward:.4f}")
        self.logger.info(f"  Average token count: {avg_token_count:.1f}")
        self.logger.info(f"  Token count range: {min(token_counts)}-{max(token_counts)}")
        
        # Store for history
        self.metrics_history['length_rewards'].append({
            'step': step,
            'length_rewards': length_rewards.copy(),
            'token_counts': token_counts.copy(),
            'avg_length_reward': avg_length_reward,
            'avg_token_count': avg_token_count,
            'timestamp': pd.Timestamp.now()
        })
    
    def log_tr_rpg_metrics(self, step: int, kl_divergences: Dict[str, float], 
                          importance_weights: Dict[str, List[float]], policy_losses: Dict[str, float]):
        """Log TR-RPG specific metrics."""
        self.logger.info(f"Step {step} - TR-RPG Metrics:")
        
        # Log KL divergences
        for policy, kl_div in kl_divergences.items():
            self.logger.info(f"  KL divergence ({policy}): {kl_div:.6f}")
        
        # Log importance weight statistics
        for policy, weights in importance_weights.items():
            if weights:
                avg_weight = np.mean(weights)
                max_weight = np.max(weights)
                self.logger.info(f"  Importance weights ({policy}): avg={avg_weight:.4f}, max={max_weight:.4f}")
        
        # Log policy losses
        for policy, loss in policy_losses.items():
            self.logger.info(f"  Policy loss ({policy}): {loss:.6f}")
        
        # Store for history
        self.metrics_history['tr_rpg_metrics'].append({
            'step': step,
            'kl_divergences': kl_divergences.copy(),
            'importance_weights': {k: v.copy() if isinstance(v, list) else v for k, v in importance_weights.items()},
            'policy_losses': policy_losses.copy(),
            'timestamp': pd.Timestamp.now()
        })
    
    def log_trajectory_seeding_progress(self, epoch: int, loss: float, eval_loss: float = None, 
                                      learning_rate: float = None):
        """Log trajectory seeding progress."""
        self.logger.info(f"Trajectory Seeding - Epoch {epoch}:")
        self.logger.info(f"  Training loss: {loss:.6f}")
        if eval_loss is not None:
            self.logger.info(f"  Validation loss: {eval_loss:.6f}")
        if learning_rate is not None:
            self.logger.info(f"  Learning rate: {learning_rate:.2e}")
        
        # Store for history
        self.metrics_history['trajectory_seeding'].append({
            'epoch': epoch,
            'loss': loss,
            'eval_loss': eval_loss,
            'learning_rate': learning_rate,
            'timestamp': pd.Timestamp.now()
        })
    
    def save_metrics_summary(self, step: int):
        """Save comprehensive metrics summary to file."""
        summary_path = os.path.join(self.output_dir, f"metrics_summary_step_{step}.json")
        
        # Create summary statistics
        summary = {
            'step': step,
            'timestamp': pd.Timestamp.now().isoformat(),
            'reward_stats': self._compute_reward_stats(),
            'alpha_progression': self._compute_alpha_progression(),
            'creativity_trends': self._compute_creativity_trends(),
            'tr_rpg_stability': self._compute_tr_rpg_stability()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Metrics summary saved to: {summary_path}")
    
    def _compute_reward_stats(self) -> Dict[str, Any]:
        """Compute reward statistics over recent history."""
        if not self.metrics_history['rewards']:
            return {}
        
        recent_rewards = self.metrics_history['rewards'][-100:]  # Last 100 steps
        reward_types = set()
        for entry in recent_rewards:
            reward_types.update(entry['rewards'].keys())
        
        stats = {}
        for reward_type in reward_types:
            values = [entry['rewards'].get(reward_type, 0) for entry in recent_rewards]
            stats[reward_type] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return stats
    
    def _compute_alpha_progression(self) -> Dict[str, Any]:
        """Compute alpha value progression."""
        if not self.metrics_history['alpha_values']:
            return {}
        
        recent_alphas = self.metrics_history['alpha_values'][-50:]  # Last 50 steps
        alpha_s_values = [entry['alpha_s'] for entry in recent_alphas]
        alpha_c_values = [entry['alpha_c'] for entry in recent_alphas]
        
        return {
            'alpha_s': {
                'current': alpha_s_values[-1] if alpha_s_values else 0,
                'trend': np.polyfit(range(len(alpha_s_values)), alpha_s_values, 1)[0] if len(alpha_s_values) > 1 else 0
            },
            'alpha_c': {
                'current': alpha_c_values[-1] if alpha_c_values else 0,
                'trend': np.polyfit(range(len(alpha_c_values)), alpha_c_values, 1)[0] if len(alpha_c_values) > 1 else 0
            }
        }
    
    def _compute_creativity_trends(self) -> Dict[str, Any]:
        """Compute creativity score trends."""
        if not self.metrics_history['creativity_scores']:
            return {}
        
        recent_creativity = self.metrics_history['creativity_scores'][-20:]  # Last 20 evaluations
        avg_scores = [entry['avg_score'] for entry in recent_creativity]
        
        return {
            'current_avg': avg_scores[-1] if avg_scores else 0,
            'trend': np.polyfit(range(len(avg_scores)), avg_scores, 1)[0] if len(avg_scores) > 1 else 0,
            'improvement': avg_scores[-1] - avg_scores[0] if len(avg_scores) > 1 else 0
        }
    
    def _compute_tr_rpg_stability(self) -> Dict[str, Any]:
        """Compute TR-RPG training stability metrics."""
        if not self.metrics_history['tr_rpg_metrics']:
            return {}
        
        recent_metrics = self.metrics_history['tr_rpg_metrics'][-10:]  # Last 10 steps
        
        # Compute KL divergence stability
        kl_stability = {}
        for policy in ['propose', 'solve', 'critique']:
            kl_values = [entry['kl_divergences'].get(policy, 0) for entry in recent_metrics]
            if kl_values:
                kl_stability[policy] = {
                    'mean': np.mean(kl_values),
                    'std': np.std(kl_values),
                    'stability_score': 1.0 / (1.0 + np.std(kl_values))  # Higher is more stable
                }
        
        return {
            'kl_stability': kl_stability,
            'overall_stability': np.mean([v['stability_score'] for v in kl_stability.values()]) if kl_stability else 0
        }


class CodeIORewardManager():
    """The reward manager."""
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_examine: int,
        split: str,
        reward_fn_extraction_type: str,
        math_metric: str,
        splitter: str,
        output_path: str,
        generation_reward_config: Dict[str, Any],
        debug: bool = False,
        max_prompt_length: int = 8192,
        valid_program_filter: str = 'all',
        batched_estimate: bool = False,
        extract_code_block: bool = True,
        num_inputs: int = 10,
        code_f_reward_type: str = 'accuracy',
        boxed_retry: bool = False,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = partial(custom_evaluate.get_reward, math_metric=math_metric, boxed_retry=boxed_retry)
        self.reward_fn_extraction_type = reward_fn_extraction_type
        self.split = split
        self.splitter = splitter
        self.output_path = output_path
        self.max_prompt_length = max_prompt_length
        self.generation_reward_config = generation_reward_config
        self.valid_program_filter = valid_program_filter
        self.batched_estimate = batched_estimate
        self.debug = debug
        self.extract_code_block = extract_code_block
        self.use_original_code_as_ref = generation_reward_config.use_original_code_as_ref
        self.num_inputs = num_inputs
        self.code_f_reward_type = code_f_reward_type
        self.boxed_retry = boxed_retry
        
        # Initialize CRSP components
        self.critique_manager = CritiqueManager(tokenizer)
        self.alpha_scheduler = AlphaDecayScheduler()
        self.crsp_logger = CRSPLogger(output_path, log_level="INFO")
        
        # Memory optimization settings
        self.reward_batch_size = getattr(generation_reward_config, 'reward_batch_size', 16)
        self.enable_streaming = getattr(generation_reward_config, 'enable_streaming', True)
        
        # Pre-allocated tensors for memory efficiency
        self._reward_buffers = {}
        self._token_cache = {}  # Cache for tokenized content
        
        # Initialize running statistics for reward normalization
        self.running_stats = {
            'correctness': {'mean': 0.5, 'std': 0.5},
            'length': {'mean': 0.5, 'std': 0.3},
            'creativity': {'mean': 0.5, 'std': 0.3},
            'solver_reward': {'mean': 0.5, 'std': 0.3},
            'critique_reward': {'mean': 0.5, 'std': 0.3}
        }
        
        # Initialize step counter for logging
        self.current_step = 0
        self.total_steps = 1000  # Will be updated during training
    
    def update_training_progress(self, step: int, total_steps: int):
        """Update training progress for logging and alpha decay."""
        self.current_step = step
        self.total_steps = total_steps
    
    def log_comprehensive_metrics(self, step: int, batch_rewards: List[Dict[str, float]], 
                                 batch_solutions: List[str], tr_rpg_metrics: Dict[str, Any] = None):
        """
        Log comprehensive CRSP metrics for a training batch.
        
        Args:
            step: Current training step
            batch_rewards: List of reward dictionaries for each sample in batch
            batch_solutions: List of solution texts for each sample
            tr_rpg_metrics: Optional TR-RPG specific metrics
        """
        if not batch_rewards:
            return
        
        # Extract metrics from batch
        correctness_scores = [r.get('correctness', 0) for r in batch_rewards]
        length_rewards = [r.get('length_reward', 0) for r in batch_rewards]
        creativity_scores = [r.get('creativity_reward', 0) for r in batch_rewards]
        solver_rewards = [r.get('solver_reward', 0) for r in batch_rewards]
        critique_rewards = [r.get('critique_reward', 0) for r in batch_rewards]
        
        # Get alpha values for this step
        alpha_s, alpha_c = self.alpha_scheduler.compute_alpha_decay(step, self.total_steps)
        
        # Compute token counts for length analysis
        token_counts = []
        for solution in batch_solutions:
            tokens = self.tokenizer.encode(solution)
            token_counts.append(len(tokens))
        
        # Log reward breakdown
        avg_rewards = {
            'correctness': np.mean(correctness_scores),
            'length_reward': np.mean(length_rewards),
            'creativity_reward': np.mean(creativity_scores),
            'solver_reward': np.mean(solver_rewards),
            'critique_reward': np.mean(critique_rewards)
        }
        self.crsp_logger.log_reward_breakdown(step, avg_rewards, alpha_s, alpha_c)
        
        # Log creativity evaluation
        rationales = [f"Score: {score:.3f}" for score in creativity_scores]  # Simplified rationales
        self.crsp_logger.log_creativity_evaluation(step, creativity_scores, rationales)
        
        # Log length rewards
        self.crsp_logger.log_length_rewards(step, length_rewards, token_counts)
        
        # Log TR-RPG metrics if provided
        if tr_rpg_metrics:
            kl_divergences = tr_rpg_metrics.get('kl_divergences', {})
            importance_weights = tr_rpg_metrics.get('importance_weights', {})
            policy_losses = tr_rpg_metrics.get('policy_losses', {})
            self.crsp_logger.log_tr_rpg_metrics(step, kl_divergences, importance_weights, policy_losses)
        
        # Save metrics summary periodically
        if step % 100 == 0:
            self.crsp_logger.save_metrics_summary(step)
        
        # Update running statistics for normalization
        self._update_running_stats(avg_rewards)
    
    def _update_running_stats(self, rewards: Dict[str, float], momentum: float = 0.99):
        """Update running statistics for reward normalization."""
        for reward_type, value in rewards.items():
            if reward_type in self.running_stats:
                # Exponential moving average
                self.running_stats[reward_type]['mean'] = (
                    momentum * self.running_stats[reward_type]['mean'] + 
                    (1 - momentum) * value
                )
                
                # Update standard deviation estimate
                diff = value - self.running_stats[reward_type]['mean']
                self.running_stats[reward_type]['std'] = (
                    momentum * self.running_stats[reward_type]['std'] + 
                    (1 - momentum) * abs(diff)
                )
    
    def compute_crsp_rewards(self, data_dict: Dict, correctness_score: float, 
                           model=None) -> Dict[str, float]:
        """
        Compute CRSP-specific rewards including length and creativity rewards.
        
        Args:
            data_dict: Data dictionary containing generation and metadata
            correctness_score: Binary correctness score (0 or 1)
            model: Optional model for critique evaluation
            
        Returns:
            Dictionary containing all CRSP reward components
        """
        generation = data_dict.get('generation', '')
        
        # Get alpha values for current step
        alpha_s, alpha_c = self.alpha_scheduler.compute_alpha_decay(
            self.current_step, self.total_steps
        )
        
        # Compute length reward
        length_reward = self.compute_length_reward(generation, self.tokenizer)
        
        # Compute creativity reward
        creativity_reward = self.critique_manager.compute_creativity_reward(generation, model)
        
        # Compute integrated rewards
        solver_reward = alpha_s * correctness_score + (1 - alpha_s) * length_reward
        
        # For critique reward, use agreement as placeholder (can be enhanced)
        agreement_score = 1.0 if correctness_score > 0.5 else 0.0
        critique_reward = alpha_c * agreement_score + (1 - alpha_c) * creativity_reward
        
        # Create comprehensive reward dictionary
        crsp_rewards = {
            'correctness': correctness_score,
            'length_reward': length_reward,
            'creativity_reward': creativity_reward,
            'solver_reward': solver_reward,
            'critique_reward': critique_reward,
            'alpha_s': alpha_s,
            'alpha_c': alpha_c,
            'agreement_score': agreement_score
        }
        
        # Normalize rewards for training stability
        normalized_rewards = self.normalize_and_clip_rewards(crsp_rewards, self.running_stats)
        
        return normalized_rewards
    
    def integrate_crsp_rewards_with_existing(self, existing_rewards: Dict[str, float], 
                                           crsp_rewards: Dict[str, float]) -> Dict[str, float]:
        """
        Integrate CRSP rewards with existing reward computation for backward compatibility.
        
        Args:
            existing_rewards: Existing reward dictionary
            crsp_rewards: CRSP-specific rewards
            
        Returns:
            Integrated reward dictionary
        """
        # Start with existing rewards
        integrated_rewards = existing_rewards.copy()
        
        # Add CRSP-specific rewards
        integrated_rewards.update(crsp_rewards)
        
        # Use solver reward as the main reward signal for training
        # This maintains compatibility while adding CRSP enhancements
        if 'solver_reward' in crsp_rewards:
            integrated_rewards['reward'] = crsp_rewards['solver_reward']
        
        # Add critique reward as auxiliary signal
        if 'critique_reward' in crsp_rewards:
            integrated_rewards['critique'] = crsp_rewards['critique_reward']
        
        return integrated_rewards
    
    def compute_integrated_rewards(self, solution_text: str, correctness: float, step: int, total_steps: int, model=None) -> Tuple[float, float]:
        """
        Compute integrated CRSP rewards with alpha decay schedule.
        
        Args:
            solution_text: The solution text to evaluate
            correctness: Binary correctness score (0 or 1)
            step: Current training step
            total_steps: Total training steps
            model: Optional model for critique evaluation
            
        Returns:
            Tuple of (solver_reward, critique_reward)
        """
        # Get alpha values from decay schedule
        alpha_s, alpha_c = self.alpha_scheduler.compute_alpha_decay(step, total_steps)
        
        # Compute length reward
        length_reward = self.compute_length_reward(solution_text, self.tokenizer)
        
        # Compute creativity reward
        creativity_reward = self.critique_manager.compute_creativity_reward(solution_text, model)
        
        # Compute integrated rewards
        solver_reward = alpha_s * correctness + (1 - alpha_s) * length_reward
        
        # For critique reward, we use agreement as a placeholder (can be enhanced later)
        agreement_score = 1.0 if correctness > 0.5 else 0.0  # Simple agreement heuristic
        critique_reward = alpha_c * agreement_score + (1 - alpha_c) * creativity_reward
        
        return solver_reward, critique_reward
    
    def normalize_and_clip_rewards(self, rewards: Dict[str, float], running_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Normalize and clip rewards for training stability.
        
        Args:
            rewards: Dictionary of reward values
            running_stats: Running statistics for normalization
            
        Returns:
            Dictionary of normalized and clipped rewards
        """
        normalized_rewards = {}
        
        for reward_type, reward_value in rewards.items():
            if reward_type in running_stats:
                mu = running_stats[reward_type]['mean']
                sigma = running_stats[reward_type]['std']
                epsilon = 1e-8
                
                # Normalize
                normalized_reward = (reward_value - mu) / (sigma + epsilon)
                
                # Clip for stability
                if reward_type == 'creativity':
                    normalized_reward = max(-2.0, min(2.0, normalized_reward))
                
                normalized_rewards[reward_type] = normalized_reward
            else:
                normalized_rewards[reward_type] = reward_value
        
        return normalized_rewards

    def compute_length_reward(self, solution_text: str, tokenizer: AutoTokenizer, max_length: int = 2048, penalty_threshold: int = 4096) -> float:
        """
        Compute length-based reward following RLSP logic with memory optimization.
        Encourages thoughtful reasoning chains while preventing excessive verbosity.
        
        Args:
            solution_text: The solution text to evaluate
            tokenizer: Tokenizer to count tokens
            max_length: Maximum desired reasoning length
            penalty_threshold: Length threshold after which penalty is applied
            
        Returns:
            Length reward score between 0 and 1
        """
        import math
        
        # Check cache first for memory efficiency
        text_hash = hash(solution_text)
        if hasattr(self, '_token_cache') and text_hash in self._token_cache:
            token_count = self._token_cache[text_hash]
        else:
            # Extract thinking content efficiently without regex compilation
            reasoning_text = self.extract_think_content_fast(solution_text)
            
            # Efficient tokenization - avoid creating full token list if possible
            if hasattr(tokenizer, 'count_tokens'):
                token_count = tokenizer.count_tokens(reasoning_text)
            else:
                # Use efficient encoding without storing tokens
                encoding = tokenizer(
                    reasoning_text, 
                    add_special_tokens=False,
                    return_tensors=None,
                    return_length=True
                )
                token_count = len(encoding['input_ids']) if 'input_ids' in encoding else 0
            
            # Cache result with size limit
            if hasattr(self, '_token_cache'):
                self._token_cache[text_hash] = token_count
                
                # Limit cache size to prevent memory growth
                if len(self._token_cache) > 1000:
                    # Remove oldest entries (simple FIFO)
                    oldest_keys = list(self._token_cache.keys())[:500]
                    for key in oldest_keys:
                        del self._token_cache[key]
        
        if token_count == 0:
            return 0.0
        
        # Use lookup table for common logarithms to avoid repeated computation
        if not hasattr(self, '_log_lookup'):
            self._log_lookup = {i: math.log(i) for i in range(1, 10000)}
            self._log_max_length = math.log(max_length)
        
        # Efficient logarithmic scaling
        log_tokens = self._log_lookup.get(token_count, math.log(token_count))
        length_reward = min(1.0, log_tokens / self._log_max_length)
        
        # Fast penalty computation
        if token_count > penalty_threshold:
            penalty_exp = -(token_count - penalty_threshold) / penalty_threshold
            # Avoid expensive exp computation for very negative values
            length_reward *= math.exp(penalty_exp) if penalty_exp > -10 else 0.0
        
        return length_reward
    
    def extract_think_content_fast(self, solution_text: str) -> str:
        """
        Fast extraction of think content without regex compilation.
        
        Args:
            solution_text: The solution text to process
            
        Returns:
            Extracted thinking content or full text if no tags found
        """
        start_tag = '<think>'
        end_tag = '</think>'
        
        start_idx = solution_text.find(start_tag)
        if start_idx == -1:
            return solution_text.strip()
        
        end_idx = solution_text.find(end_tag, start_idx)
        if end_idx == -1:
            return solution_text[start_idx + len(start_tag):].strip()
        
        return solution_text[start_idx + len(start_tag):end_idx].strip()

    @staticmethod
    def extract_input_output(extracted_content: str, return_input: bool = True, return_output: bool = False) -> Tuple[str, str]:
        input_pattern = r"```input\s*\n?(.*?)\n?```"
        output_pattern = r"```output\s*\n?(.*?)\n?```"
        assert not (return_input and return_output), "Cannot return both input and output"
        assert return_input or return_output, "Must return at least one of input or output"

        # Use flags for case-insensitive matching and dotall
        flags = re.DOTALL | re.IGNORECASE
        if return_input:
            input_matches = list(re.finditer(input_pattern, extracted_content, flags))
            if not input_matches:
                # Try alternative pattern without explicit input block
                input_matches = list(re.finditer(r"# Input:\s*(.*?)(?=\n```|$)", extracted_content, flags))
            if not input_matches:
                # Match input() function call and preserve quotes
                input_matches = list(re.finditer(r'input\s*\((.*?)\)', extracted_content, flags))
            if not input_matches:
                # Match <input> tag with optional closing tag, strip spaces
                input_matches = list(re.finditer(r"<input>\s*(.*?)(?:</input>|\s*$)", extracted_content, flags))
            if not input_matches:
                # Match "The input is" pattern case-insensitively
                input_matches = list(re.finditer(r"the input is\s*(.*?)\.?$", extracted_content, flags))
            # if still no input matches, use the extracted answer as the input
            # Don't strip() here to preserve quotes
            input_snippet = input_matches[-1].group(1) if input_matches else extracted_content
            return input_snippet

        if return_output:
            output_matches = list(re.finditer(output_pattern, extracted_content, flags))
            if not output_matches:
                # Try alternative pattern without explicit output block
                output_matches = list(re.finditer(r"# Output:\s*(.*?)(?=\n```|$)", extracted_content, flags))
            if not output_matches:
                # Match output() function call and preserve quotes
                output_matches = list(re.finditer(r'output\s*\((.*?)\)', extracted_content, flags))
            if not output_matches:
                # Match <output> tag with optional closing tag, strip spaces
                output_matches = list(re.finditer(r"<output>\s*(.*?)(?:</output>|\s*$)", extracted_content, flags))
            if not output_matches:
                # Match "The output is" pattern case-insensitively, strip space after "is" and period at end
                output_matches = list(re.finditer(r"the output is\s*(.*?)\.?$", extracted_content, flags))
            # if still no output matches, use the extracted answer as the output
            output_snippet = output_matches[-1].group(1) if output_matches else extracted_content
            return output_snippet

    def _get_data_dict(self, data_item: DataProtoItem, problem_type: str, executor, banned_words: List[str], uid: str, banned_assertion_keywords: List[str]) -> Dict:
        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        sequences_str = self.tokenizer.decode(sequences)

        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        data_source = data_item.non_tensor_batch['data_source']
        extra_info = data_item.non_tensor_batch['extra_info']
        non_special_tokens_sequences_str = self.tokenizer.decode(self.tokenizer.encode(sequences_str), skip_special_tokens=True)
        
        generation = non_special_tokens_sequences_str.split(self.splitter)[1].strip().strip('\"\'')
        extracted_content = extract_answer(generation, self.reward_fn_extraction_type, boxed_retry=self.boxed_retry)
        thought = extract_thought(generation)

        data_dict = {
            'generation': generation,
            'data_source': data_source,
            'ground_truth': ground_truth,
            'extra_info': extra_info,
            'non_special_tokens_sequences_str': non_special_tokens_sequences_str,
            'valid_response_length': valid_response_length,
            'extracted_content': extracted_content,
            'thought': thought,
            'uid': uid,
        }
        if problem_type.startswith('gen'):
            data_dict['references'] = [ref['snippet'] for ref in data_item.non_tensor_batch['extra_info']['chosen_references']]
            if problem_type != 'gen_code_f':
                data_dict['composite_functions'] = data_item.non_tensor_batch['extra_info']['composite_functions'].tolist()
            else:
                data_dict['imports'] = [ref['imports'] for ref in data_item.non_tensor_batch['extra_info']['chosen_references']]
            if self.use_original_code_as_ref:
                data_dict['original_references'] = [ref['original_snippet'] for ref in data_item.non_tensor_batch['extra_info']['chosen_references']]
        elif problem_type.startswith('pred') and 'code_f' not in problem_type:
            data_dict['program'] = data_item.non_tensor_batch['problem']
            data_dict['input'] = data_item.non_tensor_batch['extra_info']['input']
            data_dict['output'] = data_item.non_tensor_batch['extra_info']['output']
            data_dict['imports'] = data_item.non_tensor_batch['extra_info'].get('imports', [])
        elif problem_type.startswith('pred') and 'code_f' in problem_type:
            data_dict['program'] = data_item.non_tensor_batch['problem']
            data_dict['given_inputs'] = data_item.non_tensor_batch['extra_info']['given_inputs']
            data_dict['given_outputs'] = data_item.non_tensor_batch['extra_info']['given_outputs']
            data_dict['hidden_inputs'] = data_item.non_tensor_batch['extra_info']['hidden_inputs']
            data_dict['hidden_outputs'] = data_item.non_tensor_batch['extra_info']['hidden_outputs']
            data_dict['message'] = data_item.non_tensor_batch['extra_info']['message']
            data_dict['imports'] = data_item.non_tensor_batch['extra_info'].get('imports', [])

        # if QA task, we only need to check the format
        if problem_type is None:
            format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
            data_dict['format_score'] = format_score
            return data_dict
        # first go through, we only checking the format
        elif problem_type.startswith('gen') and 'code_f' not in problem_type:
            success, result = parse_code_input_output(
                extracted_content,
                parse_output=False,
                remove_after_return=self.generation_reward_config.remove_after_return and self.split == 'train',
                remove_comments=self.generation_reward_config.remove_comments and self.split == 'train',
                remove_print=self.generation_reward_config.remove_print and self.split == 'train',
                reject_multiple_functions=self.generation_reward_config.reject_multiple_functions,
                f_replace_location=self.generation_reward_config.f_replace_location,
                reject_test_input_in_code=self.generation_reward_config.reject_test_input_in_code,
                code_location=self.generation_reward_config.code_location,
            )
            if len(data_dict['composite_functions']) > 0 and success:
                # first, check if the composite function names are redefined in the code, which we do not allow
                success = check_no_definitions(result['code'], [f'g_{i}' for i in range(len(data_dict['composite_functions']))])
                if not success: # if the composite function names are redefined, we do not allow the code
                    data_dict['code_validity'] = False
                    data_dict['format_score'] = 0.
                    return data_dict

                composite_imports = '\n'.join(
                    '\n'.join(list(d['imports'])) if list(d['imports']) else '' for d in data_dict['composite_functions']
                ).strip()

                composite_snippets = '\n\n'.join(d['snippet'] for d in data_dict['composite_functions']).strip()

                # cache the original code
                result['original_code'] = result['code']

                result['code'] = f"{composite_imports}\n\n{composite_snippets}\n\n{result['code']}".strip()
                # TODO: composite function check
                success = check_composite_function(
                    code = result['code'],
                    composite_functions = [d['snippet'] for d in data_dict['composite_functions']],
                )
            if success:
                code_validity, output = executor.check_all(
                    code=result['code'],
                    inputs=result['input'],
                    banned_keywords=banned_words,
                    check_determinism=True,
                    imports=list(set(result['imports'])),
                    check_error=problem_type == 'gen_code_e',
                    banned_keywords_for_errors_and_exceptions=banned_assertion_keywords,
                )
                if not code_validity:
                    data_dict['code_validity'] = False
                    data_dict['format_score'] = 0.
                    return data_dict
                # means the code is valid, we append any good programs, but we eval format separately
                data_dict['answer'] = {
                    'snippet': result['code'],
                    'original_snippet': result['original_code'] if 'original_code' in result else result['code'],
                    'input': result['input'],
                    'output': output,
                    'imports': result['imports'],
                    'thought': thought,
                    'composite_functions': data_dict['composite_functions']
                }
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
                data_dict['format_score'] = format_score
                data_dict['code_validity'] = True
                return data_dict
            else:
                data_dict['code_validity'] = False
                data_dict['format_score'] = 0.
                return data_dict

        elif problem_type == 'gen_code_f':
            success, result = parse_inputs_message(
                extracted_content,
                num_inputs=self.num_inputs,
            )
            if success and len(result['inputs']) == self.num_inputs: # for code_f, we need to ensure the number of inputs is correct
                outputs = []
                for inpt in result['inputs']:
                    code_validity, output = executor.check_all(
                        code=data_dict['references'][0],
                        inputs=inpt,
                        banned_keywords=[],
                        check_determinism=True,
                        imports=data_dict['imports'][0],
                        check_error=False,
                        banned_keywords_for_errors_and_exceptions=[],
                    )
                    if not code_validity:
                        data_dict['code_validity'] = False
                        data_dict['format_score'] = 0.
                        return data_dict
                    outputs.append(output)
                data_dict['answer'] = {
                    'snippet': data_dict['references'][0],
                    'inputs': result['inputs'],
                    'outputs': outputs,
                    'message': result['message'],
                    'imports': data_dict['imports'][0],
                    'thought': thought,
                }
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
                data_dict['format_score'] = format_score
                data_dict['code_validity'] = True
                return data_dict
            else:
                data_dict['code_validity'] = False
                data_dict['format_score'] = 0.
                return data_dict

        # if prediction is the task
        elif problem_type.startswith('pred'):
            # Check required blocks
            if problem_type.endswith('code_i'): # parse input
                input_snippet = self.extract_input_output(extracted_content, return_input=True, return_output=False) \
                    if self.extract_code_block else extracted_content
                if input_snippet is None:
                    data_dict['format_score'] = 0.
                    return data_dict
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
                data_dict['format_score'] = format_score
                data_dict['answer'] = input_snippet
                return data_dict
            elif problem_type.endswith('code_o') or problem_type.endswith('code_e'): #  parse output, code_e format is same as code_o
                output_snippet = self.extract_input_output(extracted_content, return_input=False, return_output=True) \
                    if self.extract_code_block else extracted_content
                if output_snippet is None:
                    data_dict['format_score'] = 0.
                    return data_dict
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
                data_dict['format_score'] = format_score
                data_dict['answer'] = output_snippet
                return data_dict
            elif problem_type.endswith('code_f'):
                success, code_snippet = parse_code_function(extracted_content)
                if not success:
                    data_dict['format_score'] = 0.
                    return data_dict
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
                data_dict['format_score'] = format_score
                data_dict['answer'] = {
                    'snippet': code_snippet,
                    'given_inputs': data_dict['given_inputs'],
                    'given_outputs': data_dict['given_outputs'],
                    'hidden_inputs': data_dict['hidden_inputs'],
                    'hidden_outputs': data_dict['hidden_outputs'],
                    'message': data_dict['message'],
                    'imports': data_dict['imports'],
                    'thought': thought,
                    'gold_program': data_dict['program'],
                }
                return data_dict
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")
        else:
            raise ValueError(f"Invalid problem type: {problem_type}")

    def __call__(
        self,
        data: DataProto,
        problem_type: str = None,
        executor = None,
        rollout_actor_wg = None,
        banned_words: List[str] = [],
        banned_assertion_keywords: List[str] = [],
        n_samples: int = 1,
        input_type_counters: Dict[str, Dict[str, int]] = None,
        output_type_counters: Dict[str, Dict[str, int]] = None,
        error_type_counters: Dict[str, Dict[str, int]] = None,
    ) -> Tuple[torch.Tensor, Dict, List[Dict], List[Dict]]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        all_scores = defaultdict(list)
        data_dicts = []
        valid_programs = [] # for gen tasks, we need to store the valid programs for later use, ignore this if prediction task
        correct_predictions = []
        uids = np.array([str(uuid.uuid4()) for _ in range(len(data))], dtype=object)
        if problem_type is None:
            problem_types = [d.non_tensor_batch['extra_info']['metric'] for d in data]
            problem_type = 'pred' # dummy set
        else:
            problem_types = [problem_type] * len(data)
        PrettyPrinter.section_header("Getting Data Dicts")
        for i in range(len(data)): # get format score
            data_dict = self._get_data_dict(data[i], problem_types[i], executor, banned_words, uids[i], banned_assertion_keywords)
            data_dicts.append(data_dict)

        if problem_type.startswith('gen') and rollout_actor_wg is not None: # get generation rewards
            PrettyPrinter.section_header("Generating Rewards for Generation Tasks")
            rewards, valid_programs = self._get_problem_generator_rewards_and_valid_programs(
                data_dicts=data_dicts,
                problem_type=problem_type,
                n_samples=n_samples,
                rollout_actor_wg=rollout_actor_wg,
                executor=executor,
                input_type_counters=input_type_counters,
                output_type_counters=output_type_counters,
                error_type_counters=error_type_counters,
            )
            PrettyPrinter.section_header("Combining Rewards for Generation Tasks")
            
            # Display detailed reward breakdown for transparency
            PrettyPrinter.status("REWARD", "Detailed reward breakdown:", "info")
            for uid in rewards:
                reward_data = rewards[uid]
                PrettyPrinter.status("REWARD", f"UID {uid[:8]}...", "info")
                PrettyPrinter.status("REWARD", f"  ├─ Accuracy: {reward_data['accuracy']:.4f}", "info")
                if 'complexity' in reward_data:
                    PrettyPrinter.status("REWARD", f"  ├─ Complexity: {reward_data['complexity']:.4f}", "info")
                if 'mean_edit_distance' in reward_data:
                    PrettyPrinter.status("REWARD", f"  ├─ Edit Distance: {reward_data['mean_edit_distance']:.4f}", "info")
                if 'halstead' in reward_data:
                    PrettyPrinter.status("REWARD", f"  ├─ Halstead: {reward_data['halstead']:.4f}", "info")
                if 'type_counts' in reward_data:
                    PrettyPrinter.status("REWARD", f"  ├─ Type Diversity: {reward_data['type_counts']:.4f}", "info")
                if 'input_type_counts' in reward_data:
                    PrettyPrinter.status("REWARD", f"  ├─ Input Type Diversity: {reward_data['input_type_counts']:.4f}", "info")
                if 'output_type_counts' in reward_data:
                    PrettyPrinter.status("REWARD", f"  └─ Output Type Diversity: {reward_data['output_type_counts']:.4f}", "info")
            
            for i in range(len(data_dicts)):
                uid = data_dicts[i]['uid']
                valid_response_length = data_dicts[i]['valid_response_length']
                acc_reward = rewards[uid]['accuracy']
                format_reward = data_dicts[i]['format_score']
                
                PrettyPrinter.status("REWARD", f"Sample {i+1}: Format={format_reward:.2f}, Accuracy={acc_reward:.4f}", "info")
                
                if format_reward > 0:
                    if acc_reward > 0:
                        # Helper function for safe reward combination
                        def _combine_rewards(acc, intrinsic_components, method):
                            components = [c for c in intrinsic_components if c is not None]

                            if method == 'sum':
                                return acc + sum(components) if components else acc
                            elif method == 'multiply':
                                return acc * np.prod([c for c in components]) if components else acc
                            elif method == 'sum_multiply':
                                return acc + np.prod([c for c in components]) if components else acc
                            elif method == 'multiply_sum':
                                return acc * sum(components) if components else acc
                            else:
                                raise ValueError(f"Unknown combination method: {method}")

                        intrinsic_reward_components = []
                        if problem_type.endswith('code_f'):
                            if self.generation_reward_config.f_input_answer_diversity_reward.enabled:
                                intrinsic_reward_components.append(min(self.generation_reward_config.f_input_answer_diversity_reward.coef * rewards[uid]['input_type_counts'],
                                    self.generation_reward_config.f_input_answer_diversity_reward.max))
                            if self.generation_reward_config.f_output_answer_diversity_reward.enabled:
                                intrinsic_reward_components.append(min(self.generation_reward_config.f_output_answer_diversity_reward.coef * rewards[uid]['output_type_counts'],
                                    self.generation_reward_config.f_output_answer_diversity_reward.max))
                        else:
                            if self.generation_reward_config.complexity_reward.enabled:
                                intrinsic_reward_components.append(min(self.generation_reward_config.complexity_reward.coef * rewards[uid]['complexity'],
                                    self.generation_reward_config.complexity_reward.max))
                            if self.generation_reward_config.mean_edit_distance_reward.enabled:
                                intrinsic_reward_components.append(min(self.generation_reward_config.mean_edit_distance_reward.coef * rewards[uid]['mean_edit_distance'],
                                    self.generation_reward_config.mean_edit_distance_reward.max))
                            if self.generation_reward_config.halstead_reward.enabled:
                                intrinsic_reward_components.append(min(self.generation_reward_config.halstead_reward.coef * rewards[uid]['halstead'],
                                    self.generation_reward_config.halstead_reward.max))
                            if self.generation_reward_config.answer_diversity_reward.enabled:
                                intrinsic_reward_components.append(min(self.generation_reward_config.answer_diversity_reward.coef * rewards[uid]['type_counts'],
                                    self.generation_reward_config.answer_diversity_reward.max))

                        # Compute CRSP rewards (length, creativity, critique)
                        generation_text = data_dicts[i].get('generation', '')
                        crsp_rewards = self.compute_crsp_rewards(
                            data_dict={'generation': generation_text}, 
                            correctness_score=acc_reward
                        )
                        
                        # Memory-efficient reward computation - avoid expensive CRSP computation during training
                        final_reward = _combine_rewards(acc_reward, intrinsic_reward_components, self.generation_reward_config.intrinsic_combine_method)
                        reward_tensor[i, valid_response_length - 1] = final_reward
                        
                        # Display lightweight reward computation (only for first few samples to avoid spam)
                        if i < 3:  # Only show first 3 samples to reduce memory and logging overhead
                            PrettyPrinter.status("REWARD", f"Sample {i+1} FINAL REWARD: {final_reward:.4f}", "info")
                            PrettyPrinter.status("REWARD", f"  ├─ Base Accuracy: {acc_reward:.4f}", "info")
                            PrettyPrinter.status("REWARD", f"  ├─ Intrinsic Components: {intrinsic_reward_components}", "info")
                            PrettyPrinter.status("REWARD", f"  └─ Final Combined: {final_reward:.4f}", "info")
                        elif i == 3:
                            PrettyPrinter.status("REWARD", f"... (showing first 3 samples only to save memory)", "info")
                        
                        # Compute lightweight CRSP metrics only for logging (not for training)
                        if i == 0:  # Only compute once per batch to save memory
                            alpha_s, alpha_c = self.alpha_scheduler.compute_alpha_decay(
                                self.current_step, self.total_steps
                            )
                            PrettyPrinter.status("CRSP", f"ALPHA SCHEDULE: α_s={alpha_s:.4f}, α_c={alpha_c:.4f}, Progress={self.current_step/self.total_steps:.2%}", "info")
                    else:
                        reward_tensor[i, valid_response_length - 1] = -0.5
                else:
                    reward_tensor[i, valid_response_length - 1] = -1.0
            all_scores['accuracy'] = [rewards[uid]['accuracy'] for uid in rewards]
            all_scores['format_score'] = [data_dicts[i]['format_score'] for i in range(len(data))]
            if 'code_f' not in problem_type:
                all_scores['answer_diversity'] = [rewards[uid]['type_counts'] for uid in rewards]
                all_scores['complexity'] = [rewards[uid]['complexity'] for uid in rewards]
                all_scores['mean_edit_distance'] = [rewards[uid]['mean_edit_distance'] for uid in rewards]
                all_scores['halstead'] = [rewards[uid]['halstead'] for uid in rewards]
            else:
                all_scores['input_answer_diversity'] = [rewards[uid]['input_type_counts'] for uid in rewards]
                all_scores['output_answer_diversity'] = [rewards[uid]['output_type_counts'] for uid in rewards]
        elif problem_type.startswith('pred'): # get prediction rewards
            PrettyPrinter.section_header("Getting Prediction Rewards")
            PrettyPrinter.status("REWARD", f"Processing {len(data_dicts)} prediction samples for {problem_type}", "info")
            all_scores['none_count'] = 0
            acc_rewards = []
            for i, data_dict in enumerate(data_dicts):
                valid_response_length = data_dict['valid_response_length']
                imports = data_dict['imports']
                input_output_accs = []  # Initialize for all problem types
                if not problem_type.endswith('code_f'):
                    answer = data_dict['answer']
                    gold_input = data_dict['input']
                    gold_output = data_dict['output']
                    program = data_dict['program']
                else:
                    hidden_inputs = data_dict['hidden_inputs']
                    hidden_outputs = data_dict['hidden_outputs']
                if not data_dicts[i]['format_score']: # early stop if the format is not correct
                    acc_reward = 0.
                elif problem_types[i].endswith('code_i'):
                    acc_reward = executor.eval_input_prediction(code=program, gold_output=gold_output, agent_input=answer, imports=list(set(imports)))
                    # problematic, but we did not encounter too much of this
                    if acc_reward is None:
                        all_scores['none_count'] += 1
                        acc_reward = 0.
                        print(f"error in pred_code_i, not in [0, 1], acc_reward={acc_reward}\nprogram:\n{program}\n---\nanswer:\n{answer}\n---\nimports:\n{imports}\n---\n")
                    if acc_reward > 0.0:
                        correct_predictions.append(data_dict)
                elif problem_types[i].endswith('code_o'):
                    acc_reward = executor.eval_output_prediction(code=program, gold_output=gold_output, agent_output=answer, imports=list(set(imports)))
                    # problematic, but we did not encounter too much of this
                    if acc_reward is None:
                        all_scores['none_count'] += 1
                        acc_reward = 0.
                        print(f"error in pred_code_o, not in [0, 1], acc_reward={acc_reward}\nprogram:\n{program}\n---\nanswer:\n{answer}\n---\nimports:\n{imports}\n---\n")
                    if acc_reward > 0.0:
                        correct_predictions.append(data_dict)
                elif problem_types[i].endswith('code_e'): # string matching for errors
                    answer = answer.split(' ')[0].split(':')[0]
                    if answer.lower() == gold_output.lower():
                        acc_reward = 1.0
                        correct_predictions.append(data_dict)
                    else:
                        acc_reward = 0.0
                elif problem_types[i].endswith('code_f'):
                    input_output_accs = []
                    program = data_dict['answer']['snippet']
                    for inpt, outpt in zip(hidden_inputs, hidden_outputs):
                        input_output_acc = executor.eval_input_prediction(
                            code=program,
                            gold_output=outpt,
                            agent_input=inpt,
                            imports=list(set(imports)),
                        )
                        if input_output_acc is not None:
                            input_output_accs.append(input_output_acc)
                    acc_reward = np.mean(input_output_accs) if input_output_accs else 0.0
                    if self.code_f_reward_type == 'binary':
                        acc_reward = 1.0 if acc_reward == 1.0 else 0.0
                    elif self.code_f_reward_type == 'if_one_correct':
                        acc_reward = 1.0 if acc_reward > 0 else 0.0
                    # note that if code_f_reward_type==accuracy, it is already handled in the above
                    if acc_reward > 0:
                        correct_predictions.append(data_dict)
                else:
                    raise ValueError(f"Invalid problem type: {problem_types[i]}")

                # Display detailed prediction reward
                PrettyPrinter.status("REWARD", f"Sample {i+1} PREDICTION REWARD:", "info")
                PrettyPrinter.status("REWARD", f"  ├─ Problem Type: {problem_types[i]}", "info")
                PrettyPrinter.status("REWARD", f"  ├─ Format Score: {data_dicts[i]['format_score']:.2f}", "info")
                PrettyPrinter.status("REWARD", f"  ├─ Accuracy Reward: {acc_reward:.4f}", "info")
                if problem_types[i].endswith('code_f') and input_output_accs:
                    PrettyPrinter.status("REWARD", f"  ├─ Individual I/O Accuracies: {[f'{acc:.2f}' for acc in input_output_accs]}", "info")
                    PrettyPrinter.status("REWARD", f"  ├─ Reward Type: {self.code_f_reward_type}", "info")
                PrettyPrinter.status("REWARD", f"  └─ Correct: {'Yes' if acc_reward > 0 else 'No'}", "info")

                if self.split == 'train':
                    if data_dicts[i]['format_score'] > 0:
                        if acc_reward > 0:
                            reward_tensor[i, valid_response_length - 1] = acc_reward
                        else:
                            reward_tensor[i, valid_response_length - 1] = -0.5
                    else:
                        reward_tensor[i, valid_response_length - 1] = -1.0
                elif self.split == 'test': # only acc reward for eval
                    if acc_reward > 0:
                        reward_tensor[i, valid_response_length - 1] = 1.0
                    else:
                        reward_tensor[i, valid_response_length - 1] = 0.0
                acc_rewards.append(acc_reward)
            all_scores['accuracy'] = acc_rewards
            all_scores['format_score'] = [data_dicts[i]['format_score'] for i in range(len(data))]
            all_scores['none_ratio'] = all_scores['none_count'] / len(data)
        return reward_tensor, all_scores, valid_programs, correct_predictions

    def _get_problem_generator_rewards_and_valid_programs(
        self,
        data_dicts: List[Dict],
        problem_type: str,
        n_samples: int,
        rollout_actor_wg,
        executor,
        input_type_counters: Dict[str, Dict[str, int]] = None,
        output_type_counters: Dict[str, Dict[str, int]] = None,
        error_type_counters: Dict[str, Dict[str, int]] = None,
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, str]]]:
        """This function uses samples to estimate the accuracy reward for each program, also computes the code complexity and mean edit distance of generated programs.
            Also returns the valid programs using filters.
            Args:
                data_dicts: List[Dict]: A list of data dictionaries.
                problem_type: str: The type of problem.
                n_samples: int: The number of samples to use.
                rollout_actor_wg: RolloutActorWG: The rollout actor.
                executor: PythonExecutor/CodeBoxExecutor: The executor.
                type_counters: Dict[str, Dict[str, int]]: The type counters.
            Returns:
               rewards: Dict[str, Dict[str, float]]: A dictionary of rewards for each program.
               valid_programs: List[Dict[str, str]]: A list of valid programs.
        """
        if problem_type.endswith('code_i'):
            type_counters = input_type_counters
        elif problem_type.endswith('code_o'):
            type_counters = output_type_counters
        elif problem_type.endswith('code_e'):
            type_counters = error_type_counters
        valid_data_dicts = [data_dict for data_dict in data_dicts if data_dict['code_validity']]
        uid2valid_dict_idx = {data_dict['uid']: i for i, data_dict in enumerate(valid_data_dicts)}
        valid_uids = [data_dict['uid'] for data_dict in data_dicts if data_dict['code_validity']]
        invalid_uids = [data_dict['uid'] for data_dict in data_dicts if not data_dict['code_validity']]
        assert len(valid_uids) + len(invalid_uids) == len(data_dicts)
        accuracies = {uid: 1.0 for uid in invalid_uids} # for invalid uids, we give maximum accuracy to the model
        rewards = defaultdict(dict)
        valid_programs = []
        if len(valid_uids) > 0:
            if self.reward_fn_extraction_type.startswith('boxed'):
                instruction_template = boxed_instruction
            elif self.reward_fn_extraction_type.startswith('answer'):
                instruction_template = instruction_following
            elif self.reward_fn_extraction_type.startswith('none'):
                instruction_template = '{}'
            else:
                raise ValueError(f"Invalid instruction type: {self.reward_fn_extraction_type}")
            prompts = []
            if problem_type.endswith('code_i'):
                pt = 'code_i'
            elif problem_type.endswith('code_o'):
                pt = 'code_o'
            elif problem_type.endswith('code_e'):
                pt = 'code_e'
            elif problem_type.endswith('code_f'):
                pt = 'code_f'
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")
            for data_dict in valid_data_dicts:
                if pt == 'code_f':
                    num_given_inputs = len(data_dict['answer']['inputs']) // 2
                    num_given_outputs = len(data_dict['answer']['outputs']) // 2
                    data_dict['answer']['given_inputs'] = data_dict['answer']['inputs'][:num_given_inputs]
                    data_dict['answer']['given_outputs'] = data_dict['answer']['outputs'][:num_given_outputs]
                    data_dict['answer']['hidden_inputs'] = data_dict['answer']['inputs'][num_given_inputs:]
                    data_dict['answer']['hidden_outputs'] = data_dict['answer']['outputs'][num_given_outputs:]
                    io_prompt = instruction_template.format(
                        get_code_problem_predictor_prompt(
                            problem_type=problem_type,
                            snippet=data_dict['answer']['snippet'],
                            message=data_dict['answer']['message'],
                            input_output_pairs=zip(data_dict['answer']['given_inputs'], data_dict['answer']['given_outputs']),
                        )
                    )
                else:
                    io_prompt = instruction_template.format(
                        get_code_problem_predictor_prompt(
                            problem_type=pt,
                            snippet=data_dict['answer']['snippet'],
                            input_args=data_dict['answer']['input'],
                            output=data_dict['answer']['output'],
                        )
                    )
                prompts_dict = {
                    'prompt': [{'role': 'user', 'content': io_prompt}],
                    'uid': data_dict['uid'],
                    'problem': data_dict['answer'],
                    'data_source': data_dict['data_source'],
                    'ground_truth': data_dict['answer']['output'] if pt != 'code_f' else data_dict['answer']['snippet'],
                    'extra_info': data_dict['extra_info'],
                    'program': data_dict['answer']['snippet'],
                    'imports': data_dict['answer']['imports'],
                    'references': data_dict['references'],
                }
                if pt == 'code_f':
                    prompts_dict.update({
                        'given_inputs': data_dict['answer']['given_inputs'],
                        'given_outputs': data_dict['answer']['given_outputs'],
                        'hidden_inputs': data_dict['answer']['hidden_inputs'],
                        'hidden_outputs': data_dict['answer']['hidden_outputs'],
                        'message': data_dict['answer']['message'],
                    })
                else:
                    prompts_dict.update({
                        'input': data_dict['answer']['input'],
                        'output': data_dict['answer']['output'],
                        'original_program': data_dict['answer']['original_snippet'],
                        'composite_functions': data_dict['answer']['composite_functions'],
                    })
                prompts.append(prompts_dict)

            # sampling to estimate the accuracy
            PrettyPrinter.section_header("Sampling to Estimate Accuracy")
            prompts = prompts * n_samples # repeat the prompts n_samples times
            pd.DataFrame(prompts).to_parquet(f'{self.output_path}/temp.parquet') # RLHFDataset expects parquet
            temp_data = RLHFDataset(
                parquet_files=f'{self.output_path}/temp.parquet',
                tokenizer=self.tokenizer,
                prompt_key='prompt',
                max_prompt_length=self.max_prompt_length,
                filter_prompts=True,
                return_raw_chat=False,
                truncation='error'
            )
            os.remove(f'{self.output_path}/temp.parquet') # we do not need this file after we load in the dataset
            sampler = torch.utils.data.SequentialSampler(data_source=temp_data)

            dataloader = torch.utils.data.DataLoader(
                dataset=temp_data,
                batch_size=len(temp_data),
                drop_last=False,
                shuffle=False,
                collate_fn=collate_fn,
                sampler=sampler,
            )
            assert len(dataloader) == 1
            data = next(iter(dataloader))
            batch = DataProto.from_single_dict(data)
            gen_batch = batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': True,
                'validate': True,
            }
            # pad to be divisible by dp_size
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, rollout_actor_wg.world_size)
            output_gen_batch_padded = rollout_actor_wg.generate_sequences(gen_batch_padded)
            # unpad
            output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            batch = batch.union(output_gen_batch)
            batched_responses = []
            for b in batch:
                batch_dict = {
                        'extracted_answers': extract_answer(
                            self.tokenizer.decode(b.batch['responses'], skip_special_tokens=True),
                            self.reward_fn_extraction_type,
                            boxed_retry=self.boxed_retry,
                        ),
                        'uid': b.non_tensor_batch['uid'],
                        'problem': b.non_tensor_batch['problem'],
                        'data_source': b.non_tensor_batch['data_source'],
                        'extra_info': b.non_tensor_batch['extra_info'],
                        'program': b.non_tensor_batch['program'],
                        'references': b.non_tensor_batch['references'],
                        'imports': b.non_tensor_batch['imports'],
                    }
                if pt == 'code_f':
                    batch_dict.update({
                        'given_inputs': b.non_tensor_batch['given_inputs'],
                        'given_outputs': b.non_tensor_batch['given_outputs'],
                        'hidden_inputs': b.non_tensor_batch['hidden_inputs'],
                        'hidden_outputs': b.non_tensor_batch['hidden_outputs'],
                        'message': b.non_tensor_batch['message'],
                    })
                else:
                    batch_dict.update({
                        'input': b.non_tensor_batch['input'],
                        'output': b.non_tensor_batch['output'],
                        'original_program': b.non_tensor_batch['original_program'],
                        'composite_functions': b.non_tensor_batch['composite_functions'].tolist(),
                    })
                batched_responses.append(batch_dict)
            df = pd.DataFrame(batched_responses)

            # estimating accuracy using python executor
            PrettyPrinter.section_header("Estimating Accuracy Using Python Executor")
            for valid_uid in valid_uids:
                df_valid = df[df['uid'] == valid_uid]
                if df_valid.empty: # the prompt got filtered out TODO: check
                    accuracies[valid_uid] = 0.0
                    continue
                if pt != 'code_f':
                    answers = [self.extract_input_output(
                        answer,
                        return_input=problem_type.endswith('code_i'),
                        return_output=(problem_type.endswith('code_o') or problem_type.endswith('code_e')) # code_e output format is same as code_o
                    ) for answer in df_valid['extracted_answers'].tolist()]
                else:
                    answers = [parse_code_function(answer) for answer in df_valid['extracted_answers'].tolist()]
                answer_cache = {} # for the same uid, the answer is the same and the program is assumed to be deterministic, therefore we cache the answer -> accuracy mapping
                if pt == 'code_f':
                    hidden_outputs = df_valid['hidden_outputs'].tolist()[0].tolist()
                    hidden_inputs = df_valid['hidden_inputs'].tolist()[0].tolist()
                else:
                    gold_output = df_valid['output'].tolist()[0]
                    program = df_valid['program'].tolist()[0]
                    # gold_input = df_valid['input'].tolist()[0]
                imports = df_valid['imports'].tolist()[0]
                problem_accuracies = []
                if problem_type.endswith('code_i'):
                    if self.batched_estimate:
                        problem_accuracies = executor.eval_k_input_prediction(code=program, gold_output=gold_output, k_agent_inputs=answers, imports=list(set(imports)))
                    else:
                        for answer in answers:
                            if answer in answer_cache:
                                problem_accuracies.append(answer_cache[answer])
                                continue
                            acc_reward = executor.eval_input_prediction(code=program, gold_output=gold_output, agent_input=answer, imports=list(set(imports)))
                            if acc_reward is not None:
                                problem_accuracies.append(acc_reward)
                            answer_cache[answer] = acc_reward
                        # if self.debug:
                        #     batched_problem_accuracies = executor.eval_k_input_prediction(code=program, gold_output=gold_output, k_agent_inputs=answers, imports=list(set(imports)))
                        #     assert np.mean(batched_problem_accuracies) == np.mean(problem_accuracies), f"Gen I batch accuracy: {np.mean(batched_problem_accuracies)}, Single accuracy: {np.mean(problem_accuracies)}"
                elif problem_type.endswith('code_o'):
                    if self.batched_estimate:
                        problem_accuracies = executor.eval_k_output_prediction(code=program, gold_output=gold_output, k_agent_outputs=answers, imports=list(set(imports)))
                    else:
                        for answer in answers:
                            if answer in answer_cache:
                                problem_accuracies.append(answer_cache[answer])
                                continue
                            acc_reward = executor.eval_output_prediction(code=program, gold_output=gold_output, agent_output=answer, imports=list(set(imports)))
                            if acc_reward is not None:
                                problem_accuracies.append(acc_reward)
                            answer_cache[answer] = acc_reward
                        # if self.debug:
                        #     batched_problem_accuracies = executor.eval_k_output_prediction(code=program, gold_output=gold_output, k_agent_outputs=answers, imports=list(set(imports)))
                        #     assert np.mean(batched_problem_accuracies) == np.mean(problem_accuracies), f"Gen O batch accuracy: {np.mean(batched_problem_accuracies)}, Single accuracy: {np.mean(problem_accuracies)}"
                elif problem_type.endswith('code_e'): # string matching for errors
                    for answer in answers:
                        answer = answer.split(' ')[0].split(':')[0]
                        if answer.lower() == gold_output.lower():
                            problem_accuracies.append(1.0)
                        else:
                            problem_accuracies.append(0.0)
                elif problem_type.endswith('code_f'):
                    for parsed, answer in answers: # for each input/output set, we sampled n codes to estimate the accuracy
                        if not parsed: # the code answer is not parsed, we assume the code is not valid
                            problem_accuracies.append(0.0)
                            continue
                        code_accuracies = []
                        for inpt, outpt in zip(hidden_inputs, hidden_outputs):
                            code_accuracies.append(executor.eval_input_prediction(code=answer, gold_output=outpt, agent_input=inpt, imports=list(set(imports))))
                        answer_acc = np.mean([a for a in code_accuracies if a is not None]) if code_accuracies else 0.0
                        if self.code_f_reward_type == 'binary':
                            problem_accuracies.append(1.0 if answer_acc == 1.0 else 0.0)
                        elif self.code_f_reward_type == 'if_one_correct':
                            problem_accuracies.append(1.0 if answer_acc > 0 else 0.0)
                        elif self.code_f_reward_type == 'accuracy':
                            problem_accuracies.append(answer_acc)
                        else:
                            raise ValueError(f"Invalid code_f_reward_type: {self.code_f_reward_type}")
                accuracies[valid_uid] = sum(problem_accuracies) / len(problem_accuracies) if problem_accuracies else 0.0

                # filtering valid programs
                if self.valid_program_filter == 'all':
                    valid_programs.append(valid_data_dicts[uid2valid_dict_idx[valid_uid]]['answer'])
                elif self.valid_program_filter == 'non_one':
                    if accuracies[valid_uid] < 1.0:
                        valid_programs.append(valid_data_dicts[uid2valid_dict_idx[valid_uid]]['answer'])
                elif self.valid_program_filter == 'non_extremes':
                    if accuracies[valid_uid] > 0.0 and accuracies[valid_uid] < 1.0:
                        valid_programs.append(valid_data_dicts[uid2valid_dict_idx[valid_uid]]['answer'])
                else:
                    raise ValueError(f"Invalid valid program filter: {self.valid_program_filter}")

        # getting other rewards
        PrettyPrinter.section_header("Getting Other Rewards")
        # outputting rewards
        for d in data_dicts:
            uid = d['uid']
            if self.generation_reward_config.generation_accuracy_convertion == 'one_minus':
                rewards[uid]['accuracy'] = (1 - accuracies[uid]) if accuracies[uid] > 0 else 0.0
            elif self.generation_reward_config.generation_accuracy_convertion == 'inverse':
                rewards[uid]['accuracy'] = 1 - accuracies[uid]
            else:
                raise ValueError(f"Invalid generation accuracy convertion: {self.generation_reward_config.generation_accuracy_convertion}")

        if not problem_type.endswith('code_f'):
            code_key = 'original_snippet' if self.use_original_code_as_ref else 'snippet'
            reference_key = 'original_references' if self.use_original_code_as_ref else 'references'
            if problem_type.endswith('code_i'):
                type_counter_key = 'input'
            elif problem_type.endswith('code_o'):
                type_counter_key = 'output'
            elif problem_type.endswith('code_e'):
                type_counter_key = 'error'
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['complexity'] = get_code_complexity_reward(data_dict['answer'][code_key]) if 'answer' in data_dict else 0.0
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['mean_edit_distance'] = np.mean([ast_edit_distance(data_dict['answer'][code_key], ref) for ref in data_dict[reference_key]]) if 'answer' in data_dict else 0.0
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['halstead'] = get_halstead_reward(data_dict['answer'][code_key]) if 'answer' in data_dict else 0.0
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['type_counts'] = get_type_counts_reward(
                    data_dict['answer'][type_counter_key],
                    type_counters,
                    hierarchical=self.generation_reward_config.answer_diversity_reward.hierarchical
                ) if 'answer' in data_dict else 0.0
            if self.debug:
                for data_dict in data_dicts:
                    if 'answer' in data_dict:
                        continue
        else:
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['input_type_counts'] = []
                rewards[data_dict['uid']]['output_type_counts'] = []
                if 'answer' in data_dict:
                    for inpt, outpt in zip(data_dict['answer']['inputs'], data_dict['answer']['outputs']):
                        rewards[data_dict['uid']]['input_type_counts'].append(get_type_counts_reward(
                            inpt,
                            input_type_counters,
                            hierarchical=self.generation_reward_config.answer_diversity_reward.hierarchical
                        ))
                        rewards[data_dict['uid']]['output_type_counts'].append(get_type_counts_reward(
                            outpt,
                            output_type_counters,
                            hierarchical=self.generation_reward_config.answer_diversity_reward.hierarchical
                        ))
                    rewards[data_dict['uid']]['input_type_counts'] = np.mean(rewards[data_dict['uid']]['input_type_counts'])
                    rewards[data_dict['uid']]['output_type_counts'] = np.mean(rewards[data_dict['uid']]['output_type_counts'])
                else:
                    rewards[data_dict['uid']]['input_type_counts'] = 0.0
                    rewards[data_dict['uid']]['output_type_counts'] = 0.0

        # turn into normal dict
        rewards = dict(rewards)
        return rewards, valid_programs
