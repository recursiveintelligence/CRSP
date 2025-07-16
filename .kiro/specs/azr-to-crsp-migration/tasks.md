# Implementation Plan

## Phase 1: Core Migration and Identifier Replacement

- [x] 1. Rename main directory structure
  - Rename `absolute_zero_reasoner/` to `rewarded_self_play/`
  - Update all import statements referencing the old module name
  - _Requirements: 1.1, 4.1_

- [x] 2. Update core trainer files with CRSP naming
  - Rename `azr_ray_trainer.py` to `crsp_ray_trainer.py`
  - Replace all class names and identifiers within the file
  - Update import statements in dependent files
  - _Requirements: 1.1, 4.2_

- [x] 3. Migrate configuration files
  - Rename `azr_ppo_trainer.yaml` to `crsp_ppo_trainer.yaml`
  - Replace all parameter names with CRSP equivalents
  - Update configuration loading logic in main files
  - _Requirements: 1.3, 4.3_

- [x] 4. Update script files and shell scripts
  - Replace identifiers in all `.sh` files under `scripts/`
  - Update command-line argument names and help text
  - Ensure script execution paths reference new module names
  - _Requirements: 1.1, 4.4_

- [x] 5. Systematic identifier replacement across codebase
  - Replace `azr` → `crsp` in all Python files
  - Replace `absolute_zero_reasoner` → `rewarded_self_play`
  - Replace `Absolute Zero Reasoner` → `Cognition via Rewarded Self-Play`
  - Update docstrings and comments
  - _Requirements: 1.1, 1.2_

## Phase 2: Enhanced Reward System Implementation

- [x] 6. Implement length-based reward mechanism
  - Create `compute_length_reward()` function in reward managers
  - Add logarithmic scaling with penalty for excessive length
  - Integrate with existing reward computation pipeline
  - _Requirements: 2.1, 2.5_

- [x] 7. Add creativity grading prompt to prompts.py
  - Create `CREATIVITY_GRADING_PROMPT` with JSON output format
  - Include evaluation dimensions (novelty, depth, diversity, persistence, rigor)
  - Add prompt formatting functions for critique generation
  - _Requirements: 2.4_

- [x] 8. Implement critique mechanism
  - Create `CritiqueManager` class for reasoning evaluation
  - Add `extract_think_tags()` function to parse reasoning content
  - Implement JSON response parsing for creativity scores
  - _Requirements: 2.3, 2.4_

- [x] 9. Integrate alpha decay schedule
  - Implement RLSP-inspired coupled alpha decay function
  - Add three-phase schedule (exploration → interpolation → convergence)
  - Update reward computation to use time-varying alphas
  - _Requirements: 2.5_

- [x] 10. Create integrated reward computation
  - Modify `CodeIORewardManager` to handle multi-dimensional rewards
  - Implement solver reward (correctness + length) computation
  - Implement critique reward (agreement + creativity) computation
  - Add reward normalization and clipping mechanisms
  - _Requirements: 2.1, 2.2, 2.5_

## Phase 3: TR-RPG Implementation

- [x] 11. Create TR-RPG base classes
  - Implement `TaskRelativeRPG` class with policy-specific handling
  - Add KL regularization coefficient management
  - Create importance weight computation functions
  - _Requirements: 3.1, 3.4_

- [x] 12. Implement KL-regularized gradient computation
  - Create `compute_kl_regularized_gradient()` function
  - Add importance weighting with off-policy correction
  - Implement automatic gradient stabilization via KL terms
  - _Requirements: 3.2, 3.3, 3.4_

- [x] 13. Add policy-specific regularization
  - Set different beta coefficients for propose/solve/critique policies
  - Implement policy-specific reward computation functions
  - Add regularization strength tuning mechanisms
  - _Requirements: 3.5_

- [x] 14. Replace TRR++ with TR-RPG in trainer
  - Remove existing TRR++ implementation from trainer files
  - Integrate TR-RPG gradient computation into training loop
  - Update optimizer calls to use new gradient computation
  - Add convergence monitoring for KL-regularized objectives
  - _Requirements: 3.1, 3.2_

- [x] 15. Implement TR-RPG training algorithm
  - Create complete TR-RPG training loop with reference policy updates
  - Add periodic reference policy synchronization
  - Implement policy-specific batch processing
  - Add gradient accumulation across multiple policies
  - _Requirements: 3.1, 3.2, 3.5_

## Phase 4: Trajectory Seeding Implementation

- [x] 16. Create trajectory seeding directory structure
  - Create `trajectory_seeding/` directory with proper module structure
  - Add `__init__.py`, `seeder.py`, `limo_processor.py`, `sft_trainer.py`
  - Create `scripts/` subdirectory with execution and configuration files
  - _Requirements: 5.3_

- [x] 17. Implement LIMO dataset integration
  - Create `LIMOProcessor` class for dataset loading and formatting
  - Add JSON format validation for question/solution/answer structure
  - Implement data preprocessing and tokenization
  - Add error handling for missing or corrupted samples
  - _Requirements: 5.2, 5.4_

- [x] 18. Implement supervised fine-tuning trainer
  - Create `SFTTrainer` class for trajectory seeding phase
  - Implement SFT objective: L_seed = -E[log π(s|q) + α log π(a|q,s)]
  - Add training loop with proper loss computation and optimization
  - Integrate with existing model loading and saving infrastructure
  - _Requirements: 5.4_

- [x] 19. Add seeding configuration integration
  - Extend CRSP configuration with trajectory_seeding section
  - Add parameters for dataset path, seeding steps, learning rate
  - Implement configuration validation and default value handling
  - Add command-line argument support for seeding parameters
  - _Requirements: 5.5_

- [x] 20. Create seeding execution scripts
  - Create `run_seeding.sh` script for trajectory seeding execution
  - Add `seeding_config.yaml` with seeding-specific parameters
  - Integrate seeding phase with main training pipeline
  - Add optional seeding control via command-line flags
  - _Requirements: 5.1, 5.5_

## Phase 5: Integration and System Updates

- [x] 21. Update main training pipeline
  - Modify `main_crsp_ppo.py` to support optional trajectory seeding
  - Add two-phase training: seeding → self-play
  - Implement phase transition logic and model state management
  - Update CLI argument parsing for new CRSP parameters
  - _Requirements: 6.1, 6.3_

- [x] 22. Implement comprehensive logging
  - Add logging for length rewards, creativity scores, and alpha values
  - Create reward breakdown logging for debugging and monitoring
  - Add TR-RPG specific metrics (KL divergence, importance weights)
  - Implement trajectory seeding progress logging
  - _Requirements: 4.3_

- [x] 23. Update reward manager integration
  - Modify existing reward managers to use new CRSP reward structure
  - Ensure backward compatibility with existing reward computation
  - Add critique mechanism integration to reward pipeline
  - Update reward normalization and stability mechanisms
  - _Requirements: 2.1, 2.2, 2.3, 6.1_

- [ ] 24. Clean up obsolete code
  - Remove unused AZR-specific code and comments
  - Clean up deprecated TRR++ implementation files
  - Remove any placeholder implementations or incomplete features
  - Ensure no Bytedance or related copyright references remain
  - _Requirements: 4.1, 6.4_

- [x] 25. Final integration testing and validation
  - Ensure all import statements and module references work correctly
  - Ensure no mentions of 'azr' / 'Absolute zero reasoner' / 'absolute_zero_reasoner', etc... in the code
  - _Requirements: 4.2, 6.1, 6.2, 6.3_