# Design Document

## Overview

The CRSP migration design transforms the AZR codebase into a comprehensive Cognition via Rewarded Self-Play implementation. The design follows a systematic approach: identifier replacement, reward system enhancement with RLSP integration, TR-RPG optimization upgrade, and optional LIMO trajectory seeding.

## Architecture

### Core Components

1. **Renamed Module Structure**
   - `absolute_zero_reasoner/` → `rewarded_self_play/`
   - `azr_ray_trainer.py` → `crsp_ray_trainer.py`
   - Configuration parameters updated throughout

2. **Enhanced Reward System**
   - Multi-dimensional reward integration (correctness + length + creativity)
   - Alpha decay schedule for exploration-exploitation balance
   - Critique mechanism for reasoning quality assessment

3. **TR-RPG Optimization**
   - KL-regularized policy gradients with importance weighting
   - Policy-specific regularization coefficients
   - Off-policy correction mechanisms

4. **Optional Trajectory Seeding**
   - LIMO dataset integration for initialization
   - Supervised fine-tuning phase before self-play
   - Configurable seeding parameters

## Components and Interfaces

### 1. Identifier Replacement System

**Files to Update:**
- All Python files in `absolute_zero_reasoner/`
- Configuration files (`configs/azr_ppo_trainer.yaml`)
- Script files in `scripts/`
- Documentation and README files

**Replacement Mapping:**
```python
IDENTIFIER_MAPPING = {
    'azr': 'crsp',
    'absolute_zero_reasoner': 'rewarded_self_play',
    'absolute zero reasoner': 'Cognition via Rewarded Self-Play',
    'AZR': 'CRSP',
    'Absolute Zero Reasoner': 'Cognition via Rewarded Self-Play'
}
```

### 2. Enhanced Reward Architecture

**Reward Components:**
```python
class CRSPRewardManager:
    def compute_integrated_reward(self, solution, correctness, alpha_s, alpha_c):
        # Solver reward: correctness + length
        r_solve = alpha_s * correctness + (1 - alpha_s) * length_reward
        
        # Critique reward: agreement + creativity  
        r_critique = alpha_c * agreement + (1 - alpha_c) * creativity_reward
        
        return r_solve, r_critique
```

**Alpha Decay Schedule:**
```python
def compute_alpha_decay(step, total_steps):
    # RLSP-inspired coupled decay
    if step < 0.2 * total_steps:
        alpha_s, alpha_c = 0.3, 0.1  # Encourage exploration
    elif step < 0.6 * total_steps:
        # Linear interpolation phase
        progress = (step - 0.2 * total_steps) / (0.4 * total_steps)
        alpha_s = 0.3 + 0.5 * progress  # 0.3 → 0.8
        alpha_c = 0.1 + 0.5 * progress  # 0.1 → 0.6
    else:
        # Final convergence phase
        progress = (step - 0.6 * total_steps) / (0.4 * total_steps)
        alpha_s = 0.8 + 0.15 * progress  # 0.8 → 0.95
        alpha_c = 0.6 + 0.2 * progress   # 0.6 → 0.8
    
    return alpha_s, alpha_c
```

### 3. TR-RPG Implementation

**Core TR-RPG Algorithm:**
```python
class TaskRelativeRPG:
    def __init__(self, policies=['propose', 'solve', 'critique']):
        self.policies = policies
        self.beta_coefficients = {
            'propose': 0.01,  # Encourage task diversity
            'solve': 0.05,    # Balance exploration/correctness
            'critique': 0.1   # Stable evaluation
        }
    
    def compute_gradient(self, policy, rewards, old_policy_probs, new_policy_probs):
        # Importance weights
        w = new_policy_probs / old_policy_probs
        
        # KL-regularized gradient
        kl_term = self.beta_coefficients[policy] * (torch.log(w) + 1)
        gradient = w * (rewards - kl_term)
        
        return gradient
```

**Policy-Specific Rewards:**
```python
def compute_policy_rewards(self, policy_type, data):
    if policy_type == 'solve':
        return self.alpha_s * correctness + (1 - self.alpha_s) * length_reward
    elif policy_type == 'critique':
        return self.alpha_c * agreement + (1 - self.alpha_c) * creativity_reward
    elif policy_type == 'propose':
        return self.learnability_reward(data)
```

### 4. Critique Mechanism Design

**Creativity Grading Prompt:**
```python
CREATIVITY_GRADING_PROMPT = """
You are a Thinking-Effort Grading Assistant. Your goal is to assess 
a solution's thinking trajectory (the reasoning process within <think> tags) 
and output a numeric score in [0,1] based on how hard the solver tried.

Key Dimensions to Evaluate:
1. Diversity of Strategies - How many different approaches considered?
2. Depth of Exploration - Detailed steps and genuine effort shown?
3. Creativity and Novelty - Unusual or "out-of-the-box" ideas?
4. Persistence and Rigor - Systematic testing and refinement?

Output Format:
Return evaluation in JSON format:
{
  "rationale": "Explanation of score based on criteria",
  "grade": 0.75
}

Focus only on the process and effort, not correctness.
"""
```

**Critique Integration:**
```python
class CritiqueManager:
    def evaluate_reasoning(self, solution_text):
        # Extract <think> content
        think_content = self.extract_think_tags(solution_text)
        
        # Generate critique using grading prompt
        critique_response = self.model.generate(
            self.format_critique_prompt(think_content)
        )
        
        # Parse JSON response
        evaluation = json.loads(critique_response)
        return evaluation['grade'], evaluation['rationale']
```

### 5. Trajectory Seeding Architecture

**LIMO Dataset Integration:**
```python
class TrajectorySeeder:
    def __init__(self, limo_dataset_path):
        self.dataset = self.load_limo_dataset(limo_dataset_path)
    
    def create_seeding_data(self):
        seeding_samples = []
        for sample in self.dataset:
            formatted_sample = {
                'prompt': [{'role': 'user', 'content': sample['question']}],
                'solution': sample['solution'],
                'answer': sample['answer']
            }
            seeding_samples.append(formatted_sample)
        return seeding_samples
    
    def supervised_fine_tune(self, model, seeding_data):
        # SFT objective: L_seed = -E[log π(s|q) + α log π(a|q,s)]
        for batch in seeding_data:
            loss = self.compute_sft_loss(model, batch)
            loss.backward()
            self.optimizer.step()
```

**Seeding Directory Structure:**
```
trajectory_seeding/
├── __init__.py
├── seeder.py              # Main seeding logic
├── limo_processor.py      # LIMO dataset processing
├── sft_trainer.py         # Supervised fine-tuning
└── scripts/
    ├── run_seeding.sh     # Seeding execution script
    └── seeding_config.yaml # Seeding configuration
```

## Data Models

### Configuration Schema Updates

**CRSP Configuration Structure:**
```yaml
# Updated from azr_ppo_trainer.yaml to crsp_ppo_trainer.yaml
crsp:
  seed: 1
  executor_max_workers: 1
  executor_cleanup_frequency: 1
  problem_types:
    - code_i
    - code_o  
    - code_f
  # ... other CRSP-specific parameters

# New trajectory seeding section
trajectory_seeding:
  enabled: false
  limo_dataset_path: "GAIR/LIMO"
  seeding_steps: 1000
  learning_rate: 1e-5
  batch_size: 32

# Enhanced reward configuration
reward_structure:
  alpha_decay:
    enabled: true
    schedule_type: "rlsp_coupled"
  creativity_grading:
    enabled: true
    dimensions: ["novelty", "depth", "diversity", "persistence", "rigor"]
  length_reward:
    max_length: 2048
    penalty_threshold: 4096
```

### Reward Data Models

```python
@dataclass
class CRSPReward:
    correctness: float
    length_score: float
    creativity_score: float
    agreement_score: float
    alpha_s: float
    alpha_c: float
    
    def compute_solve_reward(self) -> float:
        return self.alpha_s * self.correctness + (1 - self.alpha_s) * self.length_score
    
    def compute_critique_reward(self) -> float:
        return self.alpha_c * self.agreement_score + (1 - self.alpha_c) * self.creativity_score

@dataclass
class CreativityEvaluation:
    novelty: float
    depth: float
    diversity: float
    persistence: float
    rigor: float
    overall_grade: float
    rationale: str
```

## Error Handling

### Migration Error Handling

1. **Identifier Replacement Validation**
   - Verify all replacements are consistent
   - Check for missed occurrences
   - Validate import statement updates

2. **Reward System Error Handling**
   - Handle malformed critique responses
   - Validate alpha decay schedule bounds
   - Ensure reward normalization stability

3. **TR-RPG Error Handling**
   - Check for numerical instability in importance weights
   - Validate KL regularization terms
   - Handle policy divergence cases

4. **Trajectory Seeding Error Handling**
   - Validate LIMO dataset format
   - Handle missing or corrupted samples
   - Ensure SFT convergence

## Testing Strategy

### Unit Testing
- Identifier replacement verification
- Reward computation accuracy
- TR-RPG gradient calculations
- Critique mechanism functionality

### Integration Testing
- End-to-end CRSP training pipeline
- Trajectory seeding integration
- Multi-policy coordination
- Configuration loading and validation

### Performance Testing
- Training speed comparison (TRR++ vs TR-RPG)
- Memory usage optimization
- Reward computation efficiency
- Seeding phase performance impact

## Implementation Phases

### Phase 1: Core Migration (Requirements 1, 4)
1. Systematic identifier replacement
2. File and directory renaming
3. Import statement updates
4. Configuration file migration

### Phase 2: Reward Enhancement (Requirement 2)
1. Length reward implementation
2. Creativity reward and critique mechanism
3. Alpha decay schedule integration
4. Grading prompt addition

### Phase 3: TR-RPG Implementation (Requirement 3)
1. TR-RPG algorithm implementation
2. Policy-specific regularization
3. Importance weighting mechanisms
4. Gradient computation updates

### Phase 4: Trajectory Seeding (Requirement 5)
1. LIMO dataset integration
2. Seeding directory structure
3. SFT trainer implementation
4. Configuration integration

### Phase 5: Integration and Testing (Requirement 6)
1. End-to-end testing
2. Performance validation
3. Documentation updates
4. Final cleanup and optimization