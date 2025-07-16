# Requirements Document

## Introduction

This document outlines the requirements for migrating the Absolute Zero Reasoner (AZR) codebase to implement Cognition via Rewarded Self-Play (CRSP). The migration involves comprehensive identifier replacement, reward system enhancement, optimization algorithm upgrade, and optional trajectory seeding integration.

## Requirements

### Requirement 1: Identifier Replacement

**User Story:** As a developer, I want all AZR-related identifiers replaced with CRSP equivalents, so that the codebase reflects the new framework.

#### Acceptance Criteria

1. WHEN scanning the codebase THEN all occurrences of `azr`, `absolute_zero_reasoner`, `absolute zero reasoner` SHALL be replaced with `crsp`, `rewarded_self_play`, `Cognition via Rewarded Self-Play`
2. WHEN examining file names and directory structures THEN they SHALL use CRSP naming conventions
3. WHEN reviewing configuration files THEN all parameter names SHALL reflect CRSP terminology
4. WHEN checking import statements THEN all module references SHALL use updated CRSP names

### Requirement 2: Reward and Critique Integration

**User Story:** As a researcher, I want enhanced reward mechanisms with length-based and creativity rewards plus critique functionality, so that the system can evaluate reasoning quality beyond correctness.

#### Acceptance Criteria

1. WHEN implementing length rewards THEN the system SHALL encourage thoughtful reasoning chains using RLSP logic
2. WHEN implementing creativity rewards THEN the system SHALL assess solution novelty and innovative thinking
3. WHEN adding critique mechanism THEN it SHALL evaluate reasoning trajectories within `<think>` tags
4. WHEN creating grading prompts THEN they SHALL be added to the prompts.py file with proper JSON output format
5. WHEN computing rewards THEN the system SHALL integrate correctness, length, and creativity signals with alpha decay schedule

### Requirement 3: TRR++ to TR-RPG Migration

**User Story:** As a machine learning engineer, I want the optimization algorithm upgraded from TRR++ to TR-RPG, so that the system has principled KL regularization and superior theoretical guarantees.

#### Acceptance Criteria

1. WHEN replacing TRR++ THEN the system SHALL implement Task-Relative Regularized Policy Gradients (TR-RPG)
2. WHEN computing gradients THEN TR-RPG SHALL use KL regularization with importance weighting
3. WHEN handling off-policy data THEN the system SHALL correctly apply importance weights
4. WHEN preventing policy collapse THEN KL regularization terms SHALL provide automatic gradient stabilization
5. WHEN training multiple policies THEN each SHALL have policy-specific regularization coefficients

### Requirement 4: Codebase Maintenance

**User Story:** As a maintainer, I want obsolete code removed and full CRSP implementation, so that the codebase is clean and production-ready.

#### Acceptance Criteria

1. WHEN removing obsolete code THEN all unused AZR-specific components SHALL be eliminated
2. WHEN implementing CRSP THEN scheduler, reward computation, and CLI logging SHALL be complete
3. WHEN updating trainer logic THEN self-play mechanisms SHALL reflect CRSP architecture
4. WHEN reviewing components THEN trainer, rewards, and data construction modules SHALL be fully updated
5. WHEN checking implementation THEN no placeholders or partial implementations SHALL remain

### Requirement 5: Trajectory Seeding Integration

**User Story:** As a researcher, I want optional trajectory seeding using LIMO dataset, so that the system can initialize with high-quality reasoning templates.

#### Acceptance Criteria

1. WHEN implementing trajectory seeding THEN it SHALL be an optional phase before self-play
2. WHEN using LIMO dataset THEN the system SHALL handle the expected JSON format with question, solution, and answer fields
3. WHEN creating seeding directory THEN it SHALL follow the structure pattern of existing scripts directories
4. WHEN integrating seeding THEN it SHALL leverage supervised fine-tuning on LIMO traces
5. WHEN configuring seeding THEN it SHALL be controllable via command-line parameters

### Requirement 6: Integration and Scope Discipline

**User Story:** As a developer, I want focused modifications that leverage existing logic, so that the migration is efficient and maintains system stability.

#### Acceptance Criteria

1. WHEN modifying files THEN existing logic SHALL be leveraged wherever applicable
2. WHEN updating imports THEN all naming compatibilities SHALL be maintained
3. WHEN making changes THEN only necessary files SHALL be modified
4. WHEN completing migration THEN no new Bytedance or related copyright references SHALL be included
5. WHEN finalizing implementation THEN the system SHALL be production-ready with complete CRSP functionality