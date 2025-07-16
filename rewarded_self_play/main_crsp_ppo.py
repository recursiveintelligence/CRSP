# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import ray
import hydra
from pathlib import Path
from pprint import pprint

from omegaconf import OmegaConf
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils import hf_tokenizer
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

from rewarded_self_play.trainer.ppo.crsp_ray_trainer import CRSPRayPPOTrainer
from rewarded_self_play.rewards.reward_managers import CodeIORewardManager


@hydra.main(config_path='configs', config_name='crsp_ppo_trainer', version_base=None)
def main(config):
    # Support command-line override for CRSP parameters
    # Trajectory seeding examples:
    #   python main_crsp_ppo.py trajectory_seeding.enabled=true
    #   python main_crsp_ppo.py trajectory_seeding.enabled=true trajectory_seeding.epochs=2
    #   python main_crsp_ppo.py trajectory_seeding.limo_dataset_path="custom/path"
    # 
    # CRSP training examples:
    #   python main_crsp_ppo.py crsp.executor=sandboxfusion
    #   python main_crsp_ppo.py crsp.problem_types=[code_i,code_o]
    #   python main_crsp_ppo.py trainer.experiment_name=my_crsp_experiment
    
    # Validate trajectory seeding configuration
    validate_trajectory_seeding_config(config)
    
    # Print configuration summary
    print_config_summary(config)
    
    run_ppo(config)


def print_config_summary(config):
    """Print a summary of key CRSP configuration parameters."""
    print("=" * 60)
    print("CRSP CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Experiment name: {config.trainer.experiment_name}")
    print(f"Model path: {config.actor_rollout_ref.model.path}")
    print(f"Problem types: {config.crsp.problem_types}")
    print(f"Executor: {config.crsp.executor}")
    print(f"Total epochs: {config.trainer.total_epochs}")
    
    if config.trajectory_seeding.enabled:
        print(f"Trajectory seeding: ENABLED")
        print(f"  - Dataset: {config.trajectory_seeding.limo_dataset_path}")
        print(f"  - Epochs: {config.trajectory_seeding.epochs}")
        print(f"  - Batch size: {config.trajectory_seeding.batch_size}")
        print(f"  - Learning rate: {config.trajectory_seeding.learning_rate}")
    else:
        print(f"Trajectory seeding: DISABLED")
    
    print("=" * 60)


def validate_trajectory_seeding_config(config):
    """Validate trajectory seeding configuration parameters."""
    if config.trajectory_seeding.enabled:
        # Validate required parameters
        assert config.trajectory_seeding.limo_dataset_path, "limo_dataset_path must be specified when trajectory seeding is enabled"
        assert config.trajectory_seeding.batch_size > 0, "batch_size must be positive"
        assert config.trajectory_seeding.learning_rate > 0, "learning_rate must be positive"
        assert config.trajectory_seeding.epochs > 0, "epochs must be positive"
        assert 0 <= config.trajectory_seeding.alpha <= 1, "alpha must be between 0 and 1"
        
        # Validate validation configuration if enabled
        if config.trajectory_seeding.validation.enabled:
            assert config.trajectory_seeding.validation.eval_steps > 0, "eval_steps must be positive"
            assert config.trajectory_seeding.validation.save_total_limit > 0, "save_total_limit must be positive"
        
        print(f"Trajectory seeding enabled with dataset: {config.trajectory_seeding.limo_dataset_path}")
        print(f"Seeding configuration: epochs={config.trajectory_seeding.epochs}, "
              f"batch_size={config.trajectory_seeding.batch_size}, "
              f"learning_rate={config.trajectory_seeding.learning_rate}")
        
        # Set default values for optional parameters
        if not hasattr(config.trajectory_seeding, 'cache_dir') or not config.trajectory_seeding.cache_dir:
            config.trajectory_seeding.cache_dir = "~/.cache/crsp/limo"
        if not hasattr(config.trajectory_seeding, 'save_steps'):
            config.trajectory_seeding.save_steps = 500
        if not hasattr(config.trajectory_seeding, 'logging_steps'):
            config.trajectory_seeding.logging_steps = 100
    else:
        print("Trajectory seeding disabled - proceeding directly to self-play training")


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if OmegaConf.select(config.trainer, "profile_steps") is not None and len(OmegaConf.select(config.trainer, "profile_steps")) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        if config.trainer.debug:
            import debugpy
            debugpy.listen(("0.0.0.0", config.trainer.debug_port))
            print(f"Debugger listening on port {config.trainer.debug_port}")
            debugpy.wait_for_client()
            print("Debugger attached!")

        # generator one batch, solver one batch
        config.actor_rollout_ref.actor.ppo_mini_batch_size = config.data.train_batch_size * len(config.crsp.problem_types) * (2 if config.crsp.train_propose else 1)
        pprint(f"auto setting ppo_mini_batch_size: {config.actor_rollout_ref.actor.ppo_mini_batch_size}")
        config.crsp.data_selection_strategy.data_len = config.data.train_batch_size * config.crsp.data_selection_strategy.update_iteration
        pprint(f"auto setting data_len: {config.crsp.data_selection_strategy.data_len}")

        config.trainer.default_local_dir = (Path(config.trainer.default_local_dir) / config.data.train_files.split('/')[-1].split('.')[0] / config.actor_rollout_ref.model.path.split('/')[-1] / config.reward_fn.extraction_type).as_posix()

        assert not (not config.crsp.reward.generation_reward_config.reject_multiple_functions and config.crsp.data_selection_strategy.composite_function_n_min > 0), "If reject_multiple_functions is False, composite_function_n_min must be 0"

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # base model chat template
        if config.actor_rollout_ref.model.pretrained_tokenizer:
            tokenizer.chat_template = "{%- for message in messages -%}{{- '\n' if not loop.first -}}{{- message['content'] -}}{%- endfor -%}"

        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Version validation for vllm.
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rol# lout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = CodeIORewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            reward_fn_extraction_type=config.reward_fn.extraction_type,
            math_metric=config.reward_fn.math_metric,
            split='train',
            splitter=config.reward_fn.splitter,
            output_path=config.trainer.default_local_dir,
            max_prompt_length=config.data.max_prompt_length,
            generation_reward_config=config.crsp.reward.generation_reward_config,
            valid_program_filter=config.crsp.data_selection_strategy.valid_program_filter,
            debug=config.trainer.debug,
            extract_code_block=config.crsp.reward.extract_code_block,
            code_f_reward_type=config.crsp.reward.code_f_reward_type,
            boxed_retry=config.reward_fn.boxed_retry,
        )

        # Note that we always use function-based RM for validation
        val_reward_fn = CodeIORewardManager(
            tokenizer=tokenizer,
            num_examine=1,
            reward_fn_extraction_type=config.reward_fn.extraction_type,
            math_metric=config.reward_fn.math_metric,
            split='test',
            splitter=config.reward_fn.splitter,
            output_path=config.trainer.default_local_dir,
            max_prompt_length=config.data.max_prompt_length,
            generation_reward_config=config.crsp.reward.generation_reward_config,
            valid_program_filter=config.crsp.data_selection_strategy.valid_program_filter,
            debug=config.trainer.debug,
            extract_code_block=config.crsp.reward.extract_code_block,
            code_f_reward_type=config.crsp.reward.code_f_reward_type,
            boxed_retry=config.reward_fn.boxed_retry,
        )

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        wandb_tags = [
            'codeio', config.crsp.pred_data_mix_strategy, 'executor-' + config.crsp.executor,
            config.crsp.data_selection_strategy.valid_program_filter, config.crsp.gen_data_probabilities_strategy,
        ]
        wandb_tags.extend(config.crsp.problem_types)
        if config.trainer.wandb_tags is not None:
            config.trainer.wandb_tags = wandb_tags + config.trainer.wandb_tags.split(',')
        else:
            config.trainer.wandb_tags = wandb_tags

        # Phase 1: Optional trajectory seeding phase
        seeded_model_path = local_path  # Default to original model path
        if config.trajectory_seeding.enabled:
            print("=" * 60)
            print("PHASE 1: TRAJECTORY SEEDING")
            print("=" * 60)
            print("Starting trajectory seeding phase...")
            
            from rewarded_self_play.trajectory_seeding.seeder import TrajectorySeeder
            
            seeder = TrajectorySeeder(
                model_path=local_path,
                tokenizer=tokenizer,
                config=config.trajectory_seeding
            )
            
            # Run trajectory seeding
            seeded_model_path = seeder.run_seeding()
            
            print(f"Trajectory seeding completed successfully!")
            print(f"Seeded model saved to: {seeded_model_path}")
            print("=" * 60)
        else:
            print("Trajectory seeding disabled - proceeding directly to self-play training")
        
        # Phase 2: Self-play training phase
        print("=" * 60)
        print("PHASE 2: SELF-PLAY TRAINING")
        print("=" * 60)
        
        # Update model path to use seeded model (or original if seeding disabled)
        config.actor_rollout_ref.model.path = seeded_model_path
        
        # Update critic model path to match actor if not explicitly set differently
        if not hasattr(config.critic.model, 'path') or config.critic.model.path == local_path:
            config.critic.model.path = seeded_model_path
            
        print(f"Using model for self-play training: {seeded_model_path}")
        
        trainer = CRSPRayPPOTrainer(
            past_epoch_window=config.crsp.past_epoch_window,
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

        trainer.init_workers()
        trainer.fit()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        import sys
        import traceback
        traceback.print_exc()
        sys.exit(0)
    except Exception as e:
        import os
        import traceback
        traceback.print_exc()
        os._exit(1)
