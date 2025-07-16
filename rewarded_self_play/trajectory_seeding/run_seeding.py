"""
Main runner for CRSP trajectory seeding.

This script coordinates the complete trajectory seeding process,
from loading LIMO data to supervised fine-tuning.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from rewarded_self_play.trajectory_seeding import TrajectorySeeder


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('trajectory_seeding.log')
        ]
    )


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading model and tokenizer from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='left'
    )
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer


def main():
    """Main trajectory seeding function."""
    parser = argparse.ArgumentParser(description="CRSP Trajectory Seeding")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the base model")
    parser.add_argument("--limo_dataset", type=str, default="GAIR/LIMO",
                       help="LIMO dataset path or HuggingFace dataset name")
    parser.add_argument("--seeding_steps", type=int, default=1000,
                       help="Number of supervised fine-tuning steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate for SFT")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--output_dir", type=str, default="./trajectory_seeding_output",
                       help="Output directory for seeded model")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--validation_samples", type=int, default=10,
                       help="Number of samples for validation")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting CRSP Trajectory Seeding")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_path)
        
        # Initialize trajectory seeder
        seeder = TrajectorySeeder(
            limo_dataset_path=args.limo_dataset,
            seeding_steps=args.seeding_steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
        
        # Run trajectory seeding
        results = seeder.run_seeding(model, tokenizer)
        
        # Validate seeded model
        validation_metrics = seeder.validate_seeding(
            model, tokenizer, args.validation_samples
        )
        
        # Log final results
        logger.info("Trajectory seeding completed successfully!")
        logger.info(f"Results: {results}")
        logger.info(f"Validation metrics: {validation_metrics}")
        
        print("\n" + "="*50)
        print("TRAJECTORY SEEDING COMPLETED")
        print("="*50)
        print(f"Seeded model saved to: {results['seeded_model_path']}")
        print(f"Training steps: {results['seeding_steps']}")
        print(f"Seeding samples: {results['num_seeding_samples']}")
        print(f"Validation loss: {validation_metrics.get('validation_loss', 'N/A')}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Trajectory seeding failed: {e}")
        raise


if __name__ == "__main__":
    main()