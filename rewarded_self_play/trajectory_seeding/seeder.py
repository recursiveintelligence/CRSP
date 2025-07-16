"""
Main trajectory seeding logic for CRSP.

This module coordinates the trajectory seeding process using LIMO dataset
to provide high-quality cognitive templates for initialization.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from .limo_processor import LIMOProcessor
from .sft_trainer import SFTTrainer


class TrajectorySeeder:
    """
    Main coordinator for trajectory seeding phase.
    
    Handles the complete seeding workflow from LIMO data processing
    to supervised fine-tuning on high-quality reasoning traces.
    """
    
    def __init__(self, 
                 limo_dataset_path: str = "GAIR/LIMO",
                 seeding_steps: int = 1000,
                 learning_rate: float = 1e-5,
                 batch_size: int = 32,
                 output_dir: str = "./trajectory_seeding_output"):
        """
        Initialize trajectory seeder.
        
        Args:
            limo_dataset_path: Path or HuggingFace dataset name for LIMO
            seeding_steps: Number of supervised fine-tuning steps
            learning_rate: Learning rate for SFT
            batch_size: Batch size for training
            output_dir: Directory to save seeding outputs
        """
        self.limo_dataset_path = limo_dataset_path
        self.seeding_steps = seeding_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        
        # Initialize components
        self.limo_processor = LIMOProcessor(limo_dataset_path)
        self.sft_trainer = None  # Will be initialized when needed
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def prepare_seeding_data(self) -> List[Dict[str, Any]]:
        """
        Prepare LIMO data for trajectory seeding.
        
        Returns:
            List of formatted seeding samples
        """
        self.logger.info("Loading and processing LIMO dataset...")
        
        # Load LIMO dataset
        limo_data = self.limo_processor.load_dataset()
        
        # Process and format for seeding
        seeding_samples = self.limo_processor.create_seeding_data(limo_data)
        
        self.logger.info(f"Prepared {len(seeding_samples)} seeding samples")
        return seeding_samples
    
    def run_seeding(self, 
                   model,
                   tokenizer,
                   seeding_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Run the complete trajectory seeding process.
        
        Args:
            model: The model to seed
            tokenizer: Tokenizer for the model
            seeding_data: Optional pre-processed seeding data
            
        Returns:
            Dictionary containing seeding results and metrics
        """
        self.logger.info("Starting trajectory seeding process...")
        
        # Prepare data if not provided
        if seeding_data is None:
            seeding_data = self.prepare_seeding_data()
        
        # Initialize SFT trainer
        self.sft_trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            output_dir=str(self.output_dir)
        )
        
        # Run supervised fine-tuning
        self.logger.info("Running supervised fine-tuning on LIMO traces...")
        training_results = self.sft_trainer.train(
            seeding_data=seeding_data,
            num_steps=self.seeding_steps
        )
        
        # Save seeded model
        seeded_model_path = self.output_dir / "seeded_model"
        self.sft_trainer.save_model(str(seeded_model_path))
        
        self.logger.info(f"Trajectory seeding completed. Model saved to {seeded_model_path}")
        
        return {
            'seeded_model_path': str(seeded_model_path),
            'training_results': training_results,
            'num_seeding_samples': len(seeding_data),
            'seeding_steps': self.seeding_steps
        }
    
    def validate_seeding(self, model, tokenizer, validation_samples: int = 10) -> Dict[str, float]:
        """
        Validate the seeded model on a small set of samples.
        
        Args:
            model: Seeded model to validate
            tokenizer: Tokenizer for the model
            validation_samples: Number of samples to validate on
            
        Returns:
            Dictionary of validation metrics
        """
        self.logger.info("Validating seeded model...")
        
        # Get validation data
        validation_data = self.limo_processor.get_validation_samples(validation_samples)
        
        # Run validation
        if self.sft_trainer is None:
            self.sft_trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                output_dir=str(self.output_dir)
            )
        
        validation_metrics = self.sft_trainer.validate(validation_data)
        
        self.logger.info(f"Validation completed: {validation_metrics}")
        return validation_metrics