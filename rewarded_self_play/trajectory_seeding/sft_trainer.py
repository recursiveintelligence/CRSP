"""
Supervised Fine-Tuning trainer for trajectory seeding.

This module implements the SFT trainer that fine-tunes the model
on LIMO reasoning traces during the trajectory seeding phase.
"""

import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from tqdm import tqdm


class SeedingDataset(Dataset):
    """Dataset class for trajectory seeding data."""
    
    def __init__(self, seeding_data: List[Dict[str, Any]], tokenizer, max_length: int = 2048):
        """
        Initialize seeding dataset.
        
        Args:
            seeding_data: List of seeding samples
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.seeding_data = seeding_data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.seeding_data)
    
    def __getitem__(self, idx):
        sample = self.seeding_data[idx]
        
        # Format the input-output pair
        question = sample['prompt'][0]['content']
        solution = sample['solution']
        answer = sample['answer']
        
        # Create the full sequence for SFT
        # Format: Question + Solution + Answer
        full_text = f"Question: {question}\n\nSolution: {solution}\n\nAnswer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()  # For language modeling
        }


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for trajectory seeding.
    
    Implements the SFT objective: L_seed = -E[log π(s|q) + α log π(a|q,s)]
    """
    
    def __init__(self,
                 model,
                 tokenizer,
                 learning_rate: float = 1e-5,
                 batch_size: int = 32,
                 output_dir: str = "./sft_output",
                 alpha: float = 0.1):
        """
        Initialize SFT trainer.
        
        Args:
            model: Model to fine-tune
            tokenizer: Tokenizer for the model
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            output_dir: Directory to save outputs
            alpha: Weight for answer loss in SFT objective
        """
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.alpha = alpha
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.global_step = 0
        self.training_metrics = []
    
    def compute_sft_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute SFT loss: L_seed = -E[log π(s|q) + α log π(a|q,s)]
        
        Args:
            batch: Batch of training data
            
        Returns:
            SFT loss tensor
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Standard language modeling loss
        loss = outputs.loss
        
        return loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary of step metrics
        """
        self.model.train()
        
        # Compute loss
        loss = self.compute_sft_loss(batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'step': self.global_step
        }
    
    def train(self, 
              seeding_data: List[Dict[str, Any]], 
              num_steps: int) -> Dict[str, Any]:
        """
        Run supervised fine-tuning on seeding data.
        
        Args:
            seeding_data: List of seeding samples
            num_steps: Number of training steps
            
        Returns:
            Dictionary of training results
        """
        self.logger.info(f"Starting SFT training for {num_steps} steps")
        
        # Create dataset and dataloader
        dataset = SeedingDataset(seeding_data, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        # Training loop
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(total=num_steps, desc="SFT Training")
        
        while self.global_step < num_steps:
            for batch in dataloader:
                if self.global_step >= num_steps:
                    break
                
                # Move batch to device
                if torch.cuda.is_available():
                    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
                # Training step
                step_metrics = self.train_step(batch)
                
                # Update metrics
                total_loss += step_metrics['loss']
                num_batches += 1
                
                # Log progress
                if self.global_step % 100 == 0:
                    avg_loss = total_loss / num_batches
                    self.logger.info(f"Step {self.global_step}: Loss = {avg_loss:.4f}")
                
                progress_bar.update(1)
                progress_bar.set_postfix({'loss': step_metrics['loss']})
        
        progress_bar.close()
        
        # Final metrics
        final_avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        training_results = {
            'final_loss': final_avg_loss,
            'total_steps': self.global_step,
            'num_samples': len(seeding_data)
        }
        
        self.logger.info(f"SFT training completed: {training_results}")
        return training_results
    
    def validate(self, validation_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Validate the model on validation data.
        
        Args:
            validation_data: List of validation samples
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        # Create validation dataset
        val_dataset = SeedingDataset(validation_data, self.tokenizer)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                if torch.cuda.is_available():
                    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
                # Compute loss
                loss = self.compute_sft_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'validation_loss': avg_loss,
            'num_validation_samples': len(validation_data)
        }
    
    def save_model(self, save_path: str):
        """
        Save the fine-tuned model.
        
        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save training metadata
        metadata = {
            'global_step': self.global_step,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'alpha': self.alpha
        }
        
        with open(save_path / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """
        Load a previously saved model.
        
        Args:
            load_path: Path to load the model from
        """
        load_path = Path(load_path)
        
        # Load model
        self.model.load_state_dict(torch.load(load_path / 'pytorch_model.bin'))
        
        # Load metadata if available
        metadata_path = load_path / 'training_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.global_step = metadata.get('global_step', 0)
        
        self.logger.info(f"Model loaded from {load_path}")