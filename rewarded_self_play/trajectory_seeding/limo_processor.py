"""
LIMO dataset processor for trajectory seeding.

This module handles loading, processing, and formatting of the LIMO dataset
for use in trajectory seeding phase of CRSP training.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datasets import load_dataset
import random


class LIMOProcessor:
    """
    Processor for LIMO dataset integration.
    
    Handles loading the LIMO dataset and converting it to the format
    required for trajectory seeding in CRSP.
    """
    
    def __init__(self, dataset_path: str = "GAIR/LIMO"):
        """
        Initialize LIMO processor.
        
        Args:
            dataset_path: Path or HuggingFace dataset name for LIMO
        """
        self.dataset_path = dataset_path
        self.logger = logging.getLogger(__name__)
        self._dataset = None
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load LIMO dataset from HuggingFace or local path.
        
        Returns:
            List of LIMO samples
        """
        try:
            self.logger.info(f"Loading LIMO dataset from {self.dataset_path}")
            
            # Try loading from HuggingFace
            dataset = load_dataset(self.dataset_path)
            
            # Convert to list of dictionaries
            if 'train' in dataset:
                data = list(dataset['train'])
            else:
                # If no train split, use the first available split
                split_name = list(dataset.keys())[0]
                data = list(dataset[split_name])
            
            self._dataset = data
            self.logger.info(f"Loaded {len(data)} samples from LIMO dataset")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load LIMO dataset: {e}")
            
            # Fallback: create sample data for testing
            self.logger.warning("Using fallback sample data for testing")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> List[Dict[str, Any]]:
        """
        Create sample LIMO-style data for testing purposes.
        
        Returns:
            List of sample data in LIMO format
        """
        sample_data = [
            {
                "question": "Find the last three digits of the product of the positive roots of $\\sqrt{1995 \\cdot x} = x$.",
                "solution": "We need to solve $\\sqrt{1995 \\cdot x} = x$ for positive values of $x$.\n\nSquaring both sides: $1995 \\cdot x = x^2$\n\nRearranging: $x^2 - 1995x = 0$\n\nFactoring: $x(x - 1995) = 0$\n\nSo $x = 0$ or $x = 1995$.\n\nSince we want positive roots, we have $x = 1995$.\n\nThe product of positive roots is just $1995$.\n\nThe last three digits of $1995$ are $995$.",
                "answer": "995"
            },
            {
                "question": "Compute $\\sum_{k=1}^{100} \\frac{1}{k(k+1)}$.",
                "solution": "We can use partial fractions to decompose $\\frac{1}{k(k+1)}$.\n\n$\\frac{1}{k(k+1)} = \\frac{A}{k} + \\frac{B}{k+1}$\n\nMultiplying by $k(k+1)$: $1 = A(k+1) + Bk$\n\nSetting $k = 0$: $1 = A$\nSetting $k = -1$: $1 = -B$, so $B = -1$\n\nTherefore: $\\frac{1}{k(k+1)} = \\frac{1}{k} - \\frac{1}{k+1}$\n\nNow we can compute the sum:\n$\\sum_{k=1}^{100} \\frac{1}{k(k+1)} = \\sum_{k=1}^{100} \\left(\\frac{1}{k} - \\frac{1}{k+1}\\right)$\n\nThis is a telescoping series:\n$= \\left(\\frac{1}{1} - \\frac{1}{2}\\right) + \\left(\\frac{1}{2} - \\frac{1}{3}\\right) + \\cdots + \\left(\\frac{1}{100} - \\frac{1}{101}\\right)$\n\n$= 1 - \\frac{1}{101} = \\frac{100}{101}$",
                "answer": "100/101"
            }
        ]
        
        return sample_data
    
    def validate_limo_format(self, sample: Dict[str, Any]) -> bool:
        """
        Validate that a sample follows the expected LIMO format.
        
        Args:
            sample: Sample to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['question', 'solution', 'answer']
        
        for field in required_fields:
            if field not in sample:
                self.logger.warning(f"Sample missing required field: {field}")
                return False
            
            if not isinstance(sample[field], str) or not sample[field].strip():
                self.logger.warning(f"Sample has invalid {field} field")
                return False
        
        return True
    
    def create_seeding_data(self, limo_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert LIMO data to seeding format for CRSP.
        
        Args:
            limo_data: Raw LIMO dataset
            
        Returns:
            List of formatted seeding samples
        """
        seeding_samples = []
        
        for sample in limo_data:
            # Validate format
            if not self.validate_limo_format(sample):
                continue
            
            # Format for CRSP trajectory seeding
            formatted_sample = {
                'prompt': [
                    {
                        'role': 'user',
                        'content': sample['question']
                    }
                ],
                'solution': sample['solution'],
                'answer': sample['answer'],
                'source': 'LIMO'
            }
            
            seeding_samples.append(formatted_sample)
        
        self.logger.info(f"Created {len(seeding_samples)} seeding samples from {len(limo_data)} LIMO samples")
        return seeding_samples
    
    def get_validation_samples(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        """
        Get a subset of samples for validation.
        
        Args:
            num_samples: Number of validation samples to return
            
        Returns:
            List of validation samples
        """
        if self._dataset is None:
            self.load_dataset()
        
        # Create seeding data if not already done
        seeding_data = self.create_seeding_data(self._dataset)
        
        # Return random subset for validation
        if len(seeding_data) <= num_samples:
            return seeding_data
        
        return random.sample(seeding_data, num_samples)
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded dataset.
        
        Returns:
            Dictionary of dataset statistics
        """
        if self._dataset is None:
            self.load_dataset()
        
        stats = {
            'total_samples': len(self._dataset),
            'avg_question_length': 0,
            'avg_solution_length': 0,
            'avg_answer_length': 0
        }
        
        if self._dataset:
            question_lengths = [len(sample.get('question', '')) for sample in self._dataset]
            solution_lengths = [len(sample.get('solution', '')) for sample in self._dataset]
            answer_lengths = [len(sample.get('answer', '')) for sample in self._dataset]
            
            stats['avg_question_length'] = sum(question_lengths) / len(question_lengths)
            stats['avg_solution_length'] = sum(solution_lengths) / len(solution_lengths)
            stats['avg_answer_length'] = sum(answer_lengths) / len(answer_lengths)
        
        return stats