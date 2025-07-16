"""
Memory optimization utilities for CRSP training.
Implements dynamic batch size control and memory monitoring.
"""

import torch
import psutil
import gc
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization strategies."""
    
    # Dynamic batch size settings
    initial_batch_size: int = 64
    min_batch_size: int = 8
    max_batch_size: int = 256
    batch_size_reduction_factor: float = 0.75
    batch_size_increase_factor: float = 1.1
    
    # Memory monitoring settings
    memory_warning_threshold: float = 0.85
    memory_critical_threshold: float = 0.95
    enable_memory_monitoring: bool = True
    
    # OOM recovery settings
    max_oom_retries: int = 3
    oom_cooldown_steps: int = 10
    enable_gradient_accumulation: bool = True
    max_gradient_accumulation_steps: int = 8
    
    # Memory cleanup settings
    cleanup_frequency: int = 50
    enable_aggressive_cleanup: bool = True
    
    # Fallback strategies
    enable_emergency_mode: bool = True
    emergency_batch_size: int = 4
    emergency_sequence_length: int = 1024


class MemoryMonitor:
    """Monitor GPU and system memory usage."""
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.memory_history = deque(maxlen=100)
        self.peak_memory = 0.0
        self.oom_events = 0
        
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information."""
        if not torch.cuda.is_available():
            return {"total": 0.0, "used": 0.0, "free": 0.0, "utilization": 0.0}
        
        try:
            # Get memory info for current device
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            cached_memory = torch.cuda.memory_reserved(device)
            free_memory = total_memory - cached_memory
            
            # Convert to GB
            total_gb = total_memory / (1024**3)
            allocated_gb = allocated_memory / (1024**3)
            cached_gb = cached_memory / (1024**3)
            free_gb = free_memory / (1024**3)
            
            utilization = cached_memory / total_memory
            
            return {
                "total": total_gb,
                "allocated": allocated_gb,
                "cached": cached_gb,
                "free": free_gb,
                "utilization": utilization
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return {"total": 0.0, "allocated": 0.0, "cached": 0.0, "free": 0.0, "utilization": 0.0}
    
    def get_system_memory_info(self) -> Dict[str, float]:
        """Get current system memory information."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total": memory.total / (1024**3),
                "used": memory.used / (1024**3),
                "free": memory.available / (1024**3),
                "utilization": memory.percent / 100.0
            }
        except Exception as e:
            logger.warning(f"Failed to get system memory info: {e}")
            return {"total": 0.0, "used": 0.0, "free": 0.0, "utilization": 0.0}
    
    def update_memory_stats(self) -> Dict[str, Any]:
        """Update and return current memory statistics."""
        gpu_info = self.get_gpu_memory_info()
        system_info = self.get_system_memory_info()
        
        # Update peak memory
        if gpu_info["utilization"] > self.peak_memory:
            self.peak_memory = gpu_info["utilization"]
        
        # Add to history
        self.memory_history.append({
            "timestamp": time.time(),
            "gpu": gpu_info,
            "system": system_info
        })
        
        return {
            "gpu": gpu_info,
            "system": system_info,
            "peak_utilization": self.peak_memory,
            "oom_events": self.oom_events
        }
    
    def is_memory_pressure(self) -> Tuple[bool, str]:
        """Check if system is under memory pressure."""
        gpu_info = self.get_gpu_memory_info()
        
        if gpu_info["utilization"] >= self.config.memory_critical_threshold:
            return True, "critical"
        elif gpu_info["utilization"] >= self.config.memory_warning_threshold:
            return True, "warning"
        else:
            return False, "normal"
    
    def log_memory_stats(self, prefix: str = ""):
        """Log current memory statistics."""
        stats = self.update_memory_stats()
        gpu = stats["gpu"]
        system = stats["system"]
        
        logger.info(
            f"{prefix}Memory Stats - "
            f"GPU: {gpu['allocated']:.2f}GB/{gpu['total']:.2f}GB "
            f"({gpu['utilization']*100:.1f}%), "
            f"System: {system['used']:.2f}GB/{system['total']:.2f}GB "
            f"({system['utilization']*100:.1f}%)"
        )


class DynamicBatchSizeController:
    """
    Dynamic batch size controller that automatically adjusts batch sizes
    based on memory usage and OOM events.
    """
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.monitor = MemoryMonitor(config)
        
        # Current batch size state
        self.current_batch_size = config.initial_batch_size
        self.gradient_accumulation_steps = 1
        
        # OOM tracking
        self.oom_count = 0
        self.last_oom_step = -1
        self.consecutive_successes = 0
        
        # Performance tracking
        self.batch_size_history = deque(maxlen=100)
        self.throughput_history = deque(maxlen=50)
        
        logger.info(f"Initialized DynamicBatchSizeController with initial batch size: {self.current_batch_size}")
    
    def handle_oom_error(self, step: int, error: Exception) -> Tuple[int, int]:
        """
        Handle OOM error by reducing batch size and increasing gradient accumulation.
        
        Returns:
            Tuple of (new_batch_size, new_gradient_accumulation_steps)
        """
        self.oom_count += 1
        self.monitor.oom_events += 1
        self.last_oom_step = step
        self.consecutive_successes = 0
        
        # Calculate new batch size
        old_batch_size = self.current_batch_size
        self.current_batch_size = max(
            self.config.min_batch_size,
            int(self.current_batch_size * self.config.batch_size_reduction_factor)
        )
        
        # Increase gradient accumulation to maintain effective batch size
        if self.config.enable_gradient_accumulation:
            effective_batch_size = old_batch_size
            self.gradient_accumulation_steps = min(
                self.config.max_gradient_accumulation_steps,
                max(1, effective_batch_size // self.current_batch_size)
            )
        
        # Emergency mode for severe OOM
        if self.oom_count >= self.config.max_oom_retries:
            if self.config.enable_emergency_mode:
                logger.warning(f"Entering emergency mode after {self.oom_count} OOM events")
                self.current_batch_size = self.config.emergency_batch_size
                self.gradient_accumulation_steps = self.config.max_gradient_accumulation_steps
        
        logger.warning(
            f"OOM Error #{self.oom_count} at step {step}. "
            f"Reduced batch size: {old_batch_size} -> {self.current_batch_size}, "
            f"Gradient accumulation: {self.gradient_accumulation_steps}"
        )
        
        # Force memory cleanup
        self._aggressive_memory_cleanup()
        
        return self.current_batch_size, self.gradient_accumulation_steps
    
    def try_increase_batch_size(self, step: int) -> Tuple[int, int]:
        """
        Try to increase batch size if conditions are favorable.
        
        Returns:
            Tuple of (new_batch_size, new_gradient_accumulation_steps)
        """
        # Check if we should try to increase
        if (step - self.last_oom_step < self.config.oom_cooldown_steps or
            self.consecutive_successes < 10 or
            self.current_batch_size >= self.config.max_batch_size):
            return self.current_batch_size, self.gradient_accumulation_steps
        
        # Check memory pressure
        is_pressure, level = self.monitor.is_memory_pressure()
        if is_pressure:
            return self.current_batch_size, self.gradient_accumulation_steps
        
        # Try to increase batch size
        old_batch_size = self.current_batch_size
        new_batch_size = min(
            self.config.max_batch_size,
            int(self.current_batch_size * self.config.batch_size_increase_factor)
        )
        
        # Reduce gradient accumulation if possible
        if new_batch_size > old_batch_size and self.gradient_accumulation_steps > 1:
            self.gradient_accumulation_steps = max(
                1,
                self.gradient_accumulation_steps - 1
            )
        
        if new_batch_size != old_batch_size:
            logger.info(
                f"Increasing batch size: {old_batch_size} -> {new_batch_size}, "
                f"Gradient accumulation: {self.gradient_accumulation_steps}"
            )
            self.current_batch_size = new_batch_size
        
        return self.current_batch_size, self.gradient_accumulation_steps
    
    def record_successful_step(self, step: int, processing_time: float = None):
        """Record a successful training step."""
        self.consecutive_successes += 1
        
        # Record batch size history
        self.batch_size_history.append({
            "step": step,
            "batch_size": self.current_batch_size,
            "gradient_accumulation": self.gradient_accumulation_steps
        })
        
        # Record throughput if provided
        if processing_time is not None:
            throughput = self.current_batch_size / processing_time
            self.throughput_history.append(throughput)
    
    def get_current_config(self) -> Dict[str, int]:
        """Get current batch size configuration."""
        return {
            "batch_size": self.current_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.current_batch_size * self.gradient_accumulation_steps
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get controller statistics."""
        memory_stats = self.monitor.update_memory_stats()
        
        avg_throughput = 0.0
        if self.throughput_history:
            avg_throughput = sum(self.throughput_history) / len(self.throughput_history)
        
        return {
            "current_batch_size": self.current_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.current_batch_size * self.gradient_accumulation_steps,
            "oom_events": self.oom_count,
            "consecutive_successes": self.consecutive_successes,
            "avg_throughput": avg_throughput,
            "memory": memory_stats
        }
    
    def _aggressive_memory_cleanup(self):
        """Perform aggressive memory cleanup after OOM."""
        if not self.config.enable_aggressive_cleanup:
            return
        
        logger.info("Performing aggressive memory cleanup...")
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Log memory after cleanup
        self.monitor.log_memory_stats("Post-cleanup ")


def create_memory_optimized_batch_processor():
    """Factory function to create memory-optimized batch processor."""
    config = MemoryOptimizationConfig()
    controller = DynamicBatchSizeController(config)
    
    def process_batch_with_oom_handling(batch_fn, batch, step, *args, **kwargs):
        """
        Process batch with automatic OOM handling and batch size adjustment.
        
        Args:
            batch_fn: Function to process the batch
            batch: The batch to process
            step: Current training step
            *args, **kwargs: Additional arguments for batch_fn
        
        Returns:
            Result of batch_fn or None if failed
        """
        start_time = time.time()
        
        try:
            # Try to increase batch size if conditions are favorable
            controller.try_increase_batch_size(step)
            
            # Process the batch
            result = batch_fn(batch, *args, **kwargs)
            
            # Record successful step
            processing_time = time.time() - start_time
            controller.record_successful_step(step, processing_time)
            
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM Error at step {step}: {e}")
            
            # Handle OOM by reducing batch size
            new_batch_size, new_grad_accum = controller.handle_oom_error(step, e)
            
            # Log memory stats
            controller.monitor.log_memory_stats("OOM Event ")
            
            # Return None to indicate failure - caller should retry with smaller batch
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error during batch processing at step {step}: {e}")
            raise
    
    return controller, process_batch_with_oom_handling