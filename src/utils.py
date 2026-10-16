"""
Utility Module
Shared utilities for the DDoS detection research project: timer, logger, config
loader, seed setter, and directory management.
"""

import time
import json
import os
import sys
import logging
import hashlib
from datetime import datetime
from typing import Any, Dict, Optional, Union


class Timer:
    """
    Simple timer utility for measuring execution time.
    """
    
    def __init__(self):
        self._start_time = None
        self._elapsed = 0.0
        self._running = False
    
    def start(self):
        """Start the timer."""
        self._start_time = time.time()
        self._running = True
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time in seconds."""
        if self._running and self._start_time is not None:
            self._elapsed = time.time() - self._start_time
            self._running = False
        return self._elapsed
    
    def reset(self):
        """Reset the timer."""
        self._elapsed = 0.0
        self._running = False
        self._start_time = None
    
    def elapsed(self) -> float:
        """Return elapsed time."""
        if self._running:
            return time.time() - self._start_time + self._elapsed
        return self._elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


class ConfigLoader:
    """
    Load and manage configuration from YAML or JSON files.
    """
    
    @staticmethod
    def load_yaml(filepath: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Parameters
        ----------
        filepath : str
            Path to YAML configuration file
        
        Returns
        -------
        dict
            Loaded configuration
        """
        try:
            import yaml
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
            return config if config else {}
        except ImportError:
            print("  PyYAML not installed, trying JSON fallback...")
            return ConfigLoader.load_json(filepath)
        except Exception as e:
            print(f"  Error loading YAML: {e}, trying JSON fallback...")
            return ConfigLoader.load_json(filepath)
    
    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to JSON configuration file
        
        Returns
        -------
        dict
            Loaded configuration
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], filepath: str, format: str = 'yaml') -> None:
        """
        Save configuration to file.
        
        Parameters
        ----------
        config : dict
            Configuration to save
        filepath : str
            Output file path
        format : str
            'yaml' or 'json'
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        if format.lower() == 'yaml':
            try:
                import yaml
                with open(filepath, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                print(f"  Saved config (YAML): {filepath}")
            except ImportError:
                # Fall back to JSON
                ConfigLoader.save_config(config, filepath, format='json')
        elif format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"  Saved config (JSON): {filepath}")
    
    @staticmethod
    def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries (deep merge).
        
        Parameters
        ----------
        base : dict
            Base configuration
        override : dict
            Override configuration (takes precedence)
        
        Returns
        -------
        dict
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result


class Logger:
    """
    Configurable logger setup with file and console handlers.
    """
    
    @staticmethod
    def setup_logger(name: str = "ddos_detection",
                      log_file: Optional[str] = None,
                      level: int = logging.INFO,
                      format_string: Optional[str] = None) -> logging.Logger:
        """
        Set up a logger with console and optional file output.
        
        Parameters
        ----------
        name : str
            Logger name
        log_file : str, optional
            Path to log file (None = console only)
        level : int
            Logging level (default: INFO)
        format_string : str, optional
            Custom format string
        
        Returns
        -------
        logging.Logger
            Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Avoid adding handlers if already configured
        if logger.handlers:
            return logger
        
        # Default format
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        formatter = logging.Formatter(format_string)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


def set_seed(seed: int, 
             seed_numpy: bool = True, 
             seed_python: bool = True,
             seed_random: bool = True) -> None:
    """
    Set random seeds for reproducibility across libraries.
    
    Parameters
    ----------
    seed : int
        Seed value
    seed_numpy : bool
        Whether to seed numpy (default: True)
    seed_python : bool
        Whether to seed Python's random (default: True)
    seed_random : bool
        Whether to seed PyTorch/random if available (default: True)
    """
    if seed_python:
        import random
        random.seed(seed)
    
    if seed_numpy:
        import numpy as np
        np.random.seed(seed)
    
    # Note: PyTorch seeding would go here if used


def ensure_dir(filepath: str) -> str:
    """
    Ensure directory exists, creating it if necessary.
    Returns the directory path.
    
    Parameters
    ----------
    filepath : str
        File path (directory portion will be created)
    
    Returns
    -------
    str
        The directory path that was ensured
    """
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    return directory


def generate_experiment_id(config: Dict[str, Any]) -> str:
    """
    Generate a unique experiment ID from configuration hash.
    
    Parameters
    ----------
    config : dict
        Experiment configuration
    
    Returns
    -------
    str
        Unique experiment identifier (8-character hash)
    """
    # Stringify config and hash it
    config_str = json.dumps(config, sort_keys=True, default=str)
    experiment_hash = hashlib.sha256(config_str.encode()).hexdigest()
    return experiment_hash[:8].upper()


def format_metrics_report(metrics_dict: Dict[str, float], 
                         decimal_places: int = 4) -> str:
    """
    Format metrics dictionary as a human-readable string.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metric name -> value
    decimal_places : int
        Number of decimal places
    
    Returns
    -------
    str
        Formatted report string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("METRICS REPORT")
    lines.append("=" * 50)
    
    for key, value in sorted(metrics_dict.items()):
        if isinstance(value, float):
            lines.append(f"  {key:25s}: {value:.{decimal_places}f}")
        else:
            lines.append(f"  {key:25s}: {value}")
    
    lines.append("=" * 50)
    return "\n".join(lines)


def timer_decorator(func):
    """
    Decorator to automatically time function execution.
    
    Usage:
        @timer_decorator
        def my_function():
            ...
    """
    def wrapper(*args, **kwargs):
        timer = Timer()
        timer.start()
        result = func(*args, **kwargs)
        elapsed = timer.stop()
        print(f"  {func.__name__} executed in {elapsed:.4f} seconds")
        return result
    return wrapper


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that returns default if denominator is zero.
    
    Parameters
    ----------
    numerator : float
        Numerator
    denominator : float
        Denominator
    default : float
        Return value if denominator is 0
    
    Returns
    -------
    float
        Division result or default
    """
    if denominator == 0 or (isinstance(denominator, float) and np.isnan(denominator)):
        return default
    return numerator / denominator


# Auto-run test when module is imported
if __name__ == "__main__":
    print("Testing utility module...")
    
    # Test timer
    with Timer() as t:
        time.sleep(0.1)
    print(f"  Timer test: elapsed = {t.elapsed():.4f}s")
    
    # Test seed setter
    set_seed(42)
    import numpy as np
    test_np = np.random.rand(5)
    print(f"  Seed test: first numpy rand = {test_np[0]:.4f}")
    
    # Test config loader
    test_config = {'learning_rate': 0.001, 'epochs': 100, 'batch_size': 32}
    ConfigLoader.save_config(test_config, 'test_config.json', format='json')
    loaded = ConfigLoader.load_json('test_config.json')
    print(f"  Config load test: {loaded == test_config}")
    
    # Test safe divide
    print(f"  Safe divide (1/2) = {safe_divide(1, 2):.4f}")
    print(f"  Safe divide (1/0) = {safe_divide(1, 0):.4f}")
    
    # Test experiment ID
    eid = generate_experiment_id(test_config)
    print(f"  Experiment ID: {eid}")
    
    # Test logger
    logger = Logger.setup_logger('test_logger', level=logging.INFO)
    logger.info("Test log message")
    
    # Cleanup test file
    if os.path.exists('test_config.json'):
        os.remove('test_config.json')
    
    print("\n✓ Utility module tests passed!")