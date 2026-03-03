"""
Logging utilities for SRIP AI Sustainability project.
Provides consistent logging across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import yaml


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    log_format: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
        console_output: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if logger.handlers:
        return logger
    
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str = "C:\\Opensource\\DELHI_SAR\\srip_ai_sustainability\\configs\\config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_logger_from_config(config: dict, module_name: str) -> logging.Logger:
    """
    Create logger using configuration from config.yaml.
    
    Args:
        config: Configuration dictionary
        module_name: Name of the module requesting the logger
    
    Returns:
        Configured logger
    """
    log_config = config.get('logging', {})
    project_root = config.get('paths', {}).get('project_root', '.')
    
    log_file = str(Path(project_root) / log_config.get('log_file', 'srip.log'))
    
    return setup_logger(
        name=module_name,
        log_file=log_file,
        level=log_config.get('level', 'INFO'),
        log_format=log_config.get('format'),
        console_output=log_config.get('console_output', True)
    )
