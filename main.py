"""
Main entry point for SRIP AI Sustainability project.
Runs Q1, Q2, Q3 modules individually or as a complete pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger, load_config


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="SRIP AI Sustainability: Delhi Air Quality Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--run',
        type=str,
        choices=['q1', 'q2', 'q3', 'train', 'evaluate', 'all'],
        default='all',
        help='Module to run: q1 (spatial grid), q2 (dataset), q3 (model), train, evaluate, or all'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config.yaml file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def run_q1(config_path: str, logger) -> dict:
    """
    Run Q1 spatial grid analysis module.
    
    Args:
        config_path: Path to config file
        logger: Logger instance
    
    Returns:
        Statistics dictionary
    """
    logger.info("=" * 60)
    logger.info("Running Q1: Spatial Grid Analysis")
    logger.info("=" * 60)
    
    from src.q1_spatial_grid import run_q1 as q1_main
    
    stats = q1_main(config_path)
    
    return stats


def run_q2(config_path: str, logger) -> dict:
    """
    Run Q2 dataset builder module.
    
    Args:
        config_path: Path to config file
        logger: Logger instance
    
    Returns:
        Statistics dictionary
    """
    logger.info("=" * 60)
    logger.info("Running Q2: Dataset Builder")
    logger.info("=" * 60)
    
    from src.q2_dataset_builder import run_q2 as q2_main
    
    stats = q2_main(config_path)
    
    return stats


def run_q3(config_path: str, logger) -> dict:
    """
    Run Q3 model training module.
    
    Args:
        config_path: Path to config file
        logger: Logger instance
    
    Returns:
        Results dictionary
    """
    logger.info("=" * 60)
    logger.info("Running Q3: Model Training")
    logger.info("=" * 60)
    
    from src.train import run_training as train_main
    
    results = train_main(config_path)
    
    return results


def run_evaluation(config_path: str, logger) -> dict:
    """
    Run evaluation module.
    
    Args:
        config_path: Path to config file
        logger: Logger instance
    
    Returns:
        Metrics dictionary
    """
    logger.info("=" * 60)
    logger.info("Running Model Evaluation")
    logger.info("=" * 60)
    
    from src.evaluate import run_evaluation as eval_main
    
    metrics = eval_main(config_path)
    
    return metrics


def run_pipeline(config_path: str, logger) -> dict:
    """
    Run complete end-to-end pipeline.
    
    Args:
        config_path: Path to config file
        logger: Logger instance
    
    Returns:
        Complete results dictionary
    """
    logger.info("=" * 70)
    logger.info("Starting Complete Pipeline: Q1 -> Q2 -> Q3 -> Evaluation")
    logger.info("=" * 70)
    
    all_results = {}
    
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: SPATIAL GRID ANALYSIS")
    logger.info("=" * 70)
    q1_stats = run_q1(config_path, logger)
    all_results['q1'] = q1_stats
    
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: DATASET CREATION")
    logger.info("=" * 70)
    q2_stats = run_q2(config_path, logger)
    all_results['q2'] = q2_stats
    
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: MODEL TRAINING")
    logger.info("=" * 70)
    train_results = run_q3(config_path, logger)
    all_results['train'] = train_results
    
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: MODEL EVALUATION")
    logger.info("=" * 70)
    eval_metrics = run_evaluation(config_path, logger)
    all_results['evaluation'] = eval_metrics
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Grid cells created: {q1_stats.get('num_grid_cells', 'N/A')}")
    logger.info(f"Ground stations: {q1_stats.get('num_stations', 'N/A')}")
    logger.info(f"Training samples: {q2_stats.get('train_samples', 'N/A')}")
    logger.info(f"Test samples: {q2_stats.get('test_samples', 'N/A')}")
    logger.info(f"Best validation accuracy: {train_results.get('best_val_accuracy', 'N/A'):.2f}%")
    logger.info(f"Test accuracy: {eval_metrics.get('accuracy', 'N/A'):.4f}")
    logger.info("=" * 70)
    
    return all_results


def main() -> None:
    """
    Main entry point.
    """
    args = parse_arguments()
    
    if args.config:
        config_path = args.config
    else:
        base_dir = Path(__file__).parent
        config_path = str(base_dir / 'configs' / 'config.yaml')
    
    logger = setup_logger(
        name="Main",
        level="DEBUG" if args.verbose else "INFO",
        console_output=True
    )
    
    logger.info(f"Config file: {config_path}")
    logger.info(f"Running module: {args.run}")
    
    config = load_config(config_path)
    
    base_path = Path(config['paths']['project_root'])
    for subdir in ['raw', 'processed', 'outputs']:
        (base_path / 'srip_ai_sustainability' / 'data' / subdir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    try:
        if args.run == 'q1':
            results = run_q1(config_path, logger)
        elif args.run == 'q2':
            results = run_q2(config_path, logger)
        elif args.run in ['q3', 'train']:
            results = run_q3(config_path, logger)
        elif args.run == 'evaluate':
            results = run_evaluation(config_path, logger)
        elif args.run == 'all':
            results = run_pipeline(config_path, logger)
        else:
            logger.error(f"Unknown module: {args.run}")
            sys.exit(1)
        
        logger.info("=" * 60)
        logger.info("Execution completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)
    
    return


if __name__ == "__main__":
    main()
