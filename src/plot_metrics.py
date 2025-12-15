"""
Script to plot training metrics from saved JSON files.
Usage: python src/plot_metrics.py --metrics_file training_logs/metrics/training_metrics.json
"""

import argparse
from training_metrics import plot_from_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training metrics from saved JSON file')
    parser.add_argument('--metrics_file', type=str, required=True,
                        help='Path to training metrics JSON file')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save plots (default: same as metrics file)')
    parser.add_argument('--show', action='store_true',
                        help='Display plots (default: False)')
    
    args = parser.parse_args()
    
    plot_from_file(args.metrics_file, save_dir=args.save_dir, show=args.show)
    print("Plots generated successfully!")

