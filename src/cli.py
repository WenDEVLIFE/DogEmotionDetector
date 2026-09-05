"""Command-line interface for Dog Emotion Detector."""

import argparse
import sys
from pathlib import Path

from src.utils.config import load_config


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Dog Emotion Detector - Detect emotions in dogs from images"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        print(f"Configuration loaded from: {args.config}")
        print(f"Configuration: {config}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()