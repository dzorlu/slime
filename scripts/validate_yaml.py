#!/usr/bin/env python3
"""Validate YAML configuration files."""
import sys
import yaml


def validate_yaml_file(filepath):
    """Validate a single YAML file."""
    try:
        with open(filepath, "r") as f:
            yaml.safe_load(f)
        print(f"✓ {filepath} is valid YAML")
        return True
    except yaml.YAMLError as e:
        print(f"✗ {filepath} has YAML errors:")
        print(f"  {e}")
        return False
    except Exception as e:
        print(f"✗ {filepath} error: {e}")
        return False


def main():
    """Validate all SkyPilot YAML files."""
    yaml_files = [
        "sky_configs/train.yaml",
        "sky_configs/examples/train_glm4_9b.yaml",
        "sky_configs/examples/train_glm4_355b.yaml",
    ]

    all_valid = True
    for yaml_file in yaml_files:
        if not validate_yaml_file(yaml_file):
            all_valid = False

    if all_valid:
        print("\n✓ All YAML files are valid!")
        sys.exit(0)
    else:
        print("\n✗ Some YAML files have errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
