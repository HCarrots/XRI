"""Command-line entry point."""
import argparse
import json
import sys
from .config import load_config
from .pipeline import run, validate_inputs


def main():
    parser = argparse.ArgumentParser(description="XRI: foil calibration and elastic-integrated joint 3D reconstruction")
    parser.add_argument("--config", required=True, help="JSON config; relative paths resolve from this file")
    parser.add_argument("--validate-only", action="store_true", help="Inspect input metadata/ROIs without creating results")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        result = validate_inputs(config) if args.validate_only else run(config)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, OSError, RuntimeError) as error:
        print(f"XRI ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
