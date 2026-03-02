.PHONY: help install install-dev clean test format lint

PYTHON := python3
PIP := pip

help:
	@echo "VLA Arm Deployment - Available commands:"
	@echo ""
	@echo "  make install      - Install package"
	@echo "  make install-dev  - Install package with dev dependencies"
	@echo "  make clean        - Clean generated files"
	@echo "  make format       - Format code with ruff"
	@echo "  make lint         - Lint code with ruff"
	@echo "  make test         - Run tests"
	@echo ""

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

format:
	ruff format scripts/ src/ tasks/

lint:
	ruff check scripts/ src/ tasks/

test:
	pytest tests/ -v

# Recording commands
record-stack:
	@echo "Recording Franka Stack task..."
	@echo "Run: ./isaaclab.sh -p scripts/record_demos.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --dataset_file datasets/franka_stack.hdf5"

record-place:
	@echo "Recording Franka Place-into-Bin task..."
	@echo "Run: ./isaaclab.sh -p scripts/record_demos.py --task Isaac-Place-Bin-Franka-IK-Rel-v0 --dataset_file datasets/franka_place_bin.hdf5"

# Dataset analysis
analyze:
	@echo "Usage: python scripts/analyze_dataset.py datasets/<file>.hdf5"

# Conversion
convert-lerobot:
	@echo "Usage: python scripts/convert_to_lerobot.py --input datasets/<file>.hdf5 --output datasets/lerobot/<name>"