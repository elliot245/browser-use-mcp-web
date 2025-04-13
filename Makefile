# Makefile for MCP Browser Use Project

# Variables
PYTHON = python3
DOCKER_IMAGE = browser-use-mcp-web:latest
DOCKERFILE = Dockerfile

# Default target
.DEFAULT_GOAL := help

# Help target
help:
	@echo "Available targets:"
	@echo "  setup          - Set up the virtual environment and install dependencies"
	@echo "  build          - Build the project using hatchling"
	@echo "  test           - Run tests"
	@echo "  run            - Run the project"
	@echo "  docker-build   - Build the Docker image"
	@echo "  docker-run     - Run the Docker container"
	@echo "  clean          - Clean up build artifacts and virtual environment"

# Set up the virtual environment and install dependencies
setup:
	@echo "Setting up the virtual environment and installing dependencies..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync

# Build the project using hatchling
build:
	@echo "Building the project using hatchling..."
	uv build

# Run the project
dev:
	@echo "Dev the project..."
	uv run fastmcp dev ./src/mcp_browser_use/server.py

# Run the project
run:
	@echo "Running the project..."
	uv run mcp-browser-use

# Build the Docker image
docker-build:
	@echo "Building the Docker image..."
	docker build -t $(DOCKER_IMAGE) -f $(DOCKERFILE) .

# Run the Docker container
docker-run:
	@echo "Running the Docker container..."
	docker run --rm -it $(DOCKER_IMAGE)

# Clean up build artifacts and virtual environment
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	rm -rf .hatch
