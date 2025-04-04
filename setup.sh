#!/bin/bash

# Exit on error
set -e

ENV_NAME="llm_env"

# Check if the virtual environment exists, if not, create it
if [ ! -d "$ENV_NAME" ]; then
    echo "Creating virtual environment '$ENV_NAME'..."
    python -m venv "$ENV_NAME"
else
    echo "Virtual environment '$ENV_NAME' already exists."
fi

# Activate the virtual environment (Windows)
echo "Activating virtual environment '$ENV_NAME'..."
source "$ENV_NAME/Scripts/activate"

# Display the active environment
echo "Currently in environment: $(which python)"

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies from requirements.txt if the file exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping package installation."
fi

echo "Setup completed successfully!"

