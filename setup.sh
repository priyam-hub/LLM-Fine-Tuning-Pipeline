#!/bin/bash

# Exit on error
set -e

echo "Select environment type:"
echo "1. Create Python Environment"
echo "2. Create Conda Environment"
read -p "Enter your choice (1 or 2): " choice

ENV_NAME="llm_env"

if [ "$choice" == "1" ]; then
    # Python Virtual Environment
    echo "You selected Python virtual environment."

    if [ ! -d "$ENV_NAME" ]; then
        echo "Creating Python virtual environment '$ENV_NAME'..."
        python -m venv "$ENV_NAME"
    else
        echo "Virtual environment '$ENV_NAME' already exists."
    fi

    echo "Activating Python virtual environment '$ENV_NAME'..."
    source "$ENV_NAME/Scripts/activate"

elif [ "$choice" == "2" ]; then
    # Conda Environment
    echo "You selected Conda environment."

    if conda env list | grep -q "$ENV_NAME"; then
        echo "Conda environment '$ENV_NAME' already exists."
    else
        echo "Creating Conda environment '$ENV_NAME'..."
        conda create -y -n "$ENV_NAME" python=3.10
    fi

    echo "Activating Conda environment '$ENV_NAME'..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

else
    echo "Invalid choice. Exiting."
    exit 1
fi

# Show active Python environment
echo "Currently in environment: $(which python)"

# Upgrade pip and install dependencies
echo "Upgrading pip..."
python -m pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping package installation."
fi

echo "Setup completed successfully!"
