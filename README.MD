
# Tabular Data Generation Project

This project uses generative AI techniques to create synthetic tabular datasets. It focuses on leveraging the **TabGAN** library and processing data efficiently within an isolated environment.

---

## 1. Requirements

- [Anaconda](https://www.anaconda.com/download): A Python distribution for managing packages and environments.
- [VSCode](https://code.visualstudio.com/): A lightweight, powerful code editor.
- Python 3.9 or higher.

---

## 2. Initial Setup

```bash
# Step 1: Install Anaconda
# Download Anaconda from https://www.anaconda.com/download
# Follow the installation wizard to install Anaconda

# Verify the installation
conda --version

# Step 2: Create the Environment
# Create the environment from the environment.yml file
conda env create -f env/environment.yml

# If you need to start over, remove the environment
conda env remove -n tabular-data-gen

# Step 3: Activate the Environment
# Activate the environment
conda activate tabular-data-gen

# Confirm the environment is activated
conda info --envs
```

## 2. Do the Thing

# Note: Avoid using the VSCode terminal; use the Anaconda terminal instead.

# If needed, shrink the dataset for faster processing
# Replace "path/to/raw_data.csv" with your actual path

```bash
python scripts/shrink_data.py
```

# Preprocess the data
```bash
python scripts/preprocess.py
```

# Generate synthetic data
```bash
python scripts/generate_data.py
```

# Compare synthetic data with the original
```bash
python scripts/compare_data.py
```