# Tabular Data Generation Project

This project uses generative AI techniques to create synthetic tabular datasets that make sense. The focus is on using the **TabGAN** library and processing data efficiently within an isolated environment.

---

## **1. Requirements**
- [Anaconda](https://www.anaconda.com/download): A Python distribution that simplifies package and environment management.
- [VSCode](https://code.visualstudio.com/): A lightweight and powerful code editor.
- Python 3.9 or higher.

---

## **2. Initial Setup**

### **Step 1: Install Anaconda**
1. Download Anaconda from [here](https://www.anaconda.com/download).
2. Install Anaconda by following the installation wizard.
3. Verify the installation:
   ```bash
   conda --version

### **Step 2: make the env **

conda env create -f env/environment.yml

if you mess up you can do

conda env remove -n tabular-data-gen


### **Step 3: activate it***

also, dont use vs code terminal, use the conda terminal.

conda activate tabular-data-gen

conda info --envs

### **Step 4: running scripts***

if we want for speed, use shrink script. make sure replace with ur path
python scripts/shrink_data.py


then do 
python scripts/preprocess.py

then do 
python scripts/generate_data.py







   
