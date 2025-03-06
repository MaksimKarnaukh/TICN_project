# Project Setup Guide

This guide provides instructions for setting up the Conda environments required for this project. You can choose between two already-made environments:
- `environment_tf215.yml` (TensorFlow 2.15)
- `environment_tf217.yml` (TensorFlow 2.17 + keras-nightly)

Follow the steps below to set up the project.

---

## **1. Install Conda**
If you haven’t installed Conda yet, download and install it from one of the following links:
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Lightweight option)
- [Anaconda](https://www.anaconda.com/products/distribution) (Includes additional tools)
- [Miniforge](https://conda-forge.org/download/) (preferred)

---

## **2. Create and Activate the Environment (Automatic)**
You can create and activate a Conda environment using the provided `.yml` files.

### **Option A: TensorFlow 2.15 Environment**
```bash
conda env create -f environment_tf215.yml
conda activate myenv_tf215
```

### **Option B: TensorFlow 2.17 Environment**
```bash
conda env create -f environment_tf217.yml
conda activate myenv_tf217
```

---

## **3. Manual Installation (if preferred)**

### **Step 1: Create and Activate the Environment**
```bash
conda create -n myenv_tf215 python=3.11
conda activate myenv_tf215
```

### **Step 2: Install Dependencies**
```bash
conda install -c conda-forge jupyterlab numpy scipy scikit-learn matplotlib pip
pip install tensorflow==2.15  # or tensorflow==2.17 if using that version
```

For **TensorFlow 2.17**, also install:
```bash
pip install keras-nightly
```

---

## **5. Switching Between Environments**
To switch environments, deactivate the current one and activate another:
```bash
conda deactivate
conda activate myenv_tf215  # or myenv_tf217
```

---

## **6. Setting Up in PyCharm**
If using PyCharm, you can [add the Conda environment as an interpreter](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html#conda-requirements).

For VS Code, you can configure the interpreter [this way](https://docs.anaconda.com/working-with-conda/ide-tutorials/vscode/).

---

## **7. Uninstalling an Environment**
To remove an environment (after deactivating it):
```bash
conda remove -n myenv_tf215 --all
```

---
