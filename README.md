# Causal-LiTS: Counterfactual CLIP-based Causal Intervention for Instance-aware Liver Tumor Segmentation

## 📖 Project Introduction
Causal-LiTS is a novel framework designed for instance-aware liver tumor segmentation, which integrates counterfactual causal intervention based on CLIP to enhance the accuracy and instance awareness of liver tumor segmentation.

## 🚀 Quick Start

### 1. Code Execution
Our **full and complete source code** has been uploaded to this repository. To start training the Causal-LiTS model, execute the following command directly in the project root directory:
```bash
python train.py
```
## 📖 Dataset Preparation
The M2-LiTS dataset required for model training and evaluation is stored on Google Drive. Please download the dataset first before running the training script:

M2-LiTS Dataset Download Link: https://drive.google.com/drive/folders/1WvZdjxS12MJ3mD27SqYwqKdvb6sLtnEq?usp=sharing

Dataset Placement
After downloading, place the dataset in the ./data directory of the project (default path). If you need to modify the dataset path, please adjust the relevant configuration in the config.py file.

⚠️ Notes
Ensure all required dependencies are installed before running the code (see requirements.txt for details).
If you encounter access restrictions or download failures with the Google Drive link, please submit an issue in this repository, and we will handle it as soon as possible.
