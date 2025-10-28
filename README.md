# Xreport
This is the official code repo for Xreport: X-ray foundation model reduces misdiagnosis in emergency department: a randomized controlled trial. 

We propose Xreport, a generalist foundation model trained on 335,311 radiology image-report pairs from 44 distinct body parts covering 2,551 ICD-11 codes to address the challenge of misdiagnosis in emergency departments. We propose injecting medical knowledge into the foundation model through a medical ontology base to enhance its diagnostic capabilities across diverse anatomical regions. We propose that this approach significantly reduces misdiagnosis rates and improves patient outcomes, as demonstrated through a prospective randomized controlled trial in emergency departments.

## Data and Model Availability

Due to hospital privacy regulations and patient data protection requirements, the original training data and pre-trained models are **temporarily unavailable** for public release. We are actively working on data anonymization and de-identification processes to make the data available in the future. Besides, we provide the following demo files to facilitate understanding of the project:

- `expert_system/YHY5_result_label.csv`: Sample medical reports (504 cases)
- `expert_system/data/ZJYY/tag.xlsx`: Medical ontology labels
- `xreport/Comparision_exp_human_algo-YHY-5years.csv`: Clinical evaluation results
- `xreport/ChestX-Det10-Dataset/test.json`: Sample dataset format

# Installation
```bash
pip install -r requirements.txt
```

# Repository Structure
```bash
Xreport/
├── xreport/                          # Main Xreport model code
│   ├── configs/                      # Configuration files
│   ├── dataset/                      # Dataset handling code
│   ├── engine/                       # Training engine
│   ├── factory/                      # Loss and metric factories
│   ├── models/                       # Model architectures
│   ├── optim/                        # Optimizer implementations
│   ├── scheduler/                    # Learning rate schedulers
│   ├── ChestX-Det10-Dataset/        # Dataset files
│   ├── main.py                       # Main training script
│   ├── test_chestxray14.py          # ChestX-ray14 test script
│   ├── test_chexpert.py             # CheXpert test script
│   ├── test_padchest.py             # PadChest test script
│   └── plot_visualize_512.py        # Visualization utilities
├── expert_system/                    # Expert system code
│   ├── data/                        # Expert system data
│   ├── main.py                      # Expert system main script
│   └── utils.py                     # Expert system utilities
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

The `xreport/` directory contains the main Xreport model implementation, while `expert_system/` contains the expert system code for clinical decision support.

# Expert System

A medical report classification system that automatically extracts and categorizes diagnostic information from radiology reports.

## What it does
- **Input**: Raw medical reports (CSV format)
- **Process**: Extracts diseases, anatomical sites, and clinical findings
- **Output**: Structured data with standardized medical labels

## Quick Start
```bash
# Process medical reports
python expert_system/run_expert_system.py
```

## Demo Example
Process `YHY5_result_label.csv` (504 radiology reports):
- **Before**: Unstructured Chinese medical text
- **After**: Structured data with 30+ disease categories (fracture, pneumonia, etc.)

## Input Format
CSV file with columns:
- `REPORTSCONCLUSION`: Diagnosis text
- `REPORTSEVIDENCES`: Imaging findings  
- `STUDIESEXAMINEALIAS`: Body part examined

## Output
- Disease classification (fracture, pneumonia, etc.)
- Anatomical site matching
- Standardized medical labels

# Xreport Foundation Model

A multi-modal foundation model for X-ray image analysis and report generation, trained on 335,311 radiology image-report pairs.

## Architecture
- **Vision Encoder**: Vision Transformer (ViT) / ResNet
- **Text Encoder**: Clinical BERT for medical text
- **Multi-modal Learning**: CLIP-style contrastive learning
- **Fine-grained Query**: Transformer decoder for anatomical site detection

## Quick Start

### Training
```bash
# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 xreport/main.py --config configs/Res_train.yaml
```

### Testing
```bash
# Test on ChestX-ray14
python xreport/test_chestxray14.py --config configs/Res_train_test.yaml

# Test on CheXpert
python xreport/test_chexpert.py --config configs/Res_train_test.yaml

# Test on PadChest
python xreport/test_padchest.py --config configs/Res_train_test.yaml
```

## Supported Datasets
- **MIMIC-CXR**
- **ChestX-ray14** 
- **CheXpert**
- **PadChest**

## Model Performance
Results from clinical evaluation (`Comparision_exp_human_algo-YHY-5years.csv`):
- **Fracture Detection**: 9.8% sensitivity improvement
- **Pneumonia Detection**: 6.0% sensitivity improvement  
- **Intestinal Obstruction**: 5.4% sensitivity improvement
- **30-day Mortality**: Significantly reduced in all test groups