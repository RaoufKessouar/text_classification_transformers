# BERT Sarcasm Detection Project (News Headlines)

This repository implements a Natural Language Processing (NLP) pipeline for binary text classification (Sarcastic vs. Not Sarcastic). It utilizes **Transfer Learning** with the **BERT** architecture (`bert-base-uncased`) implemented in **PyTorch**.

The project demonstrates an End-to-End workflow: from downloading data via Kaggle API to fine-tuning a Transformer model and evaluating performance on unseen test data.

## Main Content
- `Sarcasm_Detection_BERT.ipynb`: The complete notebook containing data ingestion, preprocessing, model architecture, training loop, and evaluation.
- `requirements.txt`: List of dependencies required to reproduce the environment.
- `results/`: (Optional) Folder to store generated loss/accuracy plots.

## Prerequisites
- Python 3.8+ recommended
- A Kaggle account (for API credentials `kaggle.json`).
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  (Key packages: `torch`, `transformers`, `pandas`, `scikit-learn`, `opendatasets`)

## Quick Start

### Setup the repository
```
bert_sarcasm_project/
├─ text_classification_transformers.ipynb
├─ requirements.txt
└─ README.md
```

### Launch the Notebook
Open `Sarcasm_Detection_BERT.ipynb` in Jupyter or Google Colab.

**Note**: The notebook uses `opendatasets` to download the data. You will be prompted to enter your Kaggle Username and Key during the first cell execution.

The code automatically handles GPU acceleration (cuda check included).

### Expected Results
- The training runs for **25 epochs**.
- **Final Test Accuracy**: ~84.6%
- **Validation Accuracy** peaks around 87%.

## Pipeline Architecture

### Data Ingestion & Cleaning
- Downloads **News Headlines Dataset for Sarcasm Detection** (~26k samples).
- Removes duplicates and NaN values using Pandas.
- **Split**: 70% Train, 15% Validation, 15% Test.

### Preprocessing (Tokenization)
- Uses Hugging Face `AutoTokenizer` (`bert-base-uncased`).
- **Settings**: `max_length=100`, `padding='max_length'`, `truncation=True`.
- Custom PyTorch Dataset class handling tensor conversion on the fly.

### Model Architecture (Custom Head)
- **Backbone**: Pre-trained `bert-base-uncased` (Weights frozen to preserve knowledge).
- **Custom Classifier**:
  - BERT Pooler Output (768 features)
  - Linear Layer (768 $\to$ 384)
  - Dropout (0.25)
  - Linear Layer (384 $\to$ 1)
  - Sigmoid Activation (for binary output)

### Training Strategy
- **Loss Function**: `BCELoss` (Binary Cross Entropy).
- **Optimizer**: Adam (Learning Rate = 1e-4).
- **Batch Size**: 32.
- Includes manual implementation of Training and Validation loops for full control.

## Results and Visualizations
The notebook generates visualization plots for:
- **Training vs Validation Loss**: Allows monitoring of convergence and overfitting (overfitting observed after epoch 10).
- **Training vs Validation Accuracy**: Tracks performance stability.

### Key Metrics
- **Accuracy Score on Testing Data**: 84.6%

The model demonstrates strong generalization capabilities despite the relatively small dataset size for a transformer model.

## Author
[RaoufKessouar](https://github.com/RaoufKessouar)
