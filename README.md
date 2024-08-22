# SciBERT-for-Scientific-Paper-Classification-Kaggle-
https://www.kaggle.com/competitions/math80600aw21/overview
Results : 84% - Ranking 8/64


# SciBERT for Scientific Paper Classification

This project implements a SciBERT-based model for classifying scientific papers into 20 categories. It uses the HuggingFace Transformers library and PyTorch for fine-tuning the SciBERT model on a custom dataset.

## Project Overview

The main components of this project are:

1. Data preprocessing
2. Model architecture (SciBERT with additional layers)
3. Training pipeline (including pre-training and fine-tuning)
4. Evaluation and prediction

The model uses SciBERT (uncased) as the base and adds custom layers on top for classification. The training process includes a pre-training phase where only the custom layers are trained, followed by a fine-tuning phase where the entire model is trained.

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- TorchText (version 0.6.0)
- pandas
- scikit-learn
- matplotlib
- seaborn

You can install the required packages using:

```
pip install transformers torchtext==0.6.0 torch pandas scikit-learn matplotlib seaborn
```

## How to Run

1. Prepare your data:
   - Ensure you have `df_train_merged.csv`, `test.csv`, `nodeid2paperid.csv`, and `text.csv` in the `datasets` folder.

2. Set up the directory structure:
   ```
   project_root/
   ├── datasets/
   ├── output/
   └── main.py
   ```

3. Run the main script:
   ```
   python main.py
   ```

This script will:
- Preprocess the data
- Initialize and pre-train the model
- Fine-tune the model
- Evaluate the model on the test set
- Generate predictions for the Kaggle test set

## Model Architecture

The model consists of:
- SciBERT base model
- Dropout layer
- Linear layer (768 -> 96)
- Layer Normalization
- Tanh activation
- Another Dropout layer
- Final Linear layer (96 -> 20)

## Training Process

1. Pre-training: Only the custom layers are trained while SciBERT parameters are frozen.
2. Fine-tuning: The entire model is trained end-to-end.

Both stages use AdamW optimizer with a linear learning rate schedule and warmup.

## Evaluation

The model is evaluated using:
- Classification report (precision, recall, F1-score)
- Confusion matrix

## Kaggle Submission

The script generates predictions for the Kaggle test set and saves them in a CSV file for submission.

## Notes

- The code uses CUDA if available, otherwise falls back to CPU.
- Training progress and loss curves are plotted for visualization.
- The best model is saved during training and can be loaded for inference or further fine-tuning.

Feel free to modify the hyperparameters, model architecture, or training process to suit your needs.
