# DNA-Binding Cys2His2 Zinc-Finger Prediction

This project predicts the DNA-binding ability of Cys2His2-type zinc fingers for a given set of proteins from the C-RC dataset.

## Description

Each zinc finger is represented by a **feature matrix** extracted using the **ESM-DBP model**, as described in the paper:

> *Improving prediction performance of general protein language model by domain-adaptive pretraining on DNA-binding protein*

The model is trained to classify the DNA-binding potential of each zinc finger based on its feature representation.

## Model Architecture

The classifier uses the following architecture:
- Two **BiLSTM** layers
- Four **fully connected (linear)** layers

## Evaluation Protocol

Performance is evaluated using **10-fold cross-validation**, where each fold holds out a subset of proteins and all their zinc fingers.

## How to Run

1. **Download the feature matrix ZIP** file from:
   [Google Drive Link](https://drive.google.com/drive/folders/1b7LGQpQLPzUrwq5Y2rq76Cxs2WQzvCJl?usp=sharing)

2. **Extract the archive** to: `data/feature_matrices/`

3. **Train the model** by running:
```bash
python main.py
```

## Output
The predictions are saved under the results/ directory:

- `results/zf_pred_df.csv`: contains predictions for each zinc finger in the C-RC dataset, including:
    - pred_label_value: predicted probability from the sigmoid function
    - pred_label: binary classification (1.0 if predicted as binding, 0.0 otherwise)
