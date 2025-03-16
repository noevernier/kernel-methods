# Kernel Methods for Sequence Classification

## Overview
This project implements kernel-based methods for sequence classification, leveraging k-mer features and Support Vector Machines (SVMs). The core idea is to transform sequences into feature representations using k-mer kernels and train an SVM for classification. The implementation includes hyperparameter tuning using Optuna and evaluation with cross-validation.

## Features
- **Custom Kernel Methods:** Implements k-mer and k-mer mismatch kernels.
- **Hyperparameter Tuning:** Uses Optuna for efficient optimization.
- **Cross-Validation:** Ensures robust performance evaluation.
- **Configurable Training:** Settings can be adjusted via `config.json`.
- **Export Predictions:** Saves model predictions for submission.

## Directory Structure
```
kernel-methods/
├── best_params_per_idx.json  # Stores best hyperparameters found during tuning
├── config.json               # Configuration file for training and evaluation
├── grid_search.py            # Script for hyperparameter tuning using Optuna
├── main.py                   # Main script for training, evaluation, and prediction
├── data/                      # Directory for storing datasets
├── export/                    # Directory for saving results
└── src/                       # Source code directory
    ├── kernel.py             # Kernel methods implementation
    ├── model.py              # Kernel SVM implementation
    ├── tools.py              # Utility functions and dataset handling
    ├── dataset.py            # Dataset preprocessing and handling
    └── evaluation.py         # Evaluation metrics and performance analysis
```

## Installation
To run this project, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Hyperparameter Tuning
To perform a grid search for optimal hyperparameters, run:
```bash
python grid_search.py
```
This will optimize the parameters and store the best results in `best_params_per_idx.json`.

### Training and Evaluation
To train the model and evaluate its performance:
```bash
python main.py
```
The script will load the dataset, train the kernel SVM, and print evaluation metrics.

### Configuration
Modify `config.json` to adjust hyperparameters and settings:
```json
{
    "C": [0.879, 1.911, 2],
    "tol": [1e-5, 1e-5, 1e-5],
    "kmin": [6, 4, 5],
    "kmax": [18, 31, 15],
    "submit": true,
    "output": "submit4.csv",
    "cv_folds": 5
}
```
- `C`: Regularization parameter for SVM.
- `tol`: Tolerance for stopping criteria.
- `kmin`, `kmax`: Range of k-mer lengths.
- `submit`: If `true`, saves predictions for submission.
- `output`: Filename for saving predictions.
- `cv_folds`: Number of cross-validation folds.

### Exporting Predictions
If `submit` is set to `true` in `config.json`, predictions are saved in the `export/` directory:
```bash
export/submit4.csv
```

## Contributing
Feel free to submit pull requests or report issues to improve the project.

## License
This project is licensed under the MIT License.

