# Custom Machine Learning Model Implementation

## Project Overview

This project provides custom implementations of Decision Tree and Random Forest machine learning algorithms from scratch, along with a flexible training and evaluation framework.

## Features

- Custom Decision Tree classifier
- Custom Random Forest classifier
- Flexible hyperparameter tuning
- Multiple dataset support
- Comparison with scikit-learn implementations

## Project Structure

```
printmlhw3/
│
├── MLclasses.py         # Core machine learning model implementations
├── main.py              # Model training and evaluation script
├── utils.py             # Utility classes and functions
└── README.md            # Project documentation
```

## Dependencies

- NumPy
- Scikit-learn

## Installation

1. Clone the repository:

```bash
git clone https://github.com/smekuria1/printmlhw3
cd printmlhw3
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install required dependencies:

```bash
pip install numpy scikit-learn
```

## Usage

### Command-Line Options

```bash
python main.py --help
```

#### Available Arguments

- `--model`: Choose model type
  - `decision_tree`
  - `random_forest`
- `--dataset`: Select dataset
  - `iris`
  - `breast_cancer`
  - `wine`
- `--criterion`: Splitting criterion
  - `entropy`
  - `gini`
- `--n_trees`: Number of trees (for Random Forest)
- `--sample_size`: Proportion of data sampled for each tree

### Example Commands

1. Train Decision Tree on Iris dataset:

```bash
python main.py --model decision_tree --dataset iris --criterion entropy
```

2. Train Random Forest on Breast Cancer dataset:

```bash
python main.py --model random_forest --dataset breast_cancer --criterion gini --n_trees 20
```

```

## Model Implementations

### Decision Tree

- Supports entropy and Gini impurity as splitting criteria
- Recursive tree construction
- Custom implementation of impurity calculations

### Random Forest

- Bootstrap sampling
- Majority voting for predictions
- Configurable number of trees
- Supports different datasets and error functions

## Customization

### Hyperparameters

- `error_function`: Choose between 'entropy' and 'gini'
- `num_tree`: Set number of trees for Random Forest
- `sample_size`: Control bootstrap sampling proportion

## Performance Evaluation

The script compares custom implementations with scikit-learn's built-in models, providing:

- Training accuracy
- Testing accuracy
- Side-by-side performance comparison

## Supported Datasets

1. Iris Dataset
   - Multi-class classification
   - 4 features
   - 3 classes

2. Breast Cancer Dataset
   - Binary classification
   - 30 features
   - Medical diagnosis problem

3. Wine Dataset
   - Multi-class classification
   - 13 features
   - 3 wine types


## License

Distributed under the MIT License. See `LICENSE` for more information.
