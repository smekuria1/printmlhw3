import argparse
import math
import numpy as np
from sklearn import tree, ensemble
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    load_iris,
    load_breast_cancer,
    load_wine,
)

# Import classes from MLclasses
from MLclasses import DecisionTree, TreeNode, RandomForest

# Dictionary of available datasets
DATASETS = {
    "iris": load_iris,
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
}


def decision_tree_model(
    X: np.ndarray, y: np.ndarray, hyperParams: dict, verbose: bool = True
):
    """
    Train and evaluate the Decision Tree model.
    """

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=25
    )

    # Initialize and train custom decision tree
    custom_tree = DecisionTree(False, hyperParams, TreeNode(0))
    custom_tree.train(hyperParams, X_train, y_train)

    # Initialize and train sklearn decision tree for comparison
    sklearn_tree = tree.DecisionTreeClassifier(
        criterion="entropy" if hyperParams["error_function"] == "entropy" else "gini"
    )
    sklearn_tree.fit(X_train, y_train)

    # Get predictions
    custom_preds_train = custom_tree.predict(X_train)
    custom_preds_test = custom_tree.predict(X_test)
    sklearn_preds_train = sklearn_tree.predict(X_train)
    sklearn_preds_test = sklearn_tree.predict(X_test)

    # Calculate accuracies
    custom_train_acc = accuracy_score(y_train, custom_preds_train)
    custom_test_acc = accuracy_score(y_test, custom_preds_test)
    sklearn_train_acc = accuracy_score(y_train, sklearn_preds_train)
    sklearn_test_acc = accuracy_score(y_test, sklearn_preds_test)

    if verbose:
        print("\nDecision Tree Results:")
        print("-" * 50)
        print("Custom Implementation:")
        print(f"Training Accuracy: {custom_train_acc:.2%}")
        print(f"Testing Accuracy: {custom_test_acc:.2%}")
        print("\nScikit-learn Implementation:")
        print(f"Training Accuracy: {sklearn_train_acc:.2%}")
        print(f"Testing Accuracy: {sklearn_test_acc:.2%}")

    return {
        "custom_train_acc": custom_train_acc,
        "custom_test_acc": custom_test_acc,
        "sklearn_train_acc": sklearn_train_acc,
        "sklearn_test_acc": sklearn_test_acc,
    }


def random_forest_model(
    X: np.ndarray, y: np.ndarray, hyperParams: dict, verbose: bool = True
):
    """
    Train and evaluate the Random Forest model.
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, shuffle=True
    )

    # Initialize and train random forest
    random_forest = RandomForest(False, hyperParams)
    random_forest.train(hyperParams, X_train, y_train)

    # Initialize and train sklearn ensemble RandomForestClassifier for comparison
    sklearn_forest = ensemble.RandomForestClassifier(
        criterion="entropy" if hyperParams["error_function"] == "entropy" else "gini",
        n_estimators=hyperParams["num_tree"],
    )
    sklearn_forest.fit(X_train, y_train)
    # Get predictions
    custom_train_predictions = random_forest.predict(X_train)
    custom_test_predictions = random_forest.predict(X_test)
    sklearn_preds_train = sklearn_forest.predict(X_train)
    sklearn_preds_test = sklearn_forest.predict(X_test)

    # Calculate accuracies
    custom_train_acc = accuracy_score(y_train, custom_train_predictions)
    custom_test_acc = accuracy_score(y_test, custom_test_predictions)
    sklearn_train_acc = accuracy_score(y_train, sklearn_preds_train)
    sklearn_test_acc = accuracy_score(y_test, sklearn_preds_test)

    if verbose:
        print("\nRandom Forest Results:")
        print("-" * 50)
        print(f"Training Accuracy: {custom_train_acc:.2%}")
        print(f"Testing Accuracy: {custom_test_acc:.2%}")
        print("\nScikit-learn Implementation:")
        print("-" * 50)
        print(f"Training Accuracy: {sklearn_train_acc:.2%}")
        print(f"Testing Accuracy: {sklearn_test_acc:.2%}")

    return {
        "custom_train_acc": custom_train_acc,
        "custom_test_acc": custom_test_acc,
        "sklearn_train_acc": sklearn_train_acc,
        "sklearn_test_acc": sklearn_test_acc,
    }


def main():
    """Main function to run the model training and evaluation."""
    parser = argparse.ArgumentParser(description="Train DecisionTree or RandomForest")

    # Model selection
    parser.add_argument(
        "--model",
        choices=["decision_tree", "random_forest"],
        required=True,
        help="Choose the model to train",
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="iris",
        help="Choose the dataset to use",
    )

    # Error criterion
    parser.add_argument(
        "--criterion",
        choices=["entropy", "gini"],
        default="entropy",
        help="Choose the splitting criterion",
    )

    # Number of trees for random forest
    parser.add_argument(
        "--n_trees", type=int, default=10, help="Number of trees for Random Forest"
    )

    # Sampling size for random forest
    parser.add_argument(
        "--sample_size",
        type=float,
        default=0.8,
        help="Proportion of data to sample for each tree (Random Forest)",
    )

    args = parser.parse_args()

    # Load dataset
    dataset_loader = DATASETS[args.dataset]

    dataset = dataset_loader()
    X, y = dataset.data, dataset.target

    if args.dataset == "iris":
        print("Binning IRis")
        for i in range(len(X)):
            X[i] = [round(x) for x in X[i]]
    else:
        print(
            f"Using {args.dataset} not handling binning for this dataset at this moment"
        )
    # Define hyperparameters
    hyperParams = {
        "error_function": args.criterion,
        "num_tree": args.n_trees,
        "sample_size": args.sample_size,
    }

    print(f"\nTraining {args.model.replace('_', ' ').title()} model...")
    print(
        f"Dataset: {args.dataset.replace('_', ' ').title()} (samples: {len(X)}, features: {X.shape[1]})"
    )
    print(f"Hyperparameters: {hyperParams}")

    # Train selected model
    if args.model == "decision_tree":
        decision_tree_model(X, y, hyperParams)
    else:
        random_forest_model(X, y, hyperParams)


if __name__ == "__main__":
    main()
