import argparse
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import classes from MLclasses
from MLclasses import DecisionTree, TreeNode, RandomForest


def decision_tree_model(X: np.ndarray, y: np.ndarray, hyperParams: dict):
    """
    Train and evaluate the Decision Tree model.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        hyperParams (dict): Hyperparameters for the model
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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

    # Print results
    print("\nDecision Tree Results:")
    print("-" * 50)
    print("Custom Implementation:")
    print(f"Training Accuracy: {accuracy_score(y_train, custom_preds_train):.2%}")
    print(f"Testing Accuracy: {accuracy_score(y_test, custom_preds_test):.2%}")
    print("\nScikit-learn Implementation:")
    print(f"Training Accuracy: {accuracy_score(y_train, sklearn_preds_train):.2%}")
    print(f"Testing Accuracy: {accuracy_score(y_test, sklearn_preds_test):.2%}")


def random_forest_model(X: np.ndarray, y: np.ndarray, hyperParams: dict):
    """
    Train and evaluate the Random Forest model.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        hyperParams (dict): Hyperparameters for the model
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train random forest
    random_forest = RandomForest(False, hyperParams)
    random_forest.train(hyperParams, X_train, y_train)

    # Get predictions
    train_predictions = random_forest.predict(X_train)
    test_predictions = random_forest.predict(X_test)

    # Print results
    print("\nRandom Forest Results:")
    print("-" * 50)
    print(f"Training Accuracy: {accuracy_score(y_train, train_predictions):.2%}")
    print(f"Testing Accuracy: {accuracy_score(y_test, test_predictions):.2%}")


def main():
    """Main function to run the model training and evaluation."""
    parser = argparse.ArgumentParser(description="Train DecisionTree or RandomForest")
    parser.add_argument(
        "--model",
        choices=["decision_tree", "random_forest"],
        required=True,
        help="Choose the model to train",
    )
    parser.add_argument(
        "--criterion",
        choices=["entropy", "gini"],
        default="entropy",
        help="Choose the splitting criterion",
    )
    parser.add_argument(
        "--n_trees", type=int, default=10, help="Number of trees for Random Forest"
    )

    args = parser.parse_args()

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Define hyperparameters
    hyperParams = {"error_function": args.criterion, "num_tree": args.n_trees}

    print(f"\nTraining {args.model.replace('_', ' ').title()} model...")
    print(f"Dataset: Iris (samples: {len(X)}, features: {X.shape[1]})")
    print(f"Hyperparameters: {hyperParams}")

    # Train selected model
    if args.model == "decision_tree":
        decision_tree_model(X, y, hyperParams)
    else:
        random_forest_model(X, y, hyperParams)


if __name__ == "__main__":
    main()

