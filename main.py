import argparse
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Import classes from MLclasses
from MLclasses import (
    DecisionTree,
    TreeNode,
    DEBUG_FLAG_PRINT,
    DEBUG_FLAG_TEST,
    RandomForest,
)


# Function to train and evaluate the Decision Tree model
def decision_tree_model(X_train, X_test, y_train, y_test, hyperParam):
    testTree = DecisionTree(False, hyperParam, TreeNode(0))

    if DEBUG_FLAG_TEST:
        testTree.test__entropy()
        testTree.test_gini_impurity()

    # Train custom decision tree model
    model = testTree.train(hyperParam, X_train, y_train)

    # Train sklearn decision tree for comparison
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Predictions
    badmodel = model.predict(X_test)
    goodmodel = clf.predict(X_test)

    # Convert predictions to integers
    converted = [int(x) for x in list(goodmodel)]

    # Compare Accuracy
    goodPreds = np.array(converted)
    badPreds = np.array(badmodel)

    print(f"Decision Tree Accuracy: {np.mean(y_test == badPreds) * 100}%")


# Function to train and evaluate the Random Forest model
def random_forest_model(X_train, X_test, y_train, y_test, hyperParam):
    # Initialize random forest with parameters
    forestHyperParams = {"error_function": "entropy", "num_tree": 10}
    trees = []
    randomForest = RandomForest(False, forestHyperParams, TreeNode(0), trees)

    # Train and predict
    randomForest.train(forestHyperParams, X_train, y_train)
    predictions = randomForest.predict(X_test)

    # Calculate accuracy
    preds = np.array(predictions)
    print(f"Random Forest Accuracy: {np.mean(y_test == preds) * 100}%")


def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Train DecisionTree or RandomForest")
    parser.add_argument(
        "--model",
        choices=["decision_tree", "random_forest"],
        required=True,
        help="Choose the model to train",
    )
    args = parser.parse_args()

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=33
    )

    # Define hyperparameters
    hyperParam = {"error_function": "entropy"}

    # Conditional check for which model to use
    if args.model == "decision_tree":
        decision_tree_model(X_train, X_test, y_train, y_test, hyperParam)
    elif args.model == "random_forest":
        random_forest_model(X_train, X_test, y_train, y_test, hyperParam)


if __name__ == "__main__":
    main()
