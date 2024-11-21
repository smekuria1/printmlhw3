import numpy as np
import math
from utils import TreeNode


DEBUG_FLAG_PRINT = False
DEBUG_FLAG_TEST = False
APPROXIMATION_FLAG = True


class MachineLearningTemplate:
    """
    A template for machine learning models with methods for training, prediction,
    and accessing model parameters and hyperparameters.
    """

    def __init__(self, paramsAssigned: bool, hyperParams, learned):
        """
        Initializes with hyperparameters, learned parameters, and an assignment flag.

        Args:
            paramsAssigned (bool): Flag indicating if parameters are initialized.
            hyperParams: Model hyperparameters.
            learned: Model learned parameters.

        Functions:
          train --> Implement in subclasses
          predict --> Implement in subclasses
          getParamsAssigned --> bool
          getHyperParameters --> dict of hyperparameters
        """
        self.paramsAssigned = paramsAssigned
        self.hyperParams = hyperParams
        self.learned = learned

    def train(self, hyperParams: dict, X, y):
        pass

    def predict(self, X) -> list:
        return []

    def getParamsAssigned(self) -> bool:
        return self.paramsAssigned

    def getHyperParameters(self) -> dict:
        return self.hyperParams


class DecisionTree(MachineLearningTemplate):
    def __init__(self, paramsAssigned: bool, hyperParams, learned: TreeNode):
        """ """
        super().__init__(paramsAssigned, hyperParams, learned)
        self.node = learned

    def _entropy(self, y):
        """
        Computes the entropy of a dataset using class label proportions.

        Entropy is a measure of the impurity or uncertainty in a dataset. It is calculated
        using the formula: h = - Σ(p * log2(p)), where p is the proportion of each class
        label in the dataset.

        Args:
            y (list or numpy.ndarray)

        Returns:
            float: The computed entropy of the dataset.
        """

        label_counter = {}

        for label in y:
            if label in label_counter:
                label_counter[label] += 1
            else:
                label_counter[label] = 1

        h = 0
        total_entries = len(y)

        for count in label_counter.values():
            p = count / total_entries

            if p > 0:  # Ensure no division by zero issues
                h += -p * math.log2(p)

        return h

    def test__entropy(self):
        """
        Test function for _entropy to validate its correctness.

        Tests the function using predefined datasets and compares the output
        against the expected entropy values using assert statements.
        """
        # Test 1: Example dataset with known result
        y = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
        expected_entropy = 0.97095059445466  # Known expected result
        computed_entropy = self._entropy(y)
        assert np.isclose(
            computed_entropy, expected_entropy, atol=1e-6
        ), f"Test 1 failed! Expected {expected_entropy}, but got {computed_entropy}"

        # Additional Tests
        # Test 2: Completely pure dataset (entropy should be 0)
        y_pure = [1, 1, 1, 1, 1]
        expected_entropy_pure = 0.0
        computed_entropy_pure = self._entropy(y_pure)
        assert np.isclose(
            computed_entropy_pure, expected_entropy_pure, atol=1e-6
        ), f"Test 2 failed! Expected {expected_entropy_pure}, but got {computed_entropy_pure}"

        # Test 3: Completely balanced dataset (entropy should be 1 for binary classification)
        y_balanced = [0, 1, 0, 1]
        expected_entropy_balanced = 1.0
        computed_entropy_balanced = self._entropy(y_balanced)
        assert np.isclose(
            computed_entropy_balanced, expected_entropy_balanced, atol=1e-6
        ), f"Test 3 failed! Expected {expected_entropy_balanced}, but got {computed_entropy_balanced}"

        print("All Entropy tests passed!")

    def _gini_impurity(self, y):
        """
        Computes the Gini impurity of a dataset using class label proportions.

        Gini impurity is a measure of how often a randomly chosen element would be incorrectly
        classified. It is calculated using the formula: IG(S) = 1 - Σ(p^2), where p is the
        proportion of each class label in the dataset.

        Args:
            y (list or numpy.ndarray): A list or 1D numpy array containing the class labels of the dataset.

        Returns:
        float: The computed Gini impurity of the dataset.
        """

        label_counts = {}

        for label in y:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        g = 0
        total_entries = len(y)

        for count in label_counts.values():
            p = count / total_entries

            g += p * p

        gini_impurity = 1 - g

        return gini_impurity

    def test_gini_impurity(self):
        """
        Test function for compute_gini_impurity to validate its correctness.

        Tests the function using predefined datasets and compares the output
        against the expected Gini impurity values using assert statements.
        """
        # Test 1: Example dataset
        y = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
        expected_gini = 0.48  # Known expected result for the example dataset
        computed_gini = self._gini_impurity(y)
        assert np.isclose(
            computed_gini, expected_gini, atol=1e-6
        ), f"Test 1 failed! Expected {expected_gini}, but got {computed_gini}"

        # Additional Tests
        # Test 2: Completely pure dataset (Gini impurity should be 0)
        y_pure = [1, 1, 1, 1, 1]
        expected_gini_pure = 0.0
        computed_gini_pure = self._gini_impurity(y_pure)
        assert np.isclose(
            computed_gini_pure, expected_gini_pure, atol=1e-6
        ), f"Test 2 failed! Expected {expected_gini_pure}, but got {computed_gini_pure}"

        # Test 3: Completely balanced dataset (Gini impurity should be 0.5 for binary classification)
        y_balanced = [0, 1, 0, 1]
        expected_gini_balanced = 0.5
        computed_gini_balanced = self._gini_impurity(y_balanced)
        assert np.isclose(
            computed_gini_balanced, expected_gini_balanced, atol=1e-6
        ), f"Test 3 failed! Expected {expected_gini_balanced}, but got {computed_gini_balanced}"

        print("All Gini impurity tests passed!")

    def train(self, hyperParams: dict, X, y) -> "DecisionTree":
        """Train DecisionTree on input Data"""

        queried = []
        self.hyperParams = hyperParams
        self.node = self.train_helper(hyperParams, X, y, queried).node
        self.paramsAssigned = True
        return self

    def predict(self, X) -> list:
        """Predict DecisionTree on input Data
        Parameters:
        ----------
        X : list or array-like
            Input data to predict, where each element is a feature vector.

        Returns:
        -------
        list
            A list of predicted labels for each input feature vector in X.

        NOTE:
            There is a small approximation function that finds the closest value to an unseen from the
            childern of the current node.
            I was working with Continous datasets unknowingly if were are getting properly discrete
            feature values it is less likely we run into the code branch I hope to fix this someday
        Raises:
        ------
        AssertionError
            If the model has not been trained yet (`paramsAssigned` is False).

        ValueError
            If an unknown feature value is encountered during prediction and cannot be handled.

        """

        if not self.paramsAssigned:
            raise AssertionError(
                "Please Call train before calling the predict  function"
            )
        z = []
        for row in X:
            curr_node = self.node
            while not curr_node.is_decision_node():
                feature_value = row[curr_node.val]
                if feature_value not in curr_node.childern:
                    approximation = None
                    if APPROXIMATION_FLAG:
                        key_list = list(curr_node.childern.keys())
                        approximation = self.find_closest_feature(
                            key_list, feature_value
                        )
                    if len(curr_node.childern) > 0:
                        print(
                            f"Unseen feature value {feature_value}, Approximation {approximation}"
                        )
                        feature_value = (
                            list(curr_node.childern.keys())[0]
                            if not APPROXIMATION_FLAG
                            else approximation
                        )

                    else:
                        raise ValueError(
                            f"Unknown feature value {feature_value} encountered during prediction."
                        )
                curr_node = curr_node.childern[feature_value].node
            # Append the label of the leaf node
            z.append(int(curr_node.val))
        return z

    def find_closest_feature(self, lst, target):
        """Finds the closest number to a target in a list."""

        closest = lst[0]
        min_diff = abs(target - closest)

        for num in lst:
            diff = abs(target - num)
            if diff < min_diff:
                min_diff = diff
                closest = num

        return closest

    def train_helper(self, hyperParams: dict, X, y, queried: list):
        """
        Recursive helper function to train a Decision Tree.

        Parameters:
        ----------
        hyperParams : dict
            A dictionary of hyperparameters, including the error function to use
            (either 'entropy' or 'gini').

        X : list or array-like
            The feature matrix, where each element is a feature vector.

        y : list
            The target labels corresponding to each row in `X`.

        queried : list
            A list of indices representing the features that have been queried
            in previous splits.

        Returns:
        -------
        DecisionTree
            A trained DecisionTree object.

        Raises:
        ------
        ValueError
            If an unsupported error function is specified.
        """

        # Base case: if all features values have been queried, create a leaf node with the majority label
        if len(queried) == len(y):
            labels = {}
            for label in y:
                if label in labels:
                    labels[label] += 1
                else:
                    labels[label] = 1

            majority = max(labels)
            if DEBUG_FLAG_PRINT:
                print("Base Case reached")
            node = TreeNode(majority)
            return DecisionTree(True, hyperParams, node)

        # Calculate the base error using the specified error function
        base_error = None
        error_function = hyperParams["error_function"]
        if error_function == "entropy":
            base_error = self._entropy(y)
        elif error_function == "gini":
            base_error = self._gini_impurity(y)
        else:
            raise ValueError("Incorrect Error Function provided")

        feature_value_indices = {}
        feature_error_change = {}

        # Iterate over all features to find the best split
        for c in range(len(X[0])):
            if c in queried:
                continue
            feature_value_indices[c] = {}

            # Group row indices by feature values for feature `c`
            for r in range(len(X)):
                feature_value = X[r][c]

                if feature_value not in feature_value_indices[c]:
                    feature_value_indices[c][feature_value] = []

                # Append the row index to the corresponding feature value list
                feature_value_indices[c][feature_value].append(r)

            accumulated_error_change = 0.0
            # Calculate error reduction for each unique feature value
            for v in feature_value_indices[c]:
                # Create label vector y_f,v for current spliddt
                y_f_v = []
                for r_prime in feature_value_indices[c][v]:
                    y_f_v.append(y[r_prime])

                # Compute the error for y_f,v
                if error_function == "entropy":
                    subset_error = self._entropy(y_f_v)
                else:
                    subset_error = self._gini_impurity(y_f_v)

                # Calculate the error difference from the baseline
                error_difference = base_error - subset_error
                accumulated_error_change += error_difference

            # Calculate the average change in error
            num_splits = len(feature_value_indices[c])
            if num_splits > 0:
                feature_error_change[c] = accumulated_error_change / num_splits

        best_error_change = (
            float("-inf")
            if self.getHyperParameters()["error_function"] == "entropy"
            else float("inf")
        )
        best_feature_index = None
        for feature_index, error_change in feature_error_change.items():
            if self.getHyperParameters()["error_function"] == "entropy":
                # We want to maximize the information gain (entropy)
                if error_change > best_error_change:
                    best_error_change = error_change
                    best_feature_index = feature_index
                    if DEBUG_FLAG_PRINT:
                        print(
                            f"Found Better Error Entropy {error_change} FeatureIndex {best_feature_index}"
                        )
            elif self.getHyperParameters()["error_function"] == "gini":
                # We want to minimize the Gini impurity
                if error_change < best_error_change:
                    if DEBUG_FLAG_PRINT:
                        print("Found Better Error GINI ", error_change)
                    best_error_change = error_change
                    best_feature_index = feature_index
        # Create a query node based on the best feature found
        if DEBUG_FLAG_PRINT:
            print("Best Feature index=", best_feature_index)
        node = TreeNode(best_feature_index)

        split_data = {}
        for row_index, row in enumerate(X):
            feature_value = row[best_feature_index]
            if feature_value not in split_data.items():
                split_data[feature_value] = {"X": [], "y": []}
            split_data[feature_value]["X"].append(row)
            split_data[feature_value]["y"].append(y[row_index])

        for value, data in split_data.items():
            # Recursively train on each subset of the data
            child_node = self.train_helper(
                self.getHyperParameters(),
                X=np.array(data["X"]),
                y=data["y"],
                queried=queried + [best_feature_index],
            )
            node.childern[value] = child_node

            # Return the pointer to the current node
        return DecisionTree(True, hyperParams, node)


class RandomForest(MachineLearningTemplate):
    def __init__(
        self,
        paramsAssigned: bool,
        hyperParams,
        learned=None,
    ):
        super().__init__(paramsAssigned, hyperParams, learned)
        self.trees = []

    def train(self, hyperParams: dict, X, y):
        """
        Train the Random Forest using bootstrapped samples and Decision Trees.

        Parameters:
        ----------
        hyperParams : dict
            Hyperparameters for the Random Forest. Includes 'num_tree'
            (number of trees to train) and 'sample_size' (fraction of
            data to sample for each tree).

        X : array-like
            Feature matrix for training.

        y : array-like
            Target labels for training.

        """
        num_tree = self.getHyperParameters().get("num_tree")
        if num_tree is None:
            num_tree = 10

        n_samples = int(X.shape[0] * self.getHyperParameters()["sample_size"])

        for _ in range(num_tree):
            indices = np.random.choice(len(X), size=n_samples, replace=False)

            X_prime = X[indices]
            y_prime = y[indices]

            decisionTree = DecisionTree(False, hyperParams, TreeNode(0))
            model = decisionTree.train(hyperParams, X_prime, y_prime)
            self.trees.append(model)

    def predict(self, X) -> list:
        """
        Predict the labels for the input data using majority voting from all trees.

        Parameters:
        ----------
        X : array-like
            Feature matrix for which predictions are made.

        Returns:
        -------
        list
            A list of predicted labels for the input data.

        """
        M = np.zeros((X.shape[0], self.getHyperParameters()["num_tree"]))

        z = np.zeros((X.shape[0], 1))
        for i in range(len(self.trees)):
            pred = self.trees[i].predict(X)
            pred = np.array(pred)
            M[:, i] = pred

        # Iterate over each row in M
        for row_idx in range(M.shape[0]):
            # Initialize a dictionary to count occurrences of each label
            label_counts = {}

            # Iterate over columns in the current row
            for col_idx in range(M.shape[1]):
                label = M[row_idx, col_idx]
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1

            # Find the label with the greatest count
            max_label = None
            max_count = 0
            for label, count in label_counts.items():
                if count > max_count:
                    max_label = label
                    max_count = count

            # Store the label with the greatest count into z at the current row index
            z[row_idx, 0] = max_label

        # Return z as a list of predictions
        return z.tolist()
