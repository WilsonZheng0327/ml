import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        """
        _summary_

        Args:
            feature_index (_type_, optional): _description_. Defaults to None.
            threshold (_type_, optional): _description_. Defaults to None.
            left (_type_, optional): _description_. Defaults to None.
            right (_type_, optional): _description_. Defaults to None.
            info_gain (_type_, optional): _description_. Defaults to None.
            value (_type_, optional): _description_. Defaults to None.
        """

        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        self.value = value  # leaf node only

class DecisionTreeClassifier():
    def __init__(self, max_depth=2, min_samples_split=2, min_samples_leaf=1):
        """
        Decision Tree for classification

        Args:
            max_depth (int, optional): Max depth of tree. Defaults to 2.
            min_samples_split (int, optional): Min number of samples required to be able to split. Defaults to 2.
            min_samples_leaf (int, optional): Min number of samples required to be a leaf node. Defaults to 1.
        """

        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
    def fit(self, X, y):
        """Fit the decision tree"""

        dataset = np.concatenate((X, y), axis=1)
        self.root = self._build_tree(dataset)

    def predict(self, X):
        """Predict based on inputs"""

        predictions = [self._predict(x, self.root) for x in X]
        return predictions

    def _build_tree(self, dataset, cur_depth=0):
        """Recursively built decision tree"""

        X, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        # check stopping conditions
        if num_samples >= self.min_samples_split and cur_depth <= self.max_depth:
            # test all features and get best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # if splitting is beneficial
            if best_split["info_gain"] > 0:
                left_subtree = self._build_tree(best_split["dataset_left"], cur_depth+1)
                right_subtree = self._build_tree(best_split["dataset_right"], cur_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # code at this point means this node is leaf node
        return Node(value=self._calc_leaf_value(y))

    def _get_best_split(self, dataset, num_samples, num_features):
        """Finds best split"""

        best_split = {}
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            # column of target feature
            feature_values = dataset[:, feature_index]
            # unique values of that column to choose best threshold from
            possible_thresholds = np.unique(feature_values)

            for threshold in possible_thresholds:
                # split dataset based on threshold
                dataset_left, dataset_right = self._split_dataset(dataset, feature_index, threshold)

                if len(dataset_left) >= self.min_samples_leaf and len(dataset_right) >= self.min_samples_leaf:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    info_gain = self._calc_information_gain(y, left_y, right_y)

                    # keep track of max info gain
                    if info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = info_gain
                        max_info_gain = info_gain

    def _split_dataset(self, dataset, feature_index, threshold):
        """Split dataset based on given threshold for some feature"""

        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])

        return dataset_left, dataset_right
    
    def _calc_information_gain(self, y, left_y, right_y):
        """Calculate the information gain of a certain split"""

        left_weight = len(left_y) / len(y)
        right_weight = len(right_y) / len(y)

        # info gain is the difference between Gini Impurity before split 
        #       and weighted average of Gini Impurity after split
        gain = self._gini(y) - (left_weight * self._gini(left_y) + right_weight * self._gini(right_y))

        return gain

    def _gini(self, y):
        """Calculate the Gini Impurity for some samples"""

        labels = np.unique(y)
        impurity = 1
        for label in labels:
            prob_of_label = len(y[y == label]) / float(len(y))
            impurity -= prob_of_label ** 2
        return impurity
    
    def _calc_leaf_value(self, y):
        """Calculate the value of the leaf node"""

        y = list(y)
        return max(y, key=y.count)

    def _predict(self, X, tree):
        """Predict a single sample input"""

        # base case: leaf node, return value of leaf node
        if tree.value != None:
            return tree.value
        
        # get target feature of current node
        feature_val = X[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._predict(X, tree.left)
        else:
            return self._predict(X, tree.right)
        
    def print_tree(self, tree=None, indent=" "):
        """Pretty print the tree"""

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)