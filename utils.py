class TreeNode:
    def __init__(self, val: int) -> None:
        """
        Initializes a TreeNode for a decision tree.

        """
        self.childern = {}
        self.val = val
        self.majority_label = ...

    def is_decision_node(self) -> bool:
        """
        Checks if current node is leaf/decision node

        returns:
            bool
        """

        return len(self.childern) == 0
