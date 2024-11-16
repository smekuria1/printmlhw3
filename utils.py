class TreeNode:
    def __init__(self, val: int) -> None:
        """
        Initializes a TreeNode for a decision tree.

        """
        self.childern = {}
        self.val = val

    def is_decision_node(self) -> bool:
        """
        Checks if current node is leaf/decision node

        returns:
            bool
        """

        return len(self.childern) == 0
