import numpy as np


class DifferentiableDecisionTree:
    def __init__(self, state_size, action_size, max_depth=3):
        """
        Initialize a differentiable decision tree.
        
        Parameters:
        - state_size: Dimension of the input state.
        - action_size: Dimension of the output action.
        - max_depth: Maximum depth of the tree.
        """
        # Augmented state size to include bias
        self.state_size = state_size + 1
        self.action_size = action_size
        self.max_depth = max_depth
        self.tree_structure = self._initialize_tree()

    def _initialize_tree(self):
        """
        Initialize the weights and biases for the decision nodes and the action values for the leaf nodes.
        
        Returns:
        - tree_structure: Dictionary holding the parameters for each node and leaf.
        """
        tree_structure = {}
        # Initialize nodes (weights and biases) and leaves (action values)
        for depth in range(self.max_depth):
            num_nodes = 2 ** depth
            tree_structure[depth] = {}
            for node in range(num_nodes):
                # Initialize weights and bias for this node
                tree_structure[depth][node] = {
                    'w': np.random.randn(self.state_size),  # Weights for decision
                    'left': None,  # Pointer to left child node
                    'right': None  # Pointer to right child node
                }

        # Initialize leaf nodes (action allocations)
        num_leaves = 2 ** self.max_depth
        tree_structure[self.max_depth] = {}
        for leaf in range(num_leaves):
            tree_structure[self.max_depth][leaf] = {
                'action_value': np.random.dirichlet(np.ones(self.action_size))  # Allocation vector
            }

        return tree_structure

    def _sigmoid(self, x):
        """Sigmoid function for soft decision splits."""
        return 1 / (1+np.exp(-x))

    def predict(self, state):
        """
        Perform a forward pass through the tree to get the final action.
        
        Parameters:
        - state: Input state vector (not including the bias).
        
        Returns:
        - action: Output action vector.
        """
        # Augment state with bias term
        augmented_state = np.append(state, 1)  # Add bias term to the state vector

        # Start from root with probability 1
        probabilities = np.ones(1)  # Root node starts with probability 1
        node_indices = np.zeros(1, dtype=int)  # Start at the root node

        for depth in range(self.max_depth):
            next_probabilities = []
            next_indices = []

            for prob, node_idx in zip(probabilities, node_indices):
                node = self.tree_structure[depth][node_idx]
                # Calculate the probability of going to the left node using unified weight vector
                p_left = self._sigmoid(np.dot(node['w'], augmented_state))

                # Append probabilities for left and right nodes
                next_probabilities.extend([prob * p_left, prob * (1 - p_left)])
                next_indices.extend([2 * node_idx, 2 * node_idx + 1])  # Child nodes

            probabilities = np.array(next_probabilities)
            node_indices = np.array(next_indices)

        # Calculate the weighted average of the leaf values
        leaf_values = np.array([self.tree_structure[self.max_depth][idx]['action_value'] for idx in node_indices])
        return np.dot(probabilities, leaf_values)
    

    def train(self, state, action_gradient, lr=.01):
        """
        Backward pass through the tree using the action gradient to update parameters.
        
        Parameters:
        - state: Input state.
        - action_gradient: Gradient of the loss with respect to the output action.
        - lr: Learning rate for gradient update.
        """
        augmented_state = np.append(state, 1)

        for depth in range(self.max_depth):
            num_nodes = 2 ** depth
            for node_idx in range(num_nodes):
                node = self.tree_structure[depth][node_idx]
                # Gradient update for weights using a simplified chain rule
                # TODO: Implement gradient-based updates using action_gradient and node probabilities


                # Example update (random update for illustration)
                node['w'] -= lr * np.random.randn(self.state_size)  # Placeholder update

        # Leaf nodes update using action_gradient
        num_leaves = 2 ** self.max_depth
        for leaf_idx in range(num_leaves):
            leaf = self.tree_structure[self.max_depth][leaf_idx]
            leaf['action_value'] -= lr * action_gradient  # Simplified leaf update








class TreeActor:
    def __init__(self, num_trees, state_size, action_size, max_depth=3, lambda_poisson=1):
        """
        Initialize a forest of differentiable decision trees with online bagging.
        
        Parameters:
        - num_trees: Number of trees in the forest.
        - state_size: Dimension of the state space.
        - action_size: Dimension of the action space.
        - max_depth: Maximum depth of each tree.
        - lambda_poisson: Lambda parameter for Poisson distribution (default = 1).
        """
        self.num_trees = num_trees
        self.state_size = state_size
        self.action_size = action_size
        self.max_depth = max_depth
        self.lambda_poisson = lambda_poisson

        # Initialize a forest of differentiable decision trees
        self.trees = [DifferentiableDecisionTree(state_size, action_size, max_depth) for _ in range(num_trees)]

    def predict(self, state):
        """
        Predict the action by averaging the outputs of the decision trees in the forest.
        
        Parameters:
        - state: Input state vector.
        
        Returns:
        - action: Final action vector (averaged over all trees).
        """
        tree_outputs = np.array([tree.predict(state) for tree in self.trees])
        return np.mean(tree_outputs, axis=0)


    def update(self, state, action_gradient, lr=0.01):
        """
        Train each tree using the action gradient with Poisson-based online bagging.
        
        Parameters:
        - state: Input state.
        - action_gradient: Gradient of the loss with respect to the output action.
        - lr: Learning rate for the gradient updates.
        """
        # Use Poisson distribution to determine the number of times each data point is used for each tree
        for tree in self.trees:
            # Draw from Poisson to decide how many times to use this data point for this tree
            k = np.random.poisson(self.lambda_poisson)
            if k > 0:
                for _ in range(k):
                    tree.train(state, action_gradient, lr)


    