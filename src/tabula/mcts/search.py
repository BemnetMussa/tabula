import math
import torch
import chess

class Node:
    """
    A single board state in the Monte Carlo Tree.
    """
    def __init__(self, prior_prob: float, parent=None):
        self.prior_prob = prior_prob       # P: Probability predicted by TabulaNet
        self.parent = parent               # Parent Node
        self.children = {}                 # Dictionary mapping chess.Move -> child Node
        self.visit_count = 0               # N: Number of times this node was visited
        self.value_sum = 0.0               # W: Total value accumulated from sub-tree

    @property
    def q_value(self) -> float:
        """Q: The average value (W / N) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, action_probs: dict[chess.Move, float]):
        """
        Populate the node with its legal children.
        action_probs is a dictionary of {legal_move: probability} returned by TabulaNet.
        """
        for move, prob in action_probs.items():
            if move not in self.children:
                self.children[move] = Node(prior_prob=prob, parent=self)

    def is_expanded(self) -> bool:
        """If it has children, it has already been expanded by the Neural Net."""
        return len(self.children) > 0


class MCTS:
    """
    The Monte Carlo Tree Search algorithm for AlphaZero.
    """
    def __init__(self, model, num_simulations=100, c_puct=1.25):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct  # Hyperparameter controlling exploration

    def search(self, board, encoder):
        """
        Run the simulations to build the tree and return the root node.
        We can then pick the best move based on the visit counts of the root's children.
        """
        root = Node(prior_prob=1.0)
        
        # 0. Initial Expansion
        policy_dict, _ = self.predict(board, encoder)
        root.expand(policy_dict)

        for _ in range(self.num_simulations):
            node = root
            search_board = board.copy()
            
            # 1. SELECT
            # Traverse down the tree picking the highest PUCT score until we hit a leaf
            while node.is_expanded():
                move, node = self.select_child(node)
                search_board.push(move)
                
            # 2. EXPAND & EVALUATE
            outcome = search_board.outcome()
            if outcome is not None:
                # Terminal state (Game Over) - no need for the Neural Net
                if outcome.winner is None:
                    value = 0.0  # Draw
                else:
                    value = 1.0 if outcome.winner == search_board.turn else -1.0
            else:
                # Leaf node reached - Ask TabulaNet for P (Policy) and V (Value)
                policy_dict, value = self.predict(search_board, encoder)
                node.expand(policy_dict)
                
            # 3. BACKPROPAGATE
            # Pass the value back up the tree
            self.backpropagate(node, value)
            
        return root

    def select_child(self, node: Node):
        """Pick the child with the highest PUCT score."""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            score = self.puct_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def puct_score(self, parent: Node, child: Node) -> float:
        """
        AlphaZero PUCT Formula: 
        Exploitation (Q) + Exploration (c * P * sqrt(N_parent) / (1 + N_child))
        """
        q_val = child.q_value
        exploration_term = self.c_puct * child.prior_prob * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        return q_val + exploration_term

    def backpropagate(self, node: Node, value: float):
        """Update N (visit count) and W (value sum) all the way up to the root."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            # In chess, a move that is good for White is bad for Black. 
            # We invert the value as we step up the tree for the opponent's turn.
            value = -value 
            current = current.parent

    def predict(self, board, encoder):
        """
        Helper to wrap the PyTorch model prediction. 
        Passes the board to TabulaNet, extracts the scalar Value, and maps the Policy back to legal moves.
        """
        # encoder is the encode_board function which returns a numpy array
        numpy_board = encoder(board)
        tensor_board = torch.tensor(numpy_board).unsqueeze(0) # Shape: (1, 13, 8, 8)
        
        self.model.eval()
        with torch.no_grad():
            policy_logits, value_tensor = self.model(tensor_board)
            
        v = value_tensor.item()
        
        # Note: True AlphaZero masks illegal moves and applies softmax.
        # For now, we simulate the dictionary mapping of legal moves -> probabilities.
        policy_dict = {}
        for move in board.legal_moves:
            # We will wire up the true action-space mapping here later.
            # Right now, we fallback to uniform probability among legal moves for testing.
            policy_dict[move] = 1.0 / board.legal_moves.count()
            
        return policy_dict, v
