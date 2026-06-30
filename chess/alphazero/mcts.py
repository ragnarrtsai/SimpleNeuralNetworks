"""Monte Carlo Tree Search with a neural-network prior (AlphaZero PUCT).

No random rollouts: leaf values come from the network's value head, and the
search is guided by the network's policy prior via the PUCT formula

    U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))

All values are stored from the perspective of the player to move at each node,
so a child's value is negated when backed up to its parent.
"""
from __future__ import annotations

import math

import numpy as np

from alphazero.game import Game


class _Node:
    __slots__ = (
        "prior", "visit_count", "value_sum", "children",
        "state", "is_expanded", "_pending_action",
    )

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, "_Node"] = {}
        self.state = None
        self.is_expanded = False
        self._pending_action = -1

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class MCTS:
    def __init__(self, game: Game, network, device: str, config):
        self.game = game
        self.net = network
        self.device = device
        self.cfg = config

    def run(self, root_state, add_noise: bool = True) -> np.ndarray:
        """Return a visit-count policy (probabilities over actions) for `root_state`."""
        root = _Node(prior=0.0)
        root.state = root_state
        self._expand(root)
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.cfg.num_simulations):
            self._simulate(root)

        counts = np.zeros(self.game.action_size, dtype=np.float32)
        for action, child in root.children.items():
            counts[action] = child.visit_count
        total = counts.sum()
        return counts / total if total > 0 else counts

    def _simulate(self, root: _Node):
        path = [root]
        node = root

        # Selection: descend by PUCT until we hit an unexpanded node.
        while node.is_expanded:
            action, node = self._select_child(node)
            path.append(node)

        leaf = path[-1]
        parent = path[-2]
        # Lazily materialize the leaf state from its parent + the chosen action.
        leaf.state = self.game.next_state(parent.state, leaf._pending_action)

        if self.game.is_terminal(leaf.state):
            # Value must be from the LEAF mover's perspective, matching what
            # _expand returns for non-terminal leaves (the backup adds it to the
            # leaf node first). result() is from +1's view, so convert with the
            # leaf's own side to move. Using the parent's side here flips the
            # sign and makes the search blind to immediate wins/losses.
            outcome = self.game.result(leaf.state)
            value = outcome * self.game.current_player(leaf.state)
        else:
            value = self._expand(leaf)

        # Backup: alternate sign up the tree (zero-sum, perspective flips each ply).
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _expand(self, node: _Node) -> float:
        """Evaluate `node.state` with the network, create children, return its value."""
        legal = self.game.legal_actions(node.state)
        encoded = self.game.encode(node.state)
        probs, value = self.net.predict(encoded, legal, self.device)

        for action in np.nonzero(legal)[0]:
            child = _Node(prior=float(probs[action]))
            child._pending_action = int(action)  # state built lazily on first visit
            node.children[int(action)] = child
        node.is_expanded = True
        return value

    def _select_child(self, node: _Node):
        c_puct = self.cfg.c_puct
        sqrt_total = math.sqrt(node.visit_count)
        best_score = -float("inf")
        best = None
        for action, child in node.children.items():
            q = -child.value  # child value is from the child's mover; negate for parent
            u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best = (action, child)
        return best

    def _add_dirichlet_noise(self, root: _Node):
        actions = list(root.children.keys())
        if not actions:
            return
        noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(actions))
        eps = self.cfg.dirichlet_epsilon
        for action, n in zip(actions, noise):
            child = root.children[action]
            child.prior = (1 - eps) * child.prior + eps * n
