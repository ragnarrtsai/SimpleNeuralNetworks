"""Connect 4 — the classic 7-wide, 6-tall "drop a disc" game.

State representation
--------------------
state = (board, player)
    board : int8 array (6, 7), +1 / -1 / 0, in *absolute* coordinates
            (board[0] is the bottom row where pieces land).
    player: +1 or -1, whose turn it is.

The network sees the *canonical* board (board * player), so it always reasons as
if it were player +1. Connect 4 has a left-right mirror symmetry, which we use
to double the training data.
"""
from __future__ import annotations

import numpy as np

from alphazero.game import Game

ROWS = 6
COLS = 7
WIN = 4


class Connect4(Game):
    name = "connect4"
    rows = ROWS
    cols = COLS

    @property
    def action_size(self) -> int:
        return COLS  # one action per column

    @property
    def encoded_shape(self) -> tuple[int, int, int]:
        return (3, ROWS, COLS)

    def initial_state(self):
        return (np.zeros((ROWS, COLS), dtype=np.int8), 1)

    def current_player(self, state) -> int:
        return state[1]

    def legal_actions(self, state) -> np.ndarray:
        board, _ = state
        return board[ROWS - 1] == 0  # a column is playable if its top cell is empty

    def _drop_row(self, board, col: int) -> int:
        """Lowest empty row in `col`, or -1 if full."""
        for r in range(ROWS):
            if board[r, col] == 0:
                return r
        return -1

    def next_state(self, state, action: int):
        board, player = state
        row = self._drop_row(board, action)
        if row < 0:
            raise ValueError(f"Illegal move: column {action} is full")
        new_board = board.copy()
        new_board[row, action] = player
        return (new_board, -player)

    def is_terminal(self, state) -> bool:
        board, _ = state
        if self._winner(board) != 0:
            return True
        return not (board[ROWS - 1] == 0).any()  # board full -> draw

    def result(self, state) -> float:
        # Outcome from player +1's perspective.
        return float(self._winner(state[0]))

    def _winner(self, board) -> int:
        """+1 / -1 if someone has 4 in a row, else 0."""
        for r in range(ROWS):
            for c in range(COLS):
                p = board[r, c]
                if p == 0:
                    continue
                # Check the 4 forward directions from this cell.
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    rr, cc = r, c
                    count = 0
                    while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr, cc] == p:
                        count += 1
                        if count == WIN:
                            return int(p)
                        rr += dr
                        cc += dc
        return 0

    def encode(self, state) -> np.ndarray:
        board, player = state
        canonical = board * player  # +1 = "my" piece from the mover's view
        planes = np.zeros((3, ROWS, COLS), dtype=np.float32)
        planes[0] = (canonical == 1).astype(np.float32)   # my pieces
        planes[1] = (canonical == -1).astype(np.float32)  # opponent pieces
        planes[2] = 1.0                                    # bias / side-to-move plane
        return planes

    def string_key(self, state) -> str:
        board, player = state
        return f"{player}:" + board.tobytes().hex()

    def render(self, state) -> str:
        board, _ = state
        glyph = {1: "X", -1: "O", 0: "."}
        lines = []
        for r in range(ROWS - 1, -1, -1):  # print top row first
            lines.append(" ".join(glyph[int(board[r, c])] for c in range(COLS)))
        lines.append(" ".join(str(c) for c in range(COLS)))  # column labels
        return "\n".join(lines)

    def symmetries(self, encoded: np.ndarray, policy: np.ndarray):
        # Left-right mirror: flip the board columns and reverse the policy.
        yield encoded, policy
        yield encoded[:, :, ::-1].copy(), policy[::-1].copy()
