"""Sanity tests for the Connect 4 game logic and the network plumbing."""
import numpy as np

from alphazero.games.connect4 import Connect4


def test_drop_and_turn_alternation():
    g = Connect4()
    s = g.initial_state()
    assert g.current_player(s) == 1
    s = g.next_state(s, 3)
    assert g.current_player(s) == -1
    # Disc lands on the bottom row.
    assert s[0][0, 3] == 1


def test_vertical_win():
    g = Connect4()
    s = g.initial_state()
    # Player +1 drops 4 in column 0; player -1 answers in column 1.
    for _ in range(3):
        s = g.next_state(s, 0)  # +1
        s = g.next_state(s, 1)  # -1
    assert not g.is_terminal(s)
    s = g.next_state(s, 0)      # +1's 4th in a row
    assert g.is_terminal(s)
    assert g.result(s) == 1


def test_horizontal_win():
    g = Connect4()
    s = g.initial_state()
    for c in range(3):
        s = g.next_state(s, c)      # +1 along the bottom row
        s = g.next_state(s, c)      # -1 stacks on top
    s = g.next_state(s, 3)          # +1 completes 4 in a row
    assert g.is_terminal(s)
    assert g.result(s) == 1


def test_legal_actions_full_column():
    g = Connect4()
    s = g.initial_state()
    for _ in range(6):
        s = g.next_state(s, 2)
    assert not g.legal_actions(s)[2]   # column 2 is full
    assert g.legal_actions(s)[0]       # others still open


def test_encode_shape_and_canonical():
    g = Connect4()
    s = g.initial_state()
    s = g.next_state(s, 3)  # now it's player -1 to move
    enc = g.encode(s)
    assert enc.shape == (3, 6, 7)
    # The just-placed +1 disc is the *opponent* from the mover's (-1) view.
    assert enc[1, 0, 3] == 1.0
    assert enc[0, 0, 3] == 0.0


def test_symmetry_mirror():
    g = Connect4()
    s = g.initial_state()
    s = g.next_state(s, 0)
    enc = g.encode(s)
    pol = np.arange(7, dtype=np.float32)
    syms = list(g.symmetries(enc, pol))
    assert len(syms) == 2
    # Mirrored policy reverses column order.
    assert np.array_equal(syms[1][1], pol[::-1])
