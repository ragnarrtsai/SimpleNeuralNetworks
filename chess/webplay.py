"""Tiny web UI to play Connect 4 against a trained AlphaZero network.

No web framework — just the standard library. Run it, open the printed URL.

    python webplay.py                       # use checkpoints/best.pt
    python webplay.py --checkpoint x.pt --port 8000

The browser keeps the move history and POSTs it to /api/move; the server replays
the moves, runs MCTS for the side to move, and returns the AI's reply plus the
game status. The server is stateless — refreshing starts a new game.
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

from alphazero.games import make_game
from alphazero.mcts import MCTS
from alphazero.network import AlphaZeroNet
from config import Config, resolve_device

HTML_PATH = os.path.join(os.path.dirname(__file__), "templates", "connect4.html")

# Loaded once at startup.
CFG = Config()
DEVICE = resolve_device(CFG.device)
GAME = make_game("connect4")
NET = AlphaZeroNet(GAME, CFG.network.channels, CFG.network.num_res_blocks).to(DEVICE)


def load_net(checkpoint: str):
    if os.path.exists(checkpoint):
        NET.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        print(f"Loaded {checkpoint}")
    else:
        print(f"WARNING: no checkpoint at {checkpoint}; AI will be untrained (random).")
    NET.eval()


def replay(moves):
    """Rebuild the game state from a list of column indices."""
    state = GAME.initial_state()
    for col in moves:
        if GAME.is_terminal(state):
            break
        state = GAME.next_state(state, int(col))
    return state


def board_payload(state):
    board, _ = state
    # Return rows top-first so the browser can render row 0 at the top.
    return [[int(board[r, c]) for c in range(GAME.cols)] for r in range(GAME.rows - 1, -1, -1)]


def status_of(state):
    if not GAME.is_terminal(state):
        return {"terminal": False, "winner": 0}
    return {"terminal": True, "winner": int(GAME.result(state))}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet console
        pass

    def _send(self, code, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(HTML_PATH, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/api/move", "/api/analyze"):
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        moves = req.get("moves", [])
        sims = int(req.get("sims", CFG.mcts.num_simulations))

        state = replay(moves)

        # Analysis: per-column search preference + win estimate for the side to move.
        if self.path == "/api/analyze":
            st = status_of(state)
            if st["terminal"]:
                self._send(200, {"terminal": True, "winner": st["winner"]})
                return
            legal = GAME.legal_actions(state)
            mcts_cfg = CFG.mcts.__class__(**{**CFG.mcts.__dict__, "num_simulations": sims})
            policy = MCTS(GAME, NET, DEVICE, mcts_cfg).run(state, add_noise=False)
            _, value = NET.predict(GAME.encode(state), legal, DEVICE)
            self._send(200, {
                "terminal": False,
                "policy": [float(p) for p in policy],     # visit share per column
                "legal": [bool(b) for b in legal],
                "win_pct": (value + 1.0) / 2.0 * 100.0,    # side-to-move win estimate
                "best": int(np.argmax(policy)),
            })
            return

        # If the human's move just ended the game, report it (no AI reply).
        st = status_of(state)
        if st["terminal"]:
            self._send(200, {"board": board_payload(state), "ai_move": None, **st})
            return

        # Otherwise the AI (side to move) replies.
        mcts_cfg = CFG.mcts.__class__(**{**CFG.mcts.__dict__, "num_simulations": sims})
        mcts = MCTS(GAME, NET, DEVICE, mcts_cfg)
        policy = mcts.run(state, add_noise=False)
        ai_move = int(np.argmax(policy))
        state = GAME.next_state(state, ai_move)

        self._send(200, {"board": board_payload(state), "ai_move": ai_move,
                         **status_of(state)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    load_net(args.checkpoint)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  ▶  Open  http://127.0.0.1:{args.port}  in your browser  (Ctrl+C to stop)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye!")


if __name__ == "__main__":
    main()
