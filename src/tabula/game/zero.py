"""Generation 0: a legal move, chosen at random. No weights."""

from __future__ import annotations

import random

from tabula.game.rules import Game


def random_reply(game: Game, rng: random.Random | None = None) -> str:
    moves = game.legal_moves()
    if not moves:
        raise RuntimeError("no legal move")
    chooser = rng if rng is not None else random.Random()
    return chooser.choice(moves)
