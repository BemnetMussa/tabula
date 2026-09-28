"""Play White against generation 0 in the terminal."""

from __future__ import annotations

import random

from tabula.game import Game, IllegalMove, random_reply


def main() -> None:
    game = Game()
    rng = random.Random()
    print("You are White. Generation 0 is Black.")
    print("Moves: e2e4 or e4. Quit with q.")
    print(game.board)
    while game.result() is None:
        if game.board.turn:
            typed = input("\nYour move: ").strip()
            if typed.lower() in {"q", "quit"}:
                print("stopped")
                return
            try:
                played = game.push(typed)
            except IllegalMove:
                print(f"illegal move: {typed}")
                continue
            print(f"you: {played}")
        else:
            played = game.push(random_reply(game, rng))
            print(f"generation 0: {played}")
        print(game.board)
    print(f"\n{game.result()} ({game.reason()})")


if __name__ == "__main__":
    main()
