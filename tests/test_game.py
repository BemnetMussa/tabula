import random
import unittest

from tabula.game import Game, IllegalMove, random_reply


class GameTests(unittest.TestCase):
    def test_illegal_move_leaves_the_position(self) -> None:
        game = Game()
        before = game.board.fen()
        with self.assertRaises(IllegalMove):
            game.push("e2e5")
        self.assertEqual(game.board.fen(), before)
        self.assertIsNone(game.result())

    def test_san_and_uci_both_play(self) -> None:
        game = Game()
        self.assertEqual(game.push("e4"), "e2e4")
        self.assertEqual(game.push("e7e5"), "e7e5")

    def test_fools_mate(self) -> None:
        game = Game()
        for move in ("f3", "e5", "g4", "Qh4"):
            game.push(move)
        self.assertEqual(game.result(), "0-1")
        self.assertEqual(game.reason(), "checkmate")
        with self.assertRaises(IllegalMove):
            game.push("a2a3")

    def test_kings_only_is_a_draw(self) -> None:
        game = Game("8/8/8/4k3/8/4K3/8/8 w - - 0 1")
        self.assertEqual(game.result(), "1/2-1/2")
        self.assertEqual(game.reason(), "insufficient material")

    def test_stalemate(self) -> None:
        game = Game("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(game.result(), "1/2-1/2")
        self.assertEqual(game.reason(), "stalemate")

    def test_generation_zero_is_legal_and_repeatable(self) -> None:
        game = Game()
        first = random_reply(game, random.Random(0))
        second = random_reply(game, random.Random(0))
        self.assertEqual(first, second)
        self.assertIn(first, game.legal_moves())


if __name__ == "__main__":
    unittest.main()
