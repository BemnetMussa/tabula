"""A chess game that rejects illegal moves and names why it ended."""

from __future__ import annotations

import chess


class IllegalMove(Exception):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__(f"illegal move: {text}")


_REASONS = {
    chess.Termination.CHECKMATE: "checkmate",
    chess.Termination.STALEMATE: "stalemate",
    chess.Termination.INSUFFICIENT_MATERIAL: "insufficient material",
    chess.Termination.SEVENTYFIVE_MOVES: "seventy-five-move rule",
    chess.Termination.FIVEFOLD_REPETITION: "fivefold repetition",
    chess.Termination.FIFTY_MOVES: "fifty-move rule",
    chess.Termination.THREEFOLD_REPETITION: "threefold repetition",
}


class Game:
    def __init__(self, fen: str | None = None) -> None:
        self.board = chess.Board(fen) if fen else chess.Board()

    def legal_moves(self) -> list[str]:
        return [move.uci() for move in self.board.legal_moves]

    def push(self, text: str) -> str:
        raw = text.strip()
        move = self._parse(raw)
        if move not in self.board.legal_moves:
            raise IllegalMove(raw)
        self.board.push(move)
        return move.uci()

    def result(self) -> str | None:
        if not self.board.is_game_over(claim_draw=True):
            return None
        return self.board.result(claim_draw=True)

    def reason(self) -> str | None:
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return None
        return _REASONS.get(outcome.termination, outcome.termination.name.lower())

    def _parse(self, text: str) -> chess.Move:
        try:
            return chess.Move.from_uci(text)
        except ValueError:
            pass
        try:
            return self.board.parse_san(text)
        except ValueError:
            raise IllegalMove(text) from None
