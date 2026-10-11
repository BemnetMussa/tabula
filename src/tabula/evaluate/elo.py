"""
Tabula AI — Generational Elo Ladder & Rating Tracker.
Implements Google DeepMind's AlphaGo Zero / AlphaZero generational Elo rating system:
Delta Elo = 400 * log10(S / (1 - S)) where S is the match score fraction.
Persists ratings to data/elo_ladder.json across sessions.
"""

import os
import json
import math


class EloLadder:
    LADDER_FILE = "data/elo_ladder.json"
    BASELINE_ELO = 400

    def __init__(self):
        self.generation = 0
        self.champion_elo = self.BASELINE_ELO
        self.last_delta = 0
        self.history = [{"gen": 0, "elo": self.BASELINE_ELO, "delta": 0, "win_rate": 0.5}]
        self.load()

    def load(self):
        if os.path.exists(self.LADDER_FILE):
            try:
                with open(self.LADDER_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.generation = data.get("current_gen", 0)
                    self.champion_elo = data.get("champion_elo", self.BASELINE_ELO)
                    self.last_delta = data.get("last_delta", 0)
                    self.history = data.get("history", self.history)
            except Exception:
                pass

    def save(self):
        os.makedirs("data", exist_ok=True)
        try:
            with open(self.LADDER_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "current_gen": self.generation,
                    "champion_elo": self.champion_elo,
                    "last_delta": self.last_delta,
                    "history": self.history
                }, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def calculate_delta(win_rate: float) -> int:
        """
        DeepMind / FIDE Elo Logistic Formula:
        Delta = 400 * log10(S / (1 - S))
        """
        clamped = max(0.05, min(0.95, win_rate))
        delta = 400.0 * math.log10(clamped / (1.0 - clamped))
        return int(round(delta))

    def record_match(self, win_rate: float, promoted: bool, iteration: int = 0) -> tuple[int, int]:
        """
        Records a tournament outcome.
        Returns: (champion_elo, delta_elo)
        """
        delta = self.calculate_delta(win_rate)
        self.last_delta = delta
        if promoted:
            self.generation += 1
            self.champion_elo += delta
            self.history.append({
                "gen": self.generation,
                "iteration": iteration,
                "elo": self.champion_elo,
                "delta": delta,
                "win_rate": round(win_rate, 3)
            })
            self.save()
        return self.champion_elo, delta

    def get_sparkline_values(self, max_points: int = 16) -> list[int]:
        return [entry["elo"] for entry in self.history[-max_points:]]
