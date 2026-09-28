# Tabula

Chess learned from the rules alone. Each generation plays the previous one. A new generation is kept only when it wins that match. The board lets you play any generation that earned its place.

The contract for this phase is [prd/Tabula-PRD.md](prd/Tabula-PRD.md). Target: a demo that runs on this machine by 8 October 2026.

## Layout

```
prd/                 what this phase must do
src/tabula/game/     rules, legal moves, board encoding
src/tabula/model/    policy head and value head
src/tabula/mcts/     search guided by those heads
src/tabula/selfplay/ games Tabula plays against itself
src/tabula/train/    updates and the generation ladder
src/tabula/eval/     the 20-game match between generations
src/tabula/serve/    local board API
web/                 the board
tests/               acceptance checks for the PRD
checkpoints/         weights on this machine, not in git
```

## This machine

Training for this phase is meant for the i7-12700H, 16 GB RAM, and RTX 3050 Ti 4 GB already on this PC. A larger GPU is a later run, after the demo script passes.
