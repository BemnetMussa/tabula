# Tabula — ten days

Folder: `Documents/AI/tabula`. This overview stays loose on purpose. We read as we go, then change the net and the search so they fit this PC. The pass/fail list is in [prd/Tabula-PRD.md](prd/Tabula-PRD.md).

## This PC

Measured 29 September 2026:

- Intel i7-12700H, 20 threads
- 16 GB RAM
- NVIDIA GeForce RTX 3050 Ti Mobile, 4 GB VRAM, 35 W

Nothing about network size or search depth is frozen. If a step does not fit this GPU, we shrink it the same day and write the change below.

## Days

1. **Rules.** Legal chess. Generation 0 plays a random legal move. Play it from the terminal.
2. **Board.** A page on this machine: the position, your move, the reply, and the result.
3. **Read.** How a self-play player turns a board into input and a move into a label. Notes land in "What we changed".
4. **A small net.** A policy head and a value head. One forward pass on the 3050 Ti. If it does not fit in 4 GB, we cut it down the same day.
5. **Search.** The net guides a tree of future moves.
6. **Self-play.** Tabula plays a full game against itself, and we store that game.
7. **One update.** The net learns from games it played. Stopping the run keeps the last weights.
8. **The ladder.** A 20-game match. A new generation replaces the old one only when it scores more than half the points.
9. **Play any generation.** Pick generation 0 or the latest on the board. Training can stop and start again.
10. **Demo.** Play both generations, show the match score, and write what this GPU can sustain overnight.

## What we changed

- 29 Sep 2026 — Started. No network size chosen yet. Generation 0 is random legal moves, so day 1 does not need the GPU.
