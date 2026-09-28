# Tabula — PRD

Status: Floor locked
Owner: Bemnet
Engineering / design: Bemnet
Target release: 2026-10-08
Last updated: 2026-09-29

## Problem

Bemnet does not play chess, and his friends give him a hard time for it. He cannot sit down with them this month as someone who already knows the game, because he never learned it. On 19 July 2026 he pointed at a ten-year-old match: a player that first copied games people had already played, then played itself, and won. He wants that second kind of climb, on chess, on the computer he already has, inside ten days.

## Goals

- Done when Bemnet can play a full legal game against the latest saved generation, switch to an earlier generation, and show a stored match in which the later generation scored over half the points.
- Training for that climb reads no file of human games.
- This phase is complete when the demo script runs on this machine. A rating, a friend count, or a win against a titled player is not the exit test.

## Assumptions

- Day-one play is on this laptop, with friends in the room or watching the same screen. *Inference from the 19 July and 29 September messages.*
- The people he wants to surprise play casual chess. Beating them is a hope for later generations, not the bar for this phase. *Inference.*
- The training machine measured on 29 September 2026 is an i7-12700H, 16 GB RAM, and an RTX 3050 Ti with 4 GB of VRAM. The network for this phase has to load on that GPU.
- "Master chess" means a later training run on a larger GPU. This phase proves the climb on this PC.

## Users

- Day-one: Bemnet. He starts training, plays generations, and shows the ladder.
- Later (not required to pass): a friend who sits at the same board and plays a game.

## Analog

AlphaZero (Silver et al., 2018). The slice we copy is a player that starts from the rules, plays itself, and keeps a generation only when it beats the previous one. We are not copying the chip cluster, and we are not copying Go.

## In scope (must ship)

1. A local board where Bemnet plays a full legal game against a chosen generation.
2. Generation 0, which knows the rules and plays before any training has been saved.
3. A training loop whose only games are games Tabula played against itself.
4. A ladder that saves a new generation only when it scores over half the points in a 20-game match against the current latest generation. A win is 1, a draw is 1/2, a loss is 0. The bar is more than 10 points.
5. A way to stop training and start it again without deleting the latest generation.
6. A screen that names the generation being played and shows each generation's match score against its parent.

## This phase does not include

- This phase does not include human game records, opening books, or endgame tablebases.
- This phase does not include wrapping an existing engine and presenting that engine as Tabula.
- This phase does not include using another engine's games, evaluations, or move choices as training labels.
- This phase does not include a strength bar against friends, club players, or titled players.
- This phase does not include accounts, a public site, matchmaking, or a mobile client.
- This phase does not include a second game besides chess.
- This phase does not include the larger-GPU run that would chase master-level play. That run starts only after this demo script passes.

## Journeys

1. First-time: Bemnet opens the board before any training. He plays generation 0. Every reply is legal. The screen says generation 0.
2. Returning: He starts training, leaves it, and comes back to a later generation that has a stored win against its parent. He plays that generation.
3. Desk: A friend sits at the same machine, picks a generation, and plays a game through to a result.
4. Failure: Training stops or the GPU run errors. The board still opens on the last saved generation. If a chosen weights file is missing, the board says so and does not start the game. If no generation has been saved, the board offers generation 0.

## User stories and acceptance

| ID | Story | Given / When / Then |
| --- | --- | --- |
| P0-1 | Play before any training | Given no generation has been saved, when Bemnet opens the board and makes a legal move, then the reply is a legal move and the screen shows generation 0. |
| P0-2 | Reject an illegal move | Given a game is in progress, when a move is submitted that the rules reject, then the position stays unchanged and the board says the move is illegal. |
| P0-3 | Finish a game | Given a position is checkmate, stalemate, threefold repetition, the fifty-move rule, or insufficient material, when that position is reached, then the board shows that result and accepts no further move. |
| P0-4 | Keep a stronger generation | Given training is running, when a candidate scores more than 10 points in a 20-game match against the latest generation, then the candidate is saved and becomes the latest generation. |
| P0-5 | Drop a weaker generation | Given training is running, when a candidate scores 10 points or fewer in that same match, then the latest generation stays unchanged. |
| P0-6 | Resume | Given training was stopped after a generation was saved, when training starts again, then it continues from that generation and does not delete it. |
| P0-7 | No human games | Given a training run starts, when it looks for games to learn from, then it does not read a file of human games. The only stored games are games Tabula played. |
| P0-8 | Choose a generation | Given two or more generations are saved, when Bemnet selects an earlier one, then the next game is played by that generation. |
| P0-9 | Missing weights | Given Bemnet selects a generation whose weights file is not on disk, when the board tries to start, then it shows that the file is missing and does not start the game. |

## Entities

- Generation: id, parent id, path to weights, time saved, points scored against the parent out of 20.
- Game: id, white player (a generation id or human), black player, move list, result.
- Position: the current board, stored so a game can be resumed and so an illegal move can be rejected.
- Training run: id, latest generation id, status (running, stopped, or error), last error text when status is error.

## User interaction and design

- Figma: none. The board is built in the repo under `web/`.
- Board, default: a position, the side to move, the generation name, and a result line once the game is over.
- Board, empty: no game started. A control starts one against the selected generation. Before any save, the only choice is generation 0.
- Board, loading: a thinking state from the moment a human move is accepted until a legal reply is applied or an error is shown.
- Board, error: illegal move, missing weights, or training error. The position does not change on an illegal move.
- Ladder: each saved generation and its points against its parent. Empty of match rows when only generation 0 exists.
- Training: running, stopped, or the error text from the last failed start.

## Non-functional

- The network used for human games loads on the RTX 3050 Ti 4 GB.
- Stopping the training process does not delete the latest generation.
- Weights, replay files, and any `.env` file stay on this machine and out of git.
- The board runs locally on this machine. It does not require an account or a network connection to play a saved generation.
- The board remains on the thinking state until a legal reply is applied or an error is shown.

## Dependencies

- Python 3.12 on this machine, `python-chess` for the rules, and a PyTorch build that can use the RTX 3050 Ti.
- Local disk for weights and replay. No API keys.
- Repo layout: `prd/` for this contract, `src/tabula/` for the player (`game`, `model`, `mcts`, `selfplay`, `train`, `eval`, `serve`), `web/` for the board, `tests/` for the acceptance checks, `checkpoints/` for local weights.

## Milestones

| Phase | Scope | Exit test |
| --- | --- | --- |
| 1. Rules and board (days 1–2) | Generation 0 and the local board | P0-1, P0-2, and P0-3 pass with no weights file on disk. |
| 2. Network and search (days 3–5) | A small policy head, a value head, and search | One training step writes a weights file, and that generation's replies are legal and are not the same every time the position is unchanged. |
| 3. Ladder (days 6–8) | Self-play matches and resume | P0-4, P0-5, P0-6, and P0-7 pass. |
| 4. Demo (days 9–10) | Generation picker and failure states | P0-8 and P0-9 pass, and the demo script runs without hand-editing files. |

## Demo script

1. Delete or hide any saved weights. Open the board. Play moves against generation 0. Confirm each reply is legal and the screen says generation 0. Try one illegal move and confirm the position does not change.
2. Play until the game ends by a rule, or resign the setup and start a position that is already terminal in a test. Confirm the result is shown and no further move is accepted.
3. Start training. Wait until a candidate either replaces the latest generation or is rejected. Confirm the ladder shows the points of that match.
4. Stop training. Start it again. Confirm the latest generation is still there.
5. Play a full game against the latest generation through to a result.
6. Select generation 0 and play one move. Confirm the reply comes from generation 0.
7. Select a generation whose file has been removed. Confirm the board reports the missing file and does not start.

## Open questions

| Question | Owner | Needed by |
| --- | --- | --- |
| Do friends need to open the board from their own phones, or is the same laptop enough for the demo? | Bemnet | Day 6, before the board is polished |
| After the demo passes, which rented or free GPU runs the longer training? | Bemnet | After day 10 |

## Appendix

- Bemnet, Garage messages, 19 July 2026: the Go match, then a player that used no human games and beat the earlier one; the wish to try that on chess.
- This machine, measured 29 September 2026: i7-12700H, 16 GB RAM, RTX 3050 Ti Mobile 4 GB, 35 W.
- Analog paper: Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play," Science, 2018.
