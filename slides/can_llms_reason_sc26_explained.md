# Can LLMs reason? The SC26 talk, slide by slide

A companion to `slides/can_llms_reason_sc26.pptx`. For each slide: what it shows, how to read
its figure, what the numbers mean, the caveats, and answers to the questions the audience is
likely to ask. Every number comes from our own result files or job logs, and the source is
given. Numbers taken from the original papers, or from their human games, are tagged
UNVERIFIED because we transcribed them rather than recomputed them.

The two projects:
- **Decrypto** measures reasoning in models as they are, using a word game.
- **MARSHAL** trains a model by having it play games against itself, then measures what changed.

Both ran on Polaris at the Argonne Leadership Computing Facility.

---

## Slide 1: Title

Two projects, one question, one machine. Before presenting, fill in the presenter name and
department, and check the "SC26" label.

---

## Slide 2: We measured reasoning with one game and trained it with others

**The question.** Can language models reason? Here that means three things: planning ahead,
using information that only some players have, and thinking about what another player knows.

**Why games.**
- **A fresh problem every deal.** Exam-style questions can leak into a model's training data;
  a new game can't.
- **The rules decide the score.** No person, and no second model, has to grade the answer.
- **They need the skills we care about.** Winning takes planning, hidden information and
  reading the other player.

**Decrypto (left).** The diagram is from the original Decrypto paper (Lupu, Willi & Foerster,
2025).
1. Alice, the **encoder**, sees four secret keywords and a 3-digit code, for example 2-3-4. She
   writes one hint per digit.
2. Bob, the **decoder**, knows the keywords and tries to read the code back.
3. Eve, the **interceptor** (the opponent), never sees the keywords. She does see every earlier
   hint, filed under the keyword number it pointed to, and she guesses the code too.
4. The team loses after 2 interceptions or 2 misreads, and wins by lasting 8 rounds.

In our run a **team** is one model playing both encoder and decoder. Each of 4 models played as
a team against each of the 4 as opponent: 3 seeds × 32 games per pairing, 1,536 games, with no
training.

**MARSHAL (right).** One training step is a loop:
1. **Play:** the model plays both seats of a game, thinking in text before each move.
2. **Score:** the game engine judges the result.
3. **Learn:** the model is shifted toward the moves that preceded wins.
4. **Copy:** the updated model is sent to the players.

The model is Qwen3-4B, a 4-billion-parameter model. It trained for 200 steps on TicTacToe, Kuhn
poker and Mini Hanabi (a cooperative card game). It was tested on games it never trained on:
Connect Four, Leduc poker, and Simple and Full Hanabi. There were two runs: R6 trained on all
three games (4 nodes), and R5 on TicTacToe alone (3 nodes).

**If someone asks: how does MARSHAL inform Decrypto?**
- **Science.** MARSHAL claims that practice at games transfers to new problems. Decrypto is a
  hard test of that claim: it's played with words, it mixes cooperation and competition, and
  winning means anticipating what the opponent can infer. MARSHAL never saw it.
- **The experiment.** Put the trained models in Decrypto in place of the untrained Qwen3-4B,
  against the same opponents on the same seeds, so any difference comes from training. The
  two checkpoints are R6 checkpoint 149, the last one before that run broke down, and R5
  checkpoint 199. They were submitted on 2026-11-18 as jobs 7860393 and 7860394, and are still
  queued; see slide 8.
- **Machine.** Decrypto's job scripts reuse Polaris lessons MARSHAL learned first:
  - copy the Python environment to each node's local disk as one file, because importing it
    from the shared filesystem could take about 19 minutes;
  - compute nodes are offline, so every model is downloaded in advance;
  - each job is capped at 4,096 processes;
  - pin package versions known to work;
  - stop loading the system conda module (September 2026).

  The web-proxy hang between nodes was Decrypto's own finding (job 7186962), not MARSHAL's.

**The Polaris strip.** Each node has 4 NVIDIA A100 GPUs with 40 GB each. Our jobs used 3 to 4
nodes.

---

## Slide 3: The opponent cracked the code in 95% of 1,536 games

**What an interception is.** A round in which the opponent works out the team's secret code. If
earlier hints "raid" and "war" were filed under keyword 3 and "ship" under keyword 1, then new
hints like "norse" and "sail" give the code away. Two interceptions and the opponent wins. In
the slides, "the opponent cracked the code" means the same thing.

**How to read the figure** (`results/polaris_4model_competitive/figures/fig1_competitive_grids.png`).
- **Each square** is one pairing: a team (row) against an opponent (column), over 96 games.
  The small number is the standard error across the 3 seeds.
- **Left panel:** the share of those games the opponent won by cracking the code twice. It is
  0.81 or more in every square.
- **Right panel:** how many rounds the games lasted.
- **The Qwen3-4B column** is the lightest on the left and the darkest on the right. The weakest
  code-cracker let teams last longest.

**What the numbers mean.**

| How the 1,536 games ended | Games | Share |
|---|---|---|
| Opponent cracked the code twice | 1,454 | 94.7% |
| Team misread its own code twice | 38 | 2.5% |
| Team lasted all 8 rounds | 44 | 2.9% |

- The 1,454 includes the few games where both limits were reached in the same round, counted as
  in the paper's analysis.
- Games lasted 4.31 ± 0.03 rounds on average. The paper's models lasted 4.33 ± 0.03 (UNVERIFIED);
  the difference is not significant (p = 0.65). The paper's 10 human games, against
  Llama-3.1-70B, lasted 6.90 rounds (UNVERIFIED).

**Why the opponent usually wins.** In round 1 the opponent is guessing among 24 possible codes,
and in round 1 it cracked 3.6% of codes, close to the 1 in 24 (4.2%) of a blind guess. After
that it has a growing list of earlier hints for each keyword. Its per-round success rose to
44.0% in round 2 and peaked at 68.9% in round 4. To stop it, the encoder has to write hints its
partner will get but that don't fit the visible pattern. That means thinking about what the
opponent can infer, and the models rarely managed it.

**If someone asks: is that the model, or the setup?** Partly the setup, for the Qwen3 models.
Our round-level audit re-read all 1,536 games (job 7860225,
`logs/analysis/round_failures_7860225.txt`).
- **Cut-offs:** the game stops a Qwen3 model's thinking at 2,000 tokens. That cut off 79.7% of
  Qwen3-8B's and 42.4% of Qwen3-4B's turns as the opponent.
- **Effect:** when Qwen3-8B finished thinking, it cracked 84.8% of codes, against 35.2% when cut
  off.
- **Interpretation:** we hypothesize the cap held the Qwen3 opponents back. Harder rounds may
  also make a model think longer and fail more, so the gap may partly reflect difficulty.
  Replaying the cut-off turns with a larger budget would test this.

**Caveats.**
- Claude ran through its command-line tool, with that tool's default sampling and a 180-second
  timeout. 73 calls timed out and went through the game's retry path.
- Everything ran at one temperature (0.6).
- The paper's numbers are transcribed, not recomputed.

Sources: `polaris_4model_competitive_results.md` (analysis job 7731552); jobs 7668935–7668937.

---

## Slide 4: One 4-node job served four models; the slowest model set how many games could run at once

**The layout.**
- **Node 0** runs the games and does no GPU work. It reaches Claude Sonnet 4.6 over the
  internet through the lab's web proxy.
- **Node 1** holds Qwen2.5-72B split across all 4 GPUs.
- **Nodes 2 and 3** hold Qwen3-8B and Qwen3-4B, on one GPU each.

Only 6 of the 16 GPUs hold a model, because Polaris allocates whole nodes. Packing the two small
models onto one node would free a node; we didn't need to.

**Memory.**
- The 72-billion-parameter model's weights are 136 GiB, so it fits only when split four ways:
  34 of each GPU's 40 GiB.
- What's left holds 18,800 tokens of conversation, about 2.3 full-length requests of 8,192
  tokens at once.
- It loaded on the third configuration we tried. Jobs 7197372 and 7197374 had no memory left
  for conversations; job 7197375 worked.

**The bars** (from the earlier 3-model run, 405 games).
- **All 405 games at once:** the 72B server dropped 610 of 1,929 requests (32%), and 160 games
  were lost. Only 245 finished.
- **24 games at a time:** 0 of 1,736 requests dropped, and 403 games finished in 2 h 27 min.
  The last 2 had hit a separate fault, an answer parser that crashed on empty replies. We
  fixed it and reran them, reaching 405 of 405.
- **Why the slowest model sets the pace:** requests beyond the 72B's memory wait in line, time
  out, and take their game with them. Adding games past that point loses games instead of
  saving time.

**Scheduling.**
- **Separate jobs failed.** The servers started hours before the game job (a skew of 3.4–7 h)
  and ran out their time limits idle.
- **One fused job works.** A single job holding all four nodes starts and stops them together.
- **Shorter requests start sooner.** A 16-hour request waited 32 h with no start time; a
  10-hour request started 2.5 h after submission.

**A network detail, if someone asks.** Node 0 needed direct traffic to the model servers and
proxied traffic to Claude, both at once. The HTTP library ignored address ranges in the proxy
exception list, and each node name has two network addresses. So requests went to the proxy
and the servers never answered (jobs 7599882 and 7668624). The fix was to list each server's
exact registered address, and to fail the job at once if any request would still go through the
proxy (job 7668797).

**Cost.** The 405-game run used about 10 node-hours.

Sources: `polaris_scaleup_gpu_needs.md`; the job ledger in `polaris_pbs_notes.md`.

---

## Slide 5: Self-play removed illegal moves within 10 steps; after step ~170 the training broke down

**How to read the left panel** (panel c of `MARSHAL/analysis/r6/out/fig1_r6_vs_r5_validation.png`).
- **What it plots:** the share of test games lost by an unreadable or illegal move, with the
  model moving second, in TicTacToe and in Connect Four (never trained).
- **Lines:** solid is R6, the three-game run; dashed is R5, the TicTacToe-only run.
- **Bands:** the shaded areas are the windows we compared (steps 10–50 and 150–190).
- **Reading it:** both lines drop to near zero by step 10. Pooled over the two games, these
  forfeits went from 48.8% at step 0 to 0.8% over steps 10–50.

**How to read the right panel** (panel a of `fig2_training_selfplay.png`).
- **What it plots:** TicTacToe games the model played against itself during training: draws
  (blue) and forfeits (orange).
- **Reading it:** R6's forfeits shoot up after about step 170, while R5's stay flat.

**What training did, and didn't do.**
- **It stopped the model forfeiting:** legal moves, and reasoning short enough to finish within
  the 4,096-token turn limit.
- **One strategic gain:** in TicTacToe as the second mover, against a search player using 1,000
  simulations per move, draws among games played to the end rose from 37.8% to 59.6%
  (q = 0.048, after correcting for many comparisons). The first seat showed no gain.
- **Mini Hanabi:** the team score rose from 1.05 at step 0 to 2.07 ± 0.06 at steps 130–170;
  the paper reports 2.17 (App. J; UNVERIFIED). That came mostly from fewer over-long turns, not
  from better play.
- **Leduc poker, never trained:** no improvement. At steps 130–170 the three-game run won 23.5%
  of second-seat games, 20.9 ± 6.8 points below the one-game run (q = 0.038).

**The breakdown.** From about step 172, R6 got worse in every game, while its learning rate was
almost zero.
- TicTacToe self-play forfeits rose from 3.6% to 43.1%.
- Forfeits on the 8 never-trained test conditions rose from 24.7% to 63.8%.
- The weights did *not* start moving faster. The cause is open: either a fragile model, or stale
  state in game engines that ran 26.8 h through 95 in-place weight updates.
- **Test:** serve checkpoints 169, 189 and 199 in fresh engines. Checkpoint 149 is the last
  clean model, so it's the one we put into Decrypto.

**If someone asks: did it learn to reason?** Mostly it learned to *answer*: correctly formatted
moves, within the length limit. Better strategy showed up in one game and one seat.

Source: `MARSHAL/marshal_r6_results.md`, sections R1–R8. The MARSHAL session fact-checked these
claims on 2026-11-13.

---

## Slide 6: Across 4 nodes, generating games took 65% of each training step; copying the model took 0.2%

**The layout.**
- **Nodes 0–1 (the learner):** the model is split 4 ways inside each node, with one copy per
  node. A frozen reference copy shares these 8 GPUs.
- **Nodes 2–3 (the game players):** 8 game-playing engines, one GPU each.
- **Per step:** 192 games, 64 of each training game. Each game yields one record per seat, so
  384 records.

The released code assumes one 8-GPU server (the paper used 80 GB H100s) with every role on every
GPU. That doesn't fit on 40 GB cards, so we split the roles across nodes.

**The bar: one 15.2-minute step** (mean of 176 steps without a test pass).

| Part of the step | Seconds | Share |
|---|---|---|
| Generating games | 593 | 65% |
| Learning from the scored games | 219 | 24% |
| Re-scoring each game with the current and frozen copies (2 × 44 s) | 88 | 10% |
| Copying the updated model to the players | 2.1 | 0.2% |

The 593 s includes waiting for the slowest game to finish its quota.

**Network.**
- **The risk:** if traffic falls back from Polaris's Slingshot network to ordinary TCP, the logs
  look normal. So each job fails unless the communication library reports the Slingshot
  provider.
- **Measured:** 36 GB/s all-reduce bandwidth across 2 nodes. At the measured 21 GB/s
  point-to-point rate, an 8 GB model copy takes about 0.4 s; that's an estimate, not a timed
  copy. The whole copy step took 2.1 s.

**Memory.**
- **The fix for running out of memory:** at 32,768 tokens per sequence the learner ran out of
  memory until it took one sequence at a time.
- **The padding fix:** about 80% of the training tokens were padding. Trimming it cut learning
  from 253 s to 68 s per step, and the whole step from 12.2 to 7.5 min, in the 3-node run.

**Idle time.**
- **Busy share:** play and learning take turns, so in the final attempt (steps 105–199) the GPUs
  were busy 29% of the time on the learner nodes and 54% on the game nodes. "Busy" is the GPU
  utilization reported by `nvidia-smi`, not how much of each chip's compute was used.
- **Waiting on the slowest game:** 221 of 230 collections waited 5 minutes or more for it, and
  TicTacToe was the last game in 207 of them.

**Process cap, if someone asks.** Polaris allows 4,096 processes per node, and the game workers
peaked at 3,350. Shrinking a thread pool to stay under the cap caused a deadlock once 256 or more
requests were in flight (job 7667241). Giving responses their own thread group fixed it (job
7667360).

**Takeaway.** The cost of this training is generating games, not communication. We hypothesize
that overlapping play with learning, not a faster network, is the next speed-up.

Sources: `MARSHAL/polaris_multinode_slingshot.md`; `MARSHAL/marshal_r6_results.md` R9.

---

## Slide 7: Our 58 hours of compute took 10.6 days from submission; restarts kept 85% of the node time we held

**The bar** (MARSHAL's three-game run, job 7741511, on the `preemptable` queue).

| Segment | Hours |
|---|---|
| Waiting for the first start (submitted 2026-10-13 05:08, started 10-19 18:49) | 157.7 |
| Between restarts | 30.2 |
| Running, but the work was not kept | 9.8 |
| Compute kept (training and test passes) | 57.7 |
| **Total, submission to finish on 10-23 20:31** | **255.4 (10.6 days)** |

The middle two rows are our arithmetic: 97.7 h from first start to finish, of which the job held
nodes for 67.5 h, of which 57.7 h was kept.

**The tiles.**
- **34 starts.** The run was stopped 33 times. On one night it was stopped 16 times in a row,
  each after 1 to 22 minutes, most before finishing a step. No hardware, memory or network
  errors were logged. Urgent short jobs reclaiming the nodes fits the pattern (UNVERIFIED: the
  scheduler's records aren't readable from a user account).
- **5 steps.** A complete checkpoint was saved every 5 steps, 40 saves in all. The job was
  submitted as rerunnable, so the scheduler put it back in the queue after each stop.
  "Complete" means every GPU's random-number state and optimizer state were saved too. The
  final training history matched the attempts that produced it in all 64,537 logged values.
- **85%.** Of the 270 node-hours the job held, 85% went into steps that were kept.
- **8 replays.** Decrypto's game runner has no resume. One seed's job was stopped 8 times and
  restarted from scratch each time; it finished on its 9th start, in about 7 hours.

**For comparison.** The one-game run (R5, 3 nodes) finished in a single 40.2-hour allocation,
about 120 node-hours, with no stops.

**Takeaway.** Time to result was set by the queue and by restarts, not by compute speed.

Sources: `MARSHAL/marshal_r6_results.md` R10 and Table 6; the Decrypto job ledger
(jobs 7668935–7668937).

---

## Slide 8: Partly: models decoded hints and learned rules, but did not reliably anticipate an opponent

**The answer, in two lines.**
- **Decrypto:** partners understood each other, but the opponent cracked 95% of codes. On the
  slide, "2.5% misreads" is the share of games lost to two misreads. Per round, teams misread
  their own code 0.9% (Claude) to 9.7% (Qwen3-4B) of the time.
- **MARSHAL:** self-play fixed illegal moves, strategy improved in one game, and Leduc poker
  didn't improve.

**The five HPC lessons, with their evidence.**
1. **Size concurrency to the slowest model's memory:** 160 lost games became 0 (slide 4).
2. **Put every part of a pipeline in one job:** separate server jobs died idle while the game
   job waited in the queue (slide 4).
3. **Fail fast when traffic leaves the fast network:** the fallback to TCP is silent (slide 6).
4. **Checkpoint often, and let the scheduler requeue:** 85% of held node time was kept across
   34 starts (slide 7).
5. **Measure the step:** generating games took 65%, copying the model 0.2% (slide 6).

**The next experiment, and where it stands.**
- **Design:** the two trained checkpoints play Decrypto in place of the untrained Qwen3-4B. Each
  plays the same 7 pairings the untrained model played: as a team against Qwen2.5-72B, Qwen3-8B,
  Claude and itself, and as the opponent of the other three teams. Same seeds and game
  instances, 672 games each.
- **Settings:** identical to the untrained model's (2,000-token thinking cap, same sampling), so
  the existing untrained games are the comparison.
- **Status:** submitted 2026-11-18 as jobs 7860393 (R6 checkpoint 149) and 7860394 (R5
  checkpoint 199) on `preemptable`; queued.
- **So far:** only a 4-game check, with checkpoint 199 playing all three roles itself. It
  shows the trained model runs in Decrypto, not whether it reasons better.

**What would count as transfer.**
- **As the opponent:** the untrained Qwen3-4B was the weakest code-cracker, at 87.2% of games
  against 97.1% for the other three. A trained model that cracks more codes would show
  transfer.
- **As a team:** it misread its own hints most often. Fewer misreads, or longer games, would
  count too.

**What we expect.** We hypothesize little gain, because MARSHAL mainly taught the model to
answer cleanly. Any gain may come from cleaner answers rather than better reasoning.

**If someone asks: why would cleaner answers score better without better reasoning?** The game
can only score what it can read, and it treats an unreadable answer as a wrong one.
- **Defaults are always wrong.** After 10 failed attempts the game fills in a default: hints
  "no_hint" for an encoder, or the guess "0-0-0" for a decoder. Both lose the round.
- **Cut-off thinking.** A Qwen3 model stopped at 2,000 tokens must answer from unfinished
  reasoning. A model trained to think in shorter turns gets cut off less.
- **An analogy:** a student who knows the answers but sometimes writes them in the wrong box,
  or illegibly, is marked wrong even when they knew the answer. Fixing the handwriting adds no
  knowledge, but the score rises, because answers they already knew now count. This works only
  if those answers are usually right. In Decrypto they are: when the model answers properly,
  its team's guess is usually right.

**How we'll tell the two apart** (from the round-level audit, job 7860225).
- **Unreadable answers barely happened.** No decoder ever fell back to "0-0-0", and encoders
  fell back to "no_hint" 9 times in 1,536 games. Of Qwen3-4B's 152 misread rounds, 2 came from a
  default answer.
- **Cut-off thinking happened a lot, and mattered.** Cut-off Qwen3-4B decoders misread 45.3% of
  rounds, against 6.9% when thinking finished. Cut-off decoder turns account for 53 of Qwen3-8B's
  83 misread rounds and 53 of Qwen3-4B's 152.
- **So for the trained models** we will report cut-off rates for both arms, compare only turns
  where thinking finished, and treat a change in *interceptions* as the sign of better
  reasoning about the opponent. Misreads measure whether a model understands *itself*.

**Limits.**
- One base model, and one training seed per run; no ablations.
- The comparison numbers from the papers and their human games are UNVERIFIED.
- Claude was not run with the study's settings.
- The late windows of the three-game run were chosen after seeing the data.
- The cut-off finding is an association. Hint difficulty isn't controlled, and the replay test
  hasn't been run.
