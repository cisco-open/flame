# Real / Sim Parity — Methodical Causal Ladder

Living doc for the async_cifar10 parity checker. Kept in sync with
`scripts/parity/checks.py` (checks), `scripts/parity/report.py` (stage grouping +
verdict), and the pytest suite. **ONE `## Status` section, updated in place.**

**Comparator — give two run dirs, get a report JSON:**
```bash
cd lib/python/examples/async_cifar10
PYTHONIOENCODING=utf-8 python scripts/parity_check.py \
  --real experiments/<real_run_dir> --sim experiments/<sim_run_dir> \
  --agg-goal 10 --budget-s <runtime_s> --json-out experiments/parity_<baseline>_<tag>.json
# batch: --batch --experiments-dir experiments --baselines felix oort refl feddance
```
`--budget-s` = the run's `--runtime-s`; `--lenient` demotes DIST fails to warnings.
Per-run plots: `python ../../../scripts/analysis/analyze_run.py <run>/telemetry`.

**Launch runs** (node-agnostic): `bash scripts/debug_run.sh --baselines 'oort refl'
--runtime-s 3600 --mode both`. Reads `expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_parity.yaml`
(each baseline × sim/real), seeds real+sim identically (`SEED=1234`), applies per-baseline
sim fixes. Split across machines via `--baselines`.

**Tests** (no cluster; under lib/python with `dg_flame`): `pytest tests/mode/
tests/selector/test_oort_selector.py tests/sim/ examples/async_cifar10/scripts/parity/`.
Last green (Jun 24): selector+mode+sim+parity **351 pass / 7 skip** (incl. `TestPacerFidelity`).

---

## Workflow policy: minimize time & runs to parity

1. **Run real only when the real path changes.** Sim-only changes validate against the
   stored real dir. A real-path fix → re-run both (or real-only if sim code is unchanged).
2. **Over-instrument telemetry deliberately** — cheap to log, expensive to re-run for.
   (per-round `inflight_residence`, `SIM_CLOCK_DIAG` past-dating counters, per-commit
   `commit_gap` localized roots from stored runs.)
3. **Root-cause per baseline, scope the fix to its blast radius.** Common cause → fix once.
   A fix that could perturb another baseline → serialize (one baseline per run round).
4. **Shortest run that exhibits the issue** (table below). Reserve long runs for C1/C2.
5. **Crisp comments (≤1 sentence); context-free names** (Naming discipline below).

### Run-length budget (state min duration up front, keyed here; never default to 3–4h)
| validating | min run | why |
|---|---|---|
| telemetry field present / instrument sane | **5–10 min** | a few hundred commits populate any per-commit field |
| one MECHANISM rung (gate hold, `commit_visibility`, `residence`, `selection_detail`) | **45 min** | every mechanism fires; per-commit dists stabilize |
| compounding clock/advance residual (`K2`/`K3b`, past-dating) | **90 min – 2 h** | round-count-compounding drift needs the rounds |
| low-frequency eligibility/round-count drift (refl `K2`) | **3 h** | only surfaced at 3h despite a clean 45min pass |
| `C1`/`C2` convergence sign-off | **full (3–4 h+)** | terminal-state + curve parity only |

Smoke (5 min) before any multi-hour run. One mechanism per run round when a fix could
perturb another baseline.

---

## Status (Jun 24, post-§S.pacer 1.5h validation)

**§S.pacer VALIDATED:** the faithful two-branch pacer tracks across modes — `round_threshold` is
now logged and the monotonic-ratchet-to-100 is gone. ⚠ These are 1.5h runs (prior scoreboard was
3h); K2 is run-length-sensitive for oort+refl, so read deltas as mechanism evidence, not final
scores. Post-commit b7b36bd2.

| baseline | score | state / lowest rung | run dirs |
|---|---|---|---|
| **felix** | **46/46** ✅ (=3h) | CLOSED — §S.pacer re-validated, penalty now active (was inert under the ratchet), no regression (K8 rel 0.026, U2 0.0259, acc_diff 0.0215). §3.drain+§3.resid hold. C1/C2 LOWC only because 1.5h<7200s (3h sign-off done). | `…183825…real` / `…173614…sim` |
| **oort** | **42/46** ⬆40 | §S.pacer CLOSED the advance family (K3b 0.131→**0.059 PASS**, K2 0.117→0.058 marginal). New lowest rung = **Sd**: sim saturates `round_threshold`→100/penalty-off in 43.4% of Q4 rounds vs real 14.0% → ~0.6s-slower mix. Root = the **exploited-utility SIGNAL** feeding the (faithful) pacer is smoother in sim's back half (relax-branch over-fires) — NOT the pacer port, NOT §S.dur (A2c/eligible_speed/Sx/residence PASS). Minor: T2 training_budget tail (sim p99 36/real 31, mean+pool match). | `…203522…real` / `…202408…sim` |
| **feddance** | **43/44** (3h Jun 23) | CLOSED — U6 §6.u6 PASS; K2/K3b/A2c/K8/U2/C1 PASS. Sole FAIL = C2 loss (0.1716 vs 0.15, emergent, 8 eval rounds). | `…033223…real` / `…031608…sim` |
| **refl** | **42/46** ⬇44 | RUN-LENGTH artifact, not §S.pacer (override removal verified inert: penalty binding 0/0, removed override was already two-branch faithful). K2 0.047(3h)→0.084(1.5h) — the §S.dur-closed selection-mix residual re-surfacing at short, early-training-dominated length. Needs a 3h rerun to re-confirm closed. | `…223255…real` / `…220847…sim` |

JSONs: `parity_{felix,oort,refl}_20260624_5400` (1.5h); 3h `parity_*_20260624_3h`,
`parity_feddance_20260623_3h`.

### Next steps
1. **oort — root = the exploited-utility SIGNAL feeding the pacer (Sd Q4 saturation).** Confirm at 3h
   first (K2 0.058 marginal, lives in the back half). Then instrument the pacer input
   (`exploitation_util_history` per-round `relative_change`, binned by quartile) in both modes to find
   why sim goes flat more (candidate: per-trainer utility tail, F1-3 max_KS 0.87). Do NOT tune
   `pacer_delta`/`pacer_step` or touch the §S.dur input. Re-check T2 training_budget at 3h.
2. **refl — re-confirm at 3h (no code action).** If K2 still fails at 3h, diff OLD override vs base on
   identical stored inputs.
3. **feddance / felix — closed**, no rerun needed.
4. **Run plan: oort + refl at 3h next.**

### Roadmap: lock mechanism parity on ALL baselines BEFORE the perf pass (Jun 21)
Get 46/46 on oort+refl+feddance first, then the sim perf pass — do not interleave. The
checker reads over-instrumented per-commit telemetry; stripping/gating it before A2c roots
close removes the instrumentation needed to root-cause them. A perf change can perturb all
four baselines at once, so it must land on a clean fully-parity base. **Perf pass levers
(deferred):** flag-gate diagnostic logging behind a `simDiag` switch (default OFF), trim
per-commit JSONL volume; guard rail — every perf commit re-runs the 90-min parity (all
baselines) and must hold 46/46.

### Settled roots
| baseline | root |
|---|---|
| **felix** | **46/46 at full 3h incl. C1/C2 (Jun 24) — RE-VALIDATE after §S.pacer.** Eval-stale-`sct` ✅; commit-side ingestion (`recv_fifo` stranding) ✅ `simSctOrderedDrain` (§3.drain); overlapping re-dispatch ✅ `_sim_hold_busy_slots`/`simInflightResidence` (§3.resid). Speed model exonerated. **C1/C2 sign-off DONE** (3h: acc_diff 0.017, loss_diff 0.020, K8/U2 rel 0.023). §S.dur anchor no regression. **§S.pacer (Jun 24) touched felix's `AsyncOortSelector.pacer`** — it had the same unfaithful raise-only ratchet AND fired on the eval hand off a stale round; both fixed (faithful two-branch + train-gate). felix's penalty was inert *because* of the ratchet; now active → re-validate 46/46 on the next run (a shift would mean the prior pass leaned on the inert penalty). |
| **oort** | **MECHANISM ROOTS CLOSED Jun 22 (46/46); faithful temporal (§S.temporal) landed; 3h Jun 23 PM → 43/46, single root K2.** **(1) Stale-property recording** — real dropped a stale-returning trainer's speed/utility (`continue` before `_handle_weights_msg`) → Oort treated slow trainers as unexplored and re-picked forever; fix `_record_returned_trainer_props`. **(2) K3b stale read-wait inflation** — fix (1) recorded stale durations as `recv−dispatch`, bundling **aggregator read-wait** (up to 1.65×D, a server artifact); fix: real records **client task-train duration = `WALL_SEND_TS − dispatch`**. **(3) Temporal fidelity (Jun 23):** new faithful UCB temporal default applied symmetrically — alive & matched (3h Sx KS 0.076, 0.045/0.044), A2c PASS 0.074, K3b PASS 0.054. **3h verdict:** the 90min Sr/K8 marginals were small-N — **Sr RESOLVED** (rel 0.25 vs 0.3) and **K8 is DOWNSTREAM** of the lone surviving root **K2** rel 0.052. **(4) K2 ROOT-CAUSED + FIXED Jun 23 PM (§S.dur):** K2 = sim selects a ~0.77s-faster mix (committed per-round max-of-K 9.69 vs real 10.09) because the selector's observed-duration INPUT still diverged — fix (2) stripped the recv-side read-wait but real's `WALL_SEND − dispatch` still carried the **dispatch→recv delivery lag** (`WALL_RECV − dispatch`, +8s for D≥20 stragglers held one-in-flight), inflating `pref` (10.7 vs sim 7.0) so real under-penalized slow clients. Fix: real duration = `WALL_SEND − WALL_RECV` (intrinsic D, both client stamps = sim's `max(gpu,D)`). Guards `TestStaleTrainerPropsRecorded`. C1/C2 PASS at 3h. **(5) §S.dur VALIDATED Jun 24, but it EXPOSED a deeper SIM-side root → 43→40/46.** CLIENT_DUR_STRIP confirms real input clean (intrinsic_s≈D, delivery_lag≈0.02s — in THIS run the lag was already negligible, so the strip mostly re-anchored to WALL_SEND−WALL_RECV). With real cleaned, sim's `pref` now sits HIGHER (mean 10.14 vs 8.58) → sim binds the speed penalty 73.6% vs real 94.1% → slower max-of-K (10.5 vs 8.99) → K3b rel 0.131 + K2 rel 0.117. **New root = the DYNAMIC `round_threshold` PACER (§S.pacer)**, back-half-compounding (Q4 sim pref 19.49/real 15.19, sim disables penalty in 31 late rounds, real 0). A2c/residence/Sx PASS, so NOT the duration input. **(6) §S.pacer ROOT-CAUSED + FIXED Jun 24:** flame's base `OortSelector.pacer()` was an UNFAITHFUL port — it raised `round_threshold` on ANY utility dip (`last > curr`) with NO decrease branch (reference: FLAT `|Δ|≤0.1·last`→raise, SHARP `|Δ|≥5·last`→lower), so it ratcheted monotonically to 100 and was noise-sensitive → sim/real diverged & compounded. Fixed = faithful port (both branches, 0.1/5× bands, current-round keyed); redundant refl override removed; `round_threshold` added to telemetry. Guard `TestPacerFidelity`. **(7) §S.pacer VALIDATED Jun 24 (1.5h, 40→42): advance family CLOSED, residual one rung deeper.** `round_threshold` tracks (sim Q1–Q4 11.6/36/63/93, real 10.4/30/54/86, NO ratchet, NO compounding); **K3b 0.131→0.059 PASS**, per_round_advance PASS, **K2 0.117→0.058** (marginal). New lowest rung = **Sd**: even with the faithful two-branch pacer, sim's threshold over-climbs in the back half — **sim saturates to 100 (penalty off, `pref=99999`) in 43.4% of Q4 rounds vs real 14.0%** (Q1–Q3 both 0%) → sim binds 0.411 vs real 0.686 → ~0.6s-slower mix → K2 0.058. **New root = the exploited-utility SIGNAL feeding the pacer is SMOOTHER in sim's back half** (the relax-branch fires on FLAT `exploitation_util_history` change; sim goes flat more) — NOT the pacer port, NOT the §S.dur input (A2c/eligible_speed clean). This is the §S.pacer prediction realized: a faithful controller fed MATCHED utility tracks; since sim doesn't track, the utility signal differs (F1-3 per-trainer max_KS 0.87). NEXT: confirm at 3h, then instrument per-round pacer `relative_change` by quartile in both modes. Minor new **T2 training_budget** control (sim p99 36 vs real 31; mean+pool match → tail/run-length). |
| **refl** | Shared `oort/top_aggregator` stale-property + `WALL_SEND_TS` fix landed CLEAN. A2-residence root CONVERGED (Sr PASS 3.851/3.772). **K3b root = selection disparity from a DEAD UCB temporal term — FIX CONFIRMED Jun 23 PM (40→42/46).** `refl_oort.select()` let the term divide by a None `time_stamp` → temporal 0 (0/7513 both modes) → refl scored by stat_util alone (speed-uniform). **FIXED FAITHFULLY (§S.temporal):** `PROP_LAST_RETURNED_ROUND` = reference's last-RECEIPT round (`self.epoch`), stamped at receipt + registration-init. **VALIDATED & STABLE at 3h:** temporal term ALIVE & matched (Sx KS 0.119, real 0.026/sim 0.024), **K3b PASS** rel 0.095, **A2c PASS**, **Sr PASS** 3.72/3.78. **3h verdict (40/46):** surviving roots are **K2** rel 0.093 (the constant per-round advance residual ~0.28s/rd, quartile-flat → constant selection-mix bias, shared with oort — see Next-steps §0; its measurement tightened with rounds to expose it over K2's 0.05) and **A2** num_candidates KS-artifact (num_eligible MEANS match 251.7/249.8). S2/K8/U2 downstream. So the temporal fix was BOTH a fidelity fix AND closed K3b/A2c. **K2 ROOT-CAUSED + FIXED Jun 23 PM (§S.dur, shared with oort):** the residual was NOT a clock/overhead bug but the same selector-input delivery-lag (real `WALL_SEND − dispatch` carried `WALL_RECV − dispatch`, inflating `pref` 10.7 vs sim 7.0 → real picks a slower mix → per-round max-of-K 15.40 vs sim 14.08 → +0.28s/rd). Fix: real duration = `WALL_SEND − WALL_RECV` (intrinsic D). **§S.dur VALIDATED Jun 24 (40→43/46): throughput family CLOSED** — K2 PASS rel 0.047 (sim 2.64/real 2.77), K3b PASS 0.048, A2c PASS, Sr PASS 0.024. refl is IMMUNE to §S.pacer (round_threshold=30 → speed penalty never binds, Sd 0.0/0.0), so the oort regression doesn't touch it. **S2 ROOT-CAUSED + RESOLVED Jun 24:** `matched_count_ks` 0.464 was pure STOCHASTIC core-IDENTITY — both modes build a ~120-trainer persistent core, SPEED-MATCHED (real-only D̄ 9.8 / sim-only 9.2; participation-weighted 8.20/8.26) with only 63 of ~120 individuals shared (weighted-exploit draw locking different members via rich-get-richer); NO speed/utility-class bias (A2c/K8 pass), speed-class participation matches (TVD 0.026 vs per-second 0.187 noise). Reclassified S2 to enforce participation BY SPEED CLASS for stochastic selectors (§5), identity demoted to diagnostic → 43→44. Remaining: **A2 eligibility** KS point-mass artifact (means 251.1/248.8 match — checker read, not a sim/real lever), **C2 loss** emergent. Guards `TestREFLTemporalFidelity`/`TestTemporalUncertaintyFidelity`/`TestStaleTrainerPropsRecorded`. **§S.pacer (Jun 24) removed the redundant `REFLOortSelector.pacer` override — VERIFIED INERT for refl** (penalty binding 0/0, no round_threshold in selection telemetry; the removed override was already two-branch faithful, so inheriting the base changes nothing for refl selection). **1.5h post-pacer run (44→42) is a RUN-LENGTH artifact, NOT the removal:** K2 0.047(3h)→0.084(1.5h), sim faster mix (selected_mean 11.28 vs real 11.62, advance 2.51/2.74) — the §S.dur-closed selection-mix residual re-surfacing because 1.5h is early-training-dominated. Re-confirm at 3h; only if K2 still fails at 3h, diff OLD override vs base on identical stored inputs. |
| **feddance** | **CLOSED 43/44 (3h Jun 23).** `selection_bias`/`feddance_U` closed via inherited syncfl `WALL_SEND_TS` duration fix. **U6 §6.u6 barrier-anchor CONFIRMED on fresh real**: U6 PASS real_mean 14.39 ≈ sim 14.55 (KS 0.115, mean_diff 0.164s). **K2 PASS** (28.04 vs 27.63 s/rd) — the 1.5h 0.052-vs-0.05 was ~1.4σ noise, tightened with rounds, NO code lever was needed (correctly resisted the scalar-overhead / stat_utility-chase dead ends). K3b/A2c/K8/U2/C1 PASS. Lone residual **C2 loss** 0.1716 vs 0.15 = emergent eval-curve on 8 points; K8 terminal-state + C1 acc both PASS, so run-end state matches. No mechanism gap remains. |

---

## Durable lessons (update in place, don't append)

- **Real is the reference, but VERIFY real is correct first — a real↔sim mechanism gap has
  TWO fix directions.** Parity ≠ blindly tuning sim to real. Oort case: sim recorded observed
  durations for 298 trainers (31% ≥15s), real only 201 (3%). (a) *Recording-or-not:* real
  `continue`d before `_handle_weights_msg` so a stale-returning trainer's speed/utility were
  never recorded → Oort reads `PROP_STAT_UTILITY is None` as *unexplored* (oort.py:472-478)
  and re-explores forever. Fix is on the **real** path (record props for stale-but-returned
  updates) + complete sim's utility recording — NOT a sim-side discard. (b) *Value:* the real
  path then recorded `recv_ts − dispatch`, which for a stale straggler bundles in the
  **aggregator read-wait** (finished update sits unread until a later round drains the buffer)
  — a server artifact, not client speed, inflating slow trainers up to 1.65×D so real
  over-avoided them. **Invariant: the selector speed signal `PROP_CLIENT_TASK_TRAIN_DURATION`
  is the CLIENT's INTRINSIC task-train duration — anchor on the TWO CLIENT stamps
  (`WALL_SEND_TS − WALL_RECV_TS`), NOT any aggregator-anchored span.** Same concept both modes
  (sim's `SIM_CLIENT_TASK_TRAIN_DURATION_S = max(gpu,D)`). Single-sourced in
  `_real_client_task_train_duration`. Proof it was the scorer not residence:
  `scripts/oort_residence_discriminator.py`. See [[project_oort_a2c_root]].
- **A server-anchored duration has TWO waits to strip, and stripping only one leaves a residual
  that surfaces as the next rung (oort+refl K2, Jun 23 §S.dur).** `WALL_SEND − dispatch` excludes
  the recv-side read-wait but STILL carries the **dispatch→recv delivery lag** (`WALL_RECV −
  dispatch`): the agg stamps `dispatch` at selection, but a slow client held one-in-flight
  receives the weights later, so the lag is large for stragglers (+8s for D≥20). That inflated
  the selector's `round_preferred_duration` percentile (real 10.7 vs sim 7.0), so real
  under-penalized slow clients → committed a ~0.77s-slower mix → per-round max-of-K +0.4s → +0.5
  s/rd advance → K2. **Tells it's the selector INPUT not a clock/overhead bug:** (1) K3b/K3a on
  COMMITTED updates pass (fresh delivery lag ~0.07s); the gap is in the CANDIDATE-POOL durations
  the scorer reads, dominated by stale-straggler recordings. (2) `obs ≈ static D` for committed
  but `obs = static + 8` for the candidate pool's slow tail. (3) the trainer's OWN stamps prove
  intrinsic = D exactly (`wall_send − wall_recv == budget`; real GPU ~0.01s, device sleeps to D),
  so the +8s is provably server-side. Fix anchors BOTH ends on client stamps. **General rule: a
  duration fed to a SELECTOR must be the client-intrinsic span; any aggregator-stamped endpoint
  smuggles in a server wait that biases the speed penalty.** Diagnose by mapping committed +
  candidate-pool durations to STATIC D (registry) and reading `obs − static` per speed bucket —
  it localizes whether the divergence is the draw (input) vs commit/residence.
- **Cleaning the REFERENCE (real) input can FLIP a divergence onto the SIM side and expose a
  deeper root (oort §S.dur→§S.pacer, Jun 24).** §S.dur stripped real's selector-duration delivery
  lag (correct — CLIENT_DUR_STRIP shows `intrinsic_s≈D`, `lag≈0.02s`). For refl that CLOSED K2.
  For oort it lowered real `pref` 10→8.58 and revealed that SIM's `pref` (10.14) was the divergent
  side all along — sim under-binds the speed penalty (73.6% vs 94.1%) → slower mix → K2/K3b fail.
  **Lesson:** when a real-path fix over- or fully-corrects, re-run BOTH and check the metric didn't
  just cross zero (sim was faster, now sim is slower). The new lowest rung is a different mechanism,
  not "the fix regressed." Localize it fresh: identical pool + identical selected-MEAN (A2c) +
  divergent per-round `pref` percentile ⇒ the divergence is in whatever SETS the percentile (here
  the dynamic `round_threshold` pacer), not the durations feeding it.
- **A DYNAMIC feedback-controller knob that diverges and COMPOUNDS is usually an UNFAITHFUL PORT,
  not a mode-asymmetry — diff it against the reference line-by-line BEFORE reaching for a
  re-parameterization (oort §S.pacer, Jun 24).** flame's `pacer()` raised `round_threshold` on ANY
  utility dip (`last > curr`) with NO decrease branch; the reference makes TWO symmetric moves —
  FLAT (`|Δ|≤0.1·last`)→raise, SHARP (`|Δ|≥5·last`)→lower. The one-branch monotonic ratchet was
  hypersensitive to per-round noise, so sim/real `round_threshold` drifted to different levels and,
  never recovering, the gap GREW in the back half (Q4 sim pref 19.49 vs real 15.19; sim `pref=99999`
  in 31 late rounds, real never). **Tell it's a port bug not the round-indexing:** the reference is
  ALSO per-round, but self-corrects — a faithful symmetric controller fed matched utility
  distributions tracks across modes. So the fix is FIDELITY (port both branches), not the §3.async
  "re-parameterize cadence by wall-time" generalization (which I floated first and then SUPERSEDED).
  Bin the knob by run-quartile to see the ratchet; instrument it directly (`round_threshold` was
  only inferable from `pref`); never tune `pacer_delta`/`pacer_step` blind. **When you fix a ported
  knob, check EVERY copy against EVERY reference the baselines map to, and every SELECTOR HAND:**
  refl has its own fork (`third_party/REFL`, byte-identical pacer here — confirm, don't assume),
  and felix's `AsyncOortSelector` carried the same bug PLUS fired the (training-only) pacer on its
  EVAL hand off a stale round. A shared-concept knob copied into 3 classes drifts 3 ways.
  **VALIDATED Jun 24 (1.5h): the fix did exactly what the prediction said, and the residual landed
  on the INPUT.** Post-fix `round_threshold` tracks across modes (no ratchet) and K3b closed — but
  oort still over-relaxes in the back half (sim saturates to 100/penalty-off in 43.4% of Q4 vs real
  14.0%). A FAITHFUL controller that STILL diverges localizes its own INPUT: the relax-branch fires
  on FLAT `exploitation_util_history`, so the surviving divergence is the per-round exploited-utility
  SIGNAL being smoother in sim, not the controller. **Once you've confirmed a ported knob is
  faithful, stop touching the knob — instrument its INPUT (here the per-round `relative_change`) by
  quartile and chase whatever feeds it.** A faithful controller is a clean lens onto an upstream
  signal divergence you couldn't see while the controller itself was buggy.
- **A large per-trainer participation KS (S2) on a STOCHASTIC selector is usually core-IDENTITY,
  not a mix bias — separate identity from policy by bucketing on an INTRINSIC attribute (refl
  Jun 24).** refl's `matched_count_ks` 0.464 looked like a selection bias, but the two modes build
  ~120-trainer cores that are SPEED-MATCHED (real-only D̄ 9.8 / sim-only 9.2; participation-weighted
  8.20/8.26) with only 63 shared individuals — the weighted-exploit draw locks DIFFERENT members in
  via rich-get-richer from tiny round-1 eligibility differences. **Granularity is the discriminator:**
  at `speed_class` the participation shares match (TVD 0.026); per-SECOND buckets re-expose the same
  identity noise (TVD 0.187, sign-alternating Δ). So enforce participation BY SPEED CLASS (policy),
  demote per-trainer identity to diagnostic (§5). Confirm it's identity-not-bias with A2c + K8 +
  speed-matched mode-specific cores BEFORE reclassifying — don't suppress a real mix bias.
- **A borderline EXACT rung (K3b/K2) swaps in as the root once an upstream DIST rung (A2/Sr)
  converges with run length (refl Jun 23).** At 1.5h refl's root was A2/residence (3.63 vs 3.86)
  and K3b PASSED; at 3h residence converged (3.645 vs 3.72, Sr PASS) but K3b FAILED (rel 0.124).
  Nothing in dynamics changed — the residence gap was run-length-sensitive AND the constant ~0.4s/
  round advance gap was always there, just under the 0.1 bar until more rounds tightened the
  estimate. **Tell a constant mechanism gap from compounding feedback: bin per-round advance by
  run-fraction.** Flat gap across all bins (sim ~2.77 / real ~3.2 every bin) = a constant
  selection-mix bias, NOT a §3.async round-indexed loop (which would GROW). Don't read the K3b
  emergence as a regression; it's the next rung down surfacing.
- **An INERT score term is a tell for a DEAD mechanism, not "the selector ignores it" — verify the
  term is even being fed inputs (refl Jun 23).** refl K3b looked like an emergent stat-utility×speed
  correlation: `system_util`=1.0 and `temporal`=0 for EVERY candidate both modes, so the selector
  seemed "purely `believed_I`-driven." The right read of `temporal`=0/7513 was NOT "UCB is naturally
  small" but **"the UCB term is structurally DEAD"** — `refl_oort.select()` overrode `oort.select()`
  and dropped the `_record_last_selected_round` stamp, so `PROP_LAST_SELECTED_ROUND`=None →
  `oort_temporal_uncertainty` returns 0 forever. The reference Oort/REFL temporal bonus up-weights
  under-selected (slower-returning) trainers; with it dead, refl selected speed-uniformly (`sel/elig`
  0.092 FLAT) while real-side parity expected the speed-rising pattern. **Lesson:** when a score term
  is byte-zero across a whole run, check the INPUT property it reads is actually populated (and that
  an overriding subclass didn't drop the parent's stamp), and diff flame against the `third_party/`
  reference — don't conclude "emergent, no lever." Confirm selection-by-class with a SINGLE intrinsic
  map applied to both modes (a per-mode map masked the flat-vs-rising split as "neutral +0.001").
- **`A2 num_eligible` can FAIL (KS) while `S3/4 in_flight` PASSES — read it as the same gap at
  two tolerances (refl Jun 22).** With an all-available trace (`A1 UNKNOWN≈300`), eligible =
  `candidates − in_flight_hold`, so a small in-flight gap (60.2 vs 63.15, rel 0.047 — under
  S3/4's 0.15 bar) lands directly on eligible (252.7 vs 249.8) where A2's tight KS≤0.2 binds.
  Don't chase A2 as a separate eligibility bug; walk to the in-flight/residence channel
  (`residence_rounds` 3.63 vs 3.86 localized the §4.5 release-timing root).
- **Decompose a net selection-rate gap into channels before fixing it.**
  `sel_rate = eligible_fraction × P(sel|eligible)`. If `eligible_fraction` matches sim/real
  (the menu is identical) the gap is the **scorer** (P(sel|elig)), not residence/eligibility.
  This invalidated the "sim under-holds slow trainers" (residence-leak) hypothesis for oort:
  slow-trainer in-flight residence was dead-equal (0.00444 vs 0.00438). The `in_flight_after`
  1.68-vs-1.33 aggregate that *suggested* residence was a RED HERRING — it decomposes to FAST
  trainers (real carries 4× more, slow ≈equal), a downstream consequence of real selecting
  fast more, read backwards.
- **A check consuming `agg_rounds` must split train vs eval (Jun 20).** Eval commits emit
  `event=agg_round` (`task_to_perform="eval"`) with no `agg_goal_count`, don't advance the
  clock. Letting them into the shared stream broke K1 monotone (eval's higher `vclock_now`
  faked 2656 backward steps), U3 staleness (24,980 eval `staleness=0` deflated 15.17→9.44),
  U1/U5. **Rule:** `load_agg_jsonl` partitions eval into `eval_commits`; `agg_rounds` is
  train-only; only U6/U6e read combined. Validate monotone invariants in true `ts` order.
- **`Sdet` triage.** `eligible_match≈0` + matching aggregates = stochastic-class, PASS;
  `eligible_match≈0` + diverging clock = genuine, fix clock; `eligible_match` high but
  `decision_match≈0` = the **values** diverge, not the set.
- **`participation`/S2 vs per-round Jaccard.** Round-to-round draw mismatch (`Sdet`/`S1`) is
  expected for a stochastic selector. A **systematic per-trainer skew over the full run** (S2
  `matched_count_ks` large, `max_diff` not averaging out) is a real bias (how refl's
  `participation` finding was distinguished from noise).
- **`U6 commit_visibility` KS is a false-positive on a sub-ms point mass.** When both modes
  commit immediately (real_mean 0.004s, sim 0.001s) KS→1.0 is signal-free; read `mean_diff`
  (3ms ≪ 2.0s bar). A real divergence (felix past-dating) shows a large `mean_diff` (14.8s).
- **`U6` LARGE `mean_diff` on a STRICT SYNC BARRIER was a REAL-telemetry flaw, fixed on the real
  path — NOT a sim bug, NOT a checker gate (feddance, Jun 22).** sim 15.5s vs real 0.02s. Sim
  `vclock−sct` is correct (barrier commits all K at `max(sct)`; slowest lag=0, early finishers
  0→52s). The bug: real `_update_visibility_lag` took `committed=datetime.now()` **per-message in
  the recv loop**, measuring arrival→ingestion (~0.02s), NOT the barrier wait — even though
  updates physically arrive SPREAD (`[MSG_ARRIVAL]` 33→48s, `queue_depth=0`). Don't reach for a
  checker WARN-gate or "real is blind" — real HAS the spread; **recompute it from stored raw
  logs first**: `lag_i = max_dur − dur_i`, `dur = WALL_SEND_TS − dispatch` gave 15.65 ≈ sim 15.48,
  proving the quantity exists and the metric (not the dynamics) was wrong. Fix = barrier-anchor
  the real path (§6.u6); only the sync-barrier baseline (feddance/fedavg) needs it — a STREAMING
  aggregator (oort/refl) commits each update at its own `sct`, so per-message is already correct.
  **Tell it's a metric flaw not past-dating:** sim per-round MIN lag ≈0 (past-dating shifts the
  whole dist up) AND `U3 staleness` 0/0 matched.
- **`P3 mean_overhead` is wall-capture, not a speed-model bug**, when sub-second,
  opposite-sign across baselines, and `grid_KS`/`training_delay_s` match — trust `P3` only
  when `grid_KS` also fails.
- **`K3b` ≠ "missing overhead" when `implied_per_commit_overhead_s`≈0.** Its residual
  `real_advance − sim_advance` is also moved by the per-round **max-of-K speed** order
  statistic; if implied overhead is sub-0.05s yet K3b fails, read K3a `max_speed` (a gap
  there = thinner sim speed-tail / selection mix). Do NOT add a `simCommitOverheadSeconds`
  scalar. *(Oort Jun 22: even this read was incomplete — K3b's oort residual was the stale
  read-wait inflation in the SCORER's duration input, not the speed model; see Settled roots.)*
- **Oort `in_flight_after` decay ≠ "carry-over gate broken" once decile-0 matches.** Gate
  (pinned `_round_start_vclock`) is correct; decay is a selection-mix tail. The `system_util`
  recency guard was the WRONG fix (Jun 19): with intrinsic per-task latency the last-observed
  duration is *correct*, so returning `system_util=1` removes a correct penalty (= forbidden
  speed-tail widening, no principled threshold). Instrument with `commit_visibility` to
  classify the decay instead.
- **`phase_gpu_compute` gap is run-length-sensitive.** felix 1h sim 0.18 vs real 0.42 (FAIL);
  2.5h sim 0.422 vs real 0.356 (PASS). Re-check at 2.5h before acting on a 1h FAIL.
- **`pastdated_by_source=[fresh=…]` was an EVAL artifact (Jun 20).** Classifier keys on
  `MODEL_VERSION` (= current round for a fresh-dispatched eval), so eval commits carrying a
  stale-train `sct` are mislabeled "fresh" with huge gaps. Root: `evaluate()` reused the last
  train `_sim_completion_ts` (`syncfl/trainer.py:418`). **Tell:** one trainer's eval commits
  repeat the same `sct` for hundreds of rounds. Fixed by stamping a per-eval `sct`.
- **`mqtt_fetch_s` is re-selection wait, NOT network transfer (Jun 20).** It's `wall(recv)` —
  the trainer blocked in `recv()` between sending round N and being re-selected
  (`syncfl/trainer.py:187-192`); scales with rounds-skipped (10-gap → ~35s). Real
  `gpu_compute_s` is 0.17s; cadence is governed by re-selection timing. Do NOT add the ~18.8s
  mean to `sct` (`mqtt-on-sct` dead end). The spread sim misses is per-trainer availability
  stagger, destroyed by round-boundary batch re-dispatch.
- **Two past-dating populations, two streams.** `commit_gap_s`/U6 is emitted only in the
  train (WEIGHTS) branch; eval exits before it. The train-only U6 mean and the all-commits
  SIM_BARRIER/CLOCK_DIAG stream can diverge wildly (16.8s vs "fresh=95%"). Always
  disambiguate which stream a "past-dating" number came from.
- **`gate_holds=0` over a whole run = the gate is structurally INERT, not satisfied (Jun
  21).** A correct gate never fires when the *state it judges* is wrong (felix: per-end
  `_sim_inflight_expected` overwritten by overlapping re-dispatch → earlier `sct` untracked).
  Read a pinned-0 counter as a tell, suspect upstream accounting.
- **Diagnose past-dating by partitioning the TAIL, not the mean (Jun 21).** After a fix drops
  the mean (26s→3.9s) the median can be 0; the residual is a thin growing tail (14% >5s,
  decile-0 0.46s→decile-9 8.2s). Bin per-commit `commit_gap` by run-fraction and read the
  tail's signature; the growth over the run is the compounding tell.
- **Measure invariants from overlapping intervals, not cumulative warning counters (Jun
  21).** `[SELECTION_CHECK] N unreturned versions` over-counts (one lost update inflates it
  for the rest of the run). The clean metric for one-in-flight: overlapping per-trainer
  dispatch→commit intervals — real 0% vs sim 13.9% settled the felix root.
- **`sim_committed_fresh` = agg_goal (10) confirms the block-for-K fix is closed** (oort 2.5h);
  any future `committed_fresh` gap is a different root.
- **45min exercises every mechanism but not C1/C2 or low-frequency drift** (refl 3h-only `K2`).
  Checker-side fixes validate instantly against stored dirs; only sim *mechanism* changes
  need a cluster rerun.

## Dead ends — do NOT retry
- **refl A2 num_eligible via §4.5 `pending_after`→`pending_ends` (hold ALL buffered until
  commit) — FALSIFIED (Jun 22).** Over-holds: sim `buf_depth`≈57–63 vs `held`≈46–48, so it
  would exclude ~13 more (eligible 252.7→~240) vs the ~3-end target (real 249.8). ~13 ends sit
  ready-but-uncommitted in the reorder buffer, but real does NOT hold all of them out — its
  selected_ends/eligible accounting is subtler than buffer occupancy. Instrument the exact
  per-round eligible decomposition in both modes BEFORE any hold change.
- Overhead > 0 on the virtual clock (masks & drifts; clock must `= max(vclock, sct)`).
- Prediction-only gates with no real blocking (never fire).
- **Tuning the `_sim_recv_min` gate predictor (`exp = sim_send_ts + budget`)** — realized
  compute is deterministic so `exp == sct` exactly; `gate_holds=0` came from the
  `_sim_inflight_expected` overwrite, not a loose predictor. Enforce one-in-flight instead.
- **Expressing "busy" via the UN_AVL unavailable list** (felix `simInflightResidence` v1,
  reverted). UN_AVL = can't participate at all; a busy trainer is AVL_*, just occupied.
  Routing busy → `curr_unavail_trainer_list` makes `_handle_send_state` evict it from
  `selected_ends`, free its slot, refill `c` with NEW trainers → in-flight ramps to N≈300,
  vclock crawls 0.27 s/rd. Hold the slot in `selected_ends` until commit instead (§3.resid).
  Sync oort's §4.5 unavail-list use is unaffected (barrier re-selects the cohort).
- `version_at(sct)` staleness relabel (fedbuff consumes the real number; inert).
- Adding `mqtt_fetch` (~57s) to `sct` (not version-relevant; inflates staleness ~6×).
- `simRedispatchGapSeconds=0` (over-overlaps) and `=0.6` (no measurable effect) for felix —
  don't retune this scalar; the gap is a real mechanism (or supersede via §3.drain).
- **"21s MQTT weight-fetch spreads real completions" (Jun 20)** — disproven; real
  `gpu_compute_s`=0.17s, `mqtt_fetch_s` is re-selection wait. No `mqtt-on-sct` / transfer-leg.
- **Re-dispatch `sim_send_ts = prior_sct` / `= max(now, prior_sct+latency)`** — backdates the
  stamp before the round whose weights arrive → `MODEL_VERSION`/causality break.
- **`simStaggeredRedispatch` / event-driven re-dispatch as the felix throughput fix —
  FALSIFIED (§3.evt).** Made advance worse (1.93→1.38): injected stagger is bounded by the
  very clock advance it's meant to create (≤0.55s injectable — circular). Throughput root was
  commit-side (`recv_fifo` stranding), fixed by `simSctOrderedDrain` (§3.drain). Kept in code
  (gated off), disabled in the yaml; never enable with `simSctOrderedDrain`.
- **Reading high wall-clock commit density as sim "running fast" (Jun 20)** — expected (no
  trainer wait); judge per-round *vclock* advance and `commit_gap`, not wall cadence.
- **`system_util` recency guard for oort carry-over decay (Jun 19)** — value-fudge identical
  to speed-tail widening, no principled threshold; it's the A2c selection-mix class.
- **"Widen the oort slow-speed tail" for carry-over decay** — `trainer_speed` already passes;
  sim tail is if anything wider. Selection-mix tail effect, not a speed-model gap.
- **oort task-type-keyed latency / train-vs-eval duration split (Jun 20)** — premise
  FALSIFIED: sync oort dispatches 0 eval tasks (eval bundled into the train commit,
  `oort/top_aggregator.py:901-903`); felix's `system_util` is inert (`round_threshold=70`).
- **felix clock-jump clamp / dispatch-ts pacing for "fresh" past-dating** — wrong target; the
  "fresh" past-dating is EVAL committing a past stale `sct`. Fix is in `evaluate()` (per-eval
  `sct`), not a forward-advance cap or `_SIM_ORDER_SLACK_S` retune.
- Expecting the felix min-budget seed fix alone to kill past-dating (drops 73%→14%, recovers
  to 59%; other cascade sources remained).
- Expecting oort carry-over decay to be a run-length transient (2.5h: sim 0.47 vs real 3.65,
  zero by decile 2 — structural).
- Re-chasing: GPU contention (overrun 0), SEND_TIMEOUT (0×), MQTT drops (0); per-trainer
  exact-set/identity on a stochastic streaming selector (path-dependent).
- Expecting seeding to align per-round sets (`Sdet eligible_match≈0` is expected; judge
  S2/participation). A scalar fudge for `P3 mean_overhead` ~1s offset (wall-capture).

## Naming discipline (apply when touching baseline code)
Names must be **context-free** (round/version/time confusion caused real bugs — D5):
- `_round` = round index (int), never a timestamp; use `_ts`/`_time_s` for times.
- Qualify *whose* round (agg global `self._round` vs selector `self._last_selection_round` vs
  per-trainer `end_last_selection_round`). Deferred: base aggregator `self._round` →
  `self._agg_round` (all-baseline pass).
- Clients do **tasks** (train/eval), not rounds: the selector speed property is
  `PROP_CLIENT_TASK_TRAIN_DURATION` ("client_task_train_duration_s"); msg enums
  `SIM_CLIENT_TASK_TRAIN_DURATION_S`, `CLIENT_TASK_TRAIN_COMPUTE_S` (Jun 22 rename).
- A local says *what it is*, not its type-shape (`trainer_model_version` kept over
  `trained_round`). Don't paper over an ambiguous name with a comment — rename it. Scope
  renames to the baseline you're in (oort+refl share `OortSelector`; felix is separate).

---

## §1  The parity ladder (methodology)
An FL run is a pipeline; each round flows the same stages in both modes:
`clock/time-base → availability → selection → dispatch+training → return+ordering →
aggregation → utility → emergent outcomes`. Parity must hold at every stage; if it breaks at
stage N, every stage above also diverges — those are **consequences, not bugs**. The checker
finds the **lowest broken rung** (earliest stage with sound inputs but diverging output) = the
root.

**Three roles** (tag each check): **CONTROL** confirms a stage's *input* is identical (failing
= fix the input model); **MECHANISM** confirms one *transformation* is modeled (failing with
passing controls = the localized bug, the prize); **EMERGENT** an aggregate outcome (never
localizes alone — walk *down* the ladder, never fix an emergent directly).

**Two axes:** STAGE (0–9, drives diagnosis) × TIER (drives verdict): `INV` (sim invariant,
hard FAIL), `EXACT` (tight tolerance, hard FAIL), `DIST` (distributional, FAIL unless
`--lenient`), `DIAG` (informational, feeds root-cause).

**Dependency gating:** each check declares upstream prerequisites. The engine walks rungs
bottom-up, labels the lowest enforced FAIL with all-passing upstreams **ROOT-CAUSE**, demotes
higher FAILs whose chain contains a failed check to **DOWNSTREAM**. `deps` names the
*strongest causal link*, not a generic base (TC1 gates only K10 — a missing field makes a
check SKIP, not FAIL).

**Growth rule:** every root-caused bug leaves behind the most fine-grained check that would
have localized it, at its stage with deps. Checks are **append-only** (a redundant check is a
future regression guard). Split a coarse check into one assertion per mechanism.

## §2  The ladder — check catalog
`[NEW]` = to implement; else exists in checks.py. "Isolates" = what a FAIL means when its
upstreams pass. "Dep" = upstream prerequisites.

**Stage 0 — Telemetry coverage** (gate for everything)
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| TC1 `[NEW]` | Field coverage matrix | CONTROL/INV | a downstream-read field missing/sparse in one mode (explains every SKIP) | — |
| K10 | vclock_now present (sim) | CONTROL/INV | sim never stamps vclock | TC1 |

**Stage 1 — Clock / time-base** (the foundation; most bugs live here)
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| K1 | vclock monotone (sim) | MECHANISM/INV | vclock goes backwards | K10 |
| K7 | sim_rate in [0.01,100] | MECHANISM/INV | vclock/wall absurd | K10 |
| P3 | trainer_speed_s distribution | CONTROL/DIST | speed-model *input* differs | — |
| K3a `[NEW]` | Modeled-compute advance | MECHANISM/EXACT | advance **formula** (K-th fastest async / max-of-K sync), overhead excluded | P3,K1 |
| K3b `[NEW]` | Overhead residual | MECHANISM/EXACT | `real_advance − sim_advance` ≈ 0 (missing per-commit overhead) | K3a |
| K4 | Overlap factor | MECHANISM/DIAG | sim misses inter-round async pipelining | P3,K1 |
| K3 | Per-round advance dist | EMERGENT/EXACT | K3a+K3b+K4 rollup | K3a,K3b,K4 |
| K2 | Rounds-per-virtual-second | EMERGENT/EXACT | throughput rollup | K3 |

> Decomposition is the point: P3✓ K3a✓ **K3b✗** K4✓ → pure missing overhead. K3b
> cross-validates at Stage 4 (mqtt_fetch): trainer-level overhead = K3b residual × agg_goal.

**Stage 2 — Availability** (indexed by the clock; gated on Stage 1)
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| A1 | avail_composition parity | MECHANISM/DIST | per-state counts diverge | — |
| A2 | num_eligible / num_candidates | MECHANISM/DIST | eligible-set size diverges | A1 |
| A3 `[NEW]` | Trace time-base consistency | CONTROL/DIST | availability indexed by different clocks (REFL HIGH-1) | K3 |
| A4 `[NEW]` | Per-trainer duty-cycle | MECHANISM/DIST | on/off fraction differs even when set sizes match | A3 |

> A2-fail + A3-pass = fix the clock first; A2-fail + A3-fail = fix the trace lookup.

**Stage 3 — Selection**
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| S3/4 | num_chosen / in_flight / effective_c | MECHANISM/DIST | selector picks a different count | A2 |
| A2c | selected-vs-pool speed bias | MECHANISM/DIST | scoring bias diverges with pool matched (oort) vs pool itself (A2b, refl) | A2b |
| Sx | selector score-term localize | DIAG | which utility term drives a mix split (oort believed_I/temporal/system_util) | A2b |
| Sd | preferred-duration penalty bind | MECHANISM/DIST | oort speed-penalty binding freq + reconstructed `pref` (caught D1: real 80% vs sim 46%) | A2b |
| S2 | Participation frequency | EMERGENT/DIST | per-trainer chosen-count diverges | S3/4 |
| S1 | Per-round Jaccard | DIAG | exact set identity (WARN for stochastic) | A2 |

**Stage 4 — Dispatch & training** (per-trainer timing; the overhead source)
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| T2 `[NEW]` | training_budget_s dist | CONTROL/DIST | speed-model *input* differs | — |
| T_pre/T_w2g/T_gpu/T_w2r/T_post `[NEW]` | per-phase splits | MECHANISM/DIST | one timing phase each (T_gpu dep T2) | — |
| T_mqtt `[NEW]` | mqtt_fetch_s phase | MECHANISM/DIST | per-commit MQTT overhead (cross-checks K3b) | — |
| T3 | GPU budget respected | MECHANISM/INV | real GPU overruns modeled budget | T2 |
| K6 | sim_send_ts correctness | CONTROL/INV | sim dispatch ts not stamped/advancing | K10 |

> Split the one `trainer_phase` DIAG blob into per-phase DIST sub-checks so the report names
> the diverging phase; keep the combined table for at-a-glance reading.

**Stage 5 — Update return & ordering**
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| U5 | Inter-arrival order (Spearman) | MECHANISM/DIST (WARN) | arrival rank within a round diverges | K3,S3/4 |
| U4 | agg_goal_count cycles 1..K | MECHANISM/INV | lost/double-counted update | — |

**Stage 6 — Aggregation**
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| U6 | Commit visibility lag | MECHANISM/DIST | aggregator-clock delay READY→COMMITTED diverges (sim past-dating); upstream of staleness. Same metric both modes (sim `vclock−sct`, real `wall commit−arrival`) | K3 |
| U3 | Staleness distribution | MECHANISM/DIST | staleness diverges (async: downstream of clock under-charge) | K3,U5,U6 |
| P1 | Aggregation sequence | EMERGENT/DIST (WARN) | per-round contributing set diverges | S2,U5 |

**Stage 7 — Statistical utility**: F1-3 per-trainer utility dists (EMERGENT/DIST; dep S2,T_gpu,U3).
**Stage 8 — Emergent**: K8 terminal-state @ matched V (dep K2,S2); U2 total commits @ V (dep
K2,U4); C1 accuracy curve, C2 `[NEW]` loss curve (dep F1-3,K8).
**Stage 9 — Budget/stop sanity**: K9 stopped-by-budget-not-cap (WARN); K5 failsafe ceiling
(sim wall overshoot >20%).

---

## §3  Mechanism reference — landed sim fixes
The sim does **real GPU compute** but stamps a *modeled* completion `sct` (no wall sleep).
**Overriding principle:** parity ≠ goal, a *correct* simulator is; real is the reference only
after `validate_real` shows it admissible (concurrency 28.8/c30, double-dispatch 0). Never
tune sim to a wrong real. All mechanisms below are config-gated (flag-off ⇒ byte-identical)
and guarded by tests.

### Clock & ordering (felix async, validated)
- **Overhead → 0** (`simCommitOverheadSeconds=0`): clock TRACKS completions
  (`vclock = max(vclock, sct)`), not an overhead ramp.
- **Drain by physical READINESS** (`_sim_recv_min`): admit any in-flight end whose message
  physically arrived into the reorder buffer → slow trainers buffer as futures, commit in
  `sct` order (staleness 7.2→3.5).
- **`realDistributeSettleSeconds=0`**: removes a real-only 2×`sleep(0.1)`/commit (advance 4.1,
  staleness 2.8).

### §3.drain  sct-ordered DIRECT drain (felix async; LANDED Jun 20)
`simSctOrderedDrain`. The async clock under-advanced because in-flight updates (instant in
sim) ingested via the `recv_fifo` streamer could be **stranded** out of the reorder buffer's
view (background task + shared `_rx_queue` + per-end dedup + grace timeout). The clock advanced
off the incomplete buffer and lapped stranded lower-`sct` updates → past-dating (`queue_wait`
26s/90.6% >5s, staleness 15 vs 2.8; 1.4 s/rd vs 3.85). Fix: drain each live in-flight end's rx
queue **directly** so the buffer is a complete snapshot and the min-`sct` gate + clock-jump
clamp commit in true order (`commit_gap≈0`). `End.get_ready_nowait` (non-blocking, peek-aware)
+ `Channel.drain_ready` (pull raw on backend loop, decode off-loop so `cloudpickle.loads`
can't stall the pump); `_sim_recv_min` drains `recv_ends ∪ _sim_inflight_expected`. Sync
untouched (`recv_fifo(first_k=len(ends))` barrier already waits for the whole cohort). Guard:
`test_async_sct_ordered_drain.py`. **Validated Jun 21:** `commit_gap` median 0, staleness
15→5.1, advance 1.4→3.04. Residual = overlapping re-dispatch → §3.resid.

### §3.resid  One-in-flight-per-trainer (felix async; `simInflightResidence`; LANDED Jun 21)
**Invariant:** a trainer is re-pickable ONLY after its update returns AND is committed. Real
satisfies it by construction (channel holds an in-flight trainer out of `VAL_CH_STATE_SEND`
until aggregated — real 0% overlap). Felix sim freed trainers instantly → a fast trainer
re-selected while its prior update was in flight (sim 13.9%), overwriting `_sim_inflight_expected`
→ earlier update untracked, invisible to the `sct` gate (`gate_holds=0`) → lapped → past-dated
(14% tail, staleness 34). **First attempt (busy → UN_AVL) was WRONG** — see Dead ends.
**Correct fix:** a busy trainer HOLDS its concurrency slot in `selected_ends` (like real) until
commit. `_sim_hold_busy_slots` (`asyncfl/top_aggregator.py`, called from `_aggregate_weights`
at agg-goal) holds `pending_ends() ∪ set(_sim_inflight_expected)` in `selected_ends`/
`all_selected`/`_sim_pending_commit` (a SLOT, not unavail); released on commit in
`_sim_recv_min`. Bounds concurrency (`extra = c − len(selected_ends)`) AND excludes from the
pool. Sim only; sync uses §4.5 unavail path (correct — barrier re-selects). Guard:
`test_async_inflight_residence.py`. **VALIDATED Jun 21** (`…154600…sim` 90min): **46/46**, K3b
0.82→**−0.08**, advance 3.04→**3.93** (real 3.85), staleness 5.11→**2.83** (real 2.79), U6
mean_diff 3.91→**0.016s**. The whole K3b→{K2,U3,U6,K8,U2} cluster cleared together — one root.

### §3.evt  Event-driven re-dispatch (felix; FALSIFIED Jun 20, superseded by §3.drain)
`simStaggeredRedispatch` (gated, kept off). Pushed each commit's advanced vclock to a freed-slot
FIFO to re-stamp re-dispatched `sim_send_ts` and regain stagger. **Falsified:** advance got
worse (1.93→1.38) — the injected stagger is bounded by the clock advance it's meant to create
(≤0.55s injectable, circular). Real root was commit-side (§3.drain). Do not enable with
`simSctOrderedDrain`. Guard kept: `test_async_staggered_redispatch.py`.

### §3.async  Async ≠ sync selector knobs — do NOT inherit Oort *paper* defaults
`third_party/Oort` is sync-only; the paper defaults are SYNC values. Applied to the async
`AsyncOortSelector` (felix) they regress it (overlap 10.9× vs 6.6×) because the overlap model
is calibrated to the selected MIX and sync knobs narrow it. Root theme: many Oort knobs are
**per-round**, but "a round" differs in async (one `agg_goal` batch) vs sync (a barrier), and
async runs ~2–3× more of them.
- **`round_threshold`** (speed penalty) protects a SYNC barrier; async has no barrier →
  largely inert. Felix starts at **70**. *(Correction Jun 24, §S.pacer: the earlier "the pacer
  only raises toward 100, so the start washes out" described the BUGGY raise-only ratchet — that
  ratchet is exactly why the penalty went inert. The pacer is now the faithful two-branch
  controller and train-gated, so round_threshold no longer monotonically washes out; felix's
  penalty becomes active and its 46/46 must be re-validated.)*
- **`exploration_decay`** applied per-round; sync 0.95 floors exploration in ~29 async rounds.
  Felix uses **0.999** (`0.9999` ≈ never exploits — rejected).
- temporal/UCB `√(0.1·log(round)/last_selected)` auto-inflates with round count; pacer cadence
  fires more often in wall-time; staleness weighting is async-only.
- Principled generalization (not done): re-parameterize per-round terms by wall-time /
  samples-seen. Until then knobs are config-driven, anchored to the real run's spacing.

### §4.5  refl/oort — `sct`-gated pool exclusion (`simInflightResidence`, validated)
A trainer modeled as still computing (`vclock < sct`) must NOT re-enter the eligible pool. In
`oort/top_aggregator._distribute_weights`, `_sim_buffer.pending_after(vclock)` is added to
`trainer_unavail_list` (the *unavailable* path, NOT `selected_ends` — which would re-dispatch
and reset `sct`); released at `vclock ≥ sct`. Fixed refl's pool composition (A2b 12.4→~6.5 =
real). Guard: `test_virtual_clock.py::test_pending_after_*`, `TestSimInflightResidence`.

### §4.9  oort — `sct`-gated carry-over (`simInflightCarryover`)
Sync oort over-selects (×1.3), closes at agg_goal=10, leaving ~3 slowest computing. Real keeps
them in `selected_ends` across rounds (`in_flight_after` 3.3); sim's update arrives at once,
gets stale-rejected, frees its slot → drains to 0.15. Fix: a prior-round straggler
(`_round − MODEL_VERSION > 0`) with `sct > vclock_round_start` is held (not yielded, not
clock-advanced), re-buffered in `selected_ends`, commits a few rounds later. Three follow-on
bugs fixed: (1) **lost straggler** — re-buffer ran after the yield-loop the caller abandons;
wrapped in `try/finally` (Jun 16). (2) **threshold creep** — `vclock_round_start` re-read each
`_oort_sim_recv` call crept forward across block-for-K retries; `_aggregate_weights` now pins
`self._round_start_vclock` once (Jun 17). (3) **block-for-K-fresh starvation** — the second
poll loop ran `while not self.simulated`, so sim got one pass and skipped not-yet-ready fresh
trainers → committed stale; removed the gate (confirmed 2.5h `sim_committed_fresh=10`). Guard:
`TestSimInflightCarryover`. *(The residual carry-over "decay" was the A2c scorer-input root —
see Settled roots; NOT a gate or speed-tail bug.)*

### §S.dur  oort/refl real selector duration = intrinsic client stamps (LANDED Jun 23 PM)
The selector's per-client speed signal `PROP_CLIENT_TASK_TRAIN_DURATION` feeds
`system_util = (pref/duration)^α`, where `pref` = round_threshold-th percentile of candidate
durations. **Real recorded `WALL_SEND_TS − dispatch`**, which still carried the dispatch→recv
**delivery lag** (`WALL_RECV_TS − dispatch`): the aggregator stamps `dispatch` at selection, but
a slow client held one-in-flight receives the weights later, so the lag is large for stragglers
(+8s for D≥20). That inflated real's slow-trainer durations, raised `pref` (real 10.7s vs sim
7.0s), so real UNDER-penalized slow clients → committed a ~0.77s-slower mix → per-round max-of-K
10.09 vs sim 9.69 → +0.5s/rd advance → fewer rounds = the **oort+refl K2 root**. The earlier
read-wait fix (§Settled oort) anchored the RECV side on a client stamp (`WALL_SEND_TS`) but left
the DISPATCH side aggregator-anchored. **Fix:** anchor BOTH ends on client stamps —
`_real_client_task_train_duration` returns `WALL_SEND_TS − WALL_RECV_TS` = the client's intrinsic
compute+sleep = exactly D (proven: trainer's own `wall_send − wall_recv == budget`; real GPU
~0.01s) = sim's `max(gpu,D)`. Real-path only; selector-input only (clock/`sct` untouched, so
K3a/K3b on committed updates unaffected); fresh returns barely move (delivery lag ~0.07s), only
the stale-straggler recordings (where the divergence lived) are corrected. Shared by oort+refl
(one oort aggregator). Fallbacks: `WALL_SEND − dispatch`, then `recv − dispatch`, when a client
stamp is absent. Telemetry `[CLIENT_DUR_STRIP]` (intrinsic vs delivery_lag per stale record).
**SINGLE-SOURCED:** the definition lives in `flame/mode/horizontal/client_duration.py`
(`real_client_task_train_duration`); oort's `_real_client_task_train_duration` is now a thin
wrapper, and asyncfl/felix (line ~951, was `recv−dispatch`), syncfl/feddance+fedavg (line ~519,
was `wall_lag_s`), and fwdllm all call it (each falling back to its prior agg-anchored measure
only when a client stamp is missing). syncfl's `_real_task_dur` (`WALL_SEND−dispatch`) U6
barrier anchor (§6.u6) is a SEPARATE quantity — left untouched. Guards:
`test_client_duration.py` (helper contract), `test_sync_sim_ordering.py::TestStaleTrainerPropsRecorded`
(oort path + fallback). **VALIDATED Jun 24 3h:** CLIENT_DUR_STRIP confirms real `intrinsic_s≈D`,
`delivery_lag≈0.02s` (the +8s lag the fix was scoped against was already negligible in this run);
refl throughput family CLOSED (K2 rel 0.047), felix no regression at 3h. For OORT it exposed the
§S.pacer SIM-side root (real `pref` cleaned to 8.58, sim sits at 10.14) — see §S.pacer.

### §S.pacer  oort — dynamic `round_threshold` PACER was an UNFAITHFUL port (ROOT-CAUSED + FIXED Jun 24; validate next run)
The Oort speed penalty `system_util = min(1,(pref/duration)^α)` only binds when a candidate's
duration exceeds `pref = round_preferred_duration`, the **`round_threshold`-th percentile** of
candidate durations (`oort.py:496`). `round_threshold` is **DYNAMIC**, set by `pacer()`. **The
flame base `OortSelector.pacer()` was an UNFAITHFUL port** of reference Oort
(`third_party/Oort/oort/oort.py:184-199`): the reference makes TWO symmetric moves keyed on the
exploited-utility trend over the last two `pacer_step` windows — a FLAT plateau (`|Δ| ≤ 0.1·last`)
RELAXES (`round_threshold += pacer_delta`), a SHARP change (`|Δ| ≥ 5·last`) TIGHTENS
(`round_threshold = max(pacer_delta, round_threshold − pacer_delta)`). flame's port instead raised
on **ANY** dip (`last > curr`) and had **NO** decrease branch → a monotonic ratchet to 100,
hypersensitive to per-round utility noise. Because the noise timing/magnitude differs between sim
and real (different stochastic utility trajectories), the ratchet drifted to different levels and,
never recovering, the divergence COMPOUNDED. Jun 24 3h: pref grew in both modes (real Q1→Q4
4.53→15.19, sim 4.31→19.49) but **sim faster** — Q3 sim 11.48/real 9.09, Q4 sim 19.49/real 15.19,
sim disabling the penalty (`pref=99999`) in 31 late rounds (real 0). So sim bound 73.6%/round vs
real 94.1% → ~1.5s-slower max-of-K (10.5 vs 8.99) → +1.2s/rd advance (K3b −1.22s) → fewer rounds
(K2). **Tells it's the pacer not the input:** candidate-pool static speed (12.13), selected MEAN
(8.04/8.16, A2c PASS), residence (PASS) all match; only the per-round `pref` percentile diverged,
and it's `round_threshold` (the pacer) that sets the percentile. CLIENT_DUR_STRIP confirms the
§S.dur input is clean (intrinsic≈D). **FIX LANDED:** faithful reference port in
`OortSelector.pacer(round)` — both branches on the 0.1 / 5× bands, keyed on the current `round`
(reference's `training_round`) with a `pacer_step > 0` guard. The `REFLOortSelector.pacer`
override (which was ALREADY faithful — flat/sharp, 0.1/5×) is REMOVED so oort+refl share the one
reference-matching base. `round_threshold` is added to the selection telemetry `extra` (was only
inferable from `pref`). Guard `tests/selector/test_oort_selector.py::TestPacerFidelity`.
**Selector-side, both modes identically** (changes the BASELINE toward the reference, not parity-
tuning) — the hypothesis is that a self-correcting symmetric controller, fed matched utility
distributions (F1-3 pooled KS 0.037), keeps sim/real `round_threshold` tracking instead of
ratcheting apart. **VALIDATE on the next 3h run** via the new `round_threshold` field (binned by
run-fraction: no monotonic-to-100, no sim 99999), Sd binding real≈sim, K3b/K2/K8/U2 close. Do NOT
tune `pacer_delta`/`pacer_step` or touch the §S.dur input. *(Earlier read floated a §3.async
"re-parameterize by wall-time" generalization — SUPERSEDED: the reference pacer is also per-round
and self-corrects, so the gap was the unfaithful port, not the round-indexing.)*

**Reference cross-check (Jun 24):** the fix matches BOTH references — `third_party/Oort/oort/oort.py`
:184-199 AND the `third_party/REFL/thirdparty/oort/oort.py`:176-201 fork are BYTE-IDENTICAL on the
pacer (same 0.1 / 5× bands, flat→raise / sharp→lower, `training_round`-keyed; REFL only differs in
`round_threshold=30` default, already config'd via D3). So refl inheriting the faithful base is
correct against its OWN reference, not just Oort's.

**felix (`AsyncOortSelector`) — SAME bug, ALSO fixed (validate next run).** felix is a separate
class (no async-Oort reference) but the pacer is the SAME concept, so it must match the reference's
two-branch logic. Two defects: (1) the identical flat/sharp bug (raised on any dip, no decrease
branch → monotonic ratchet to 100, turning the speed penalty OFF — felix only "passed" because the
penalty was thus rendered largely inert, see §3.async); (2) felix has TWO selector hands (train +
eval) on ONE instance and `pacer()` fired in `_handle_send_state` for BOTH, but `self.round` /
`exploitation_util_history` advance only on TRAIN, so an eval call re-ran the pacer off a STALE
round. **Fix:** faithful two-branch `AsyncOortSelector.pacer()` (kept in-class) + the call is now
TRAIN-GATED (matches the reference's *training*-selector pacer). Guard
`TestPacerFidelity::test_async_oort_pacer_faithful`. ⚠ **This changes felix's round_threshold
dynamics** — the penalty is now active (oscillates near its start) instead of ratcheting to off —
so felix's 46/46 MUST be re-validated on the next run; if it shifts, the old pass partly relied on
the inert penalty (a real finding, not a clean regression).

### §6.u6  syncfl real U6 barrier-anchor (feddance/fedavg; LANDED Jun 22)
`update_visibility_lag_s` on the REAL strict-sync path was wrong: `_update_visibility_lag`
evaluated `committed = datetime.now()` **per-message inside the recv loop**, so it measured
arrival→ingestion (~0.02s/update) — NOT the barrier wait. A strict barrier applies all K at ONE
post-loop instant (`optimizer.do`), so an early finisher's true visibility lag = barrier − its
own completion. Real updates do physically arrive spread (`[MSG_ARRIVAL]` 33→48s within a round,
`queue_depth=0`); the per-message metric was just blind to it (sim 15.5s vs real 0.02s = U6
FAIL). **Fix:** real anchors on the single round barrier — `_barrier_anchored_lags(durs)` returns
`max_dur − dur_i` with `dur = WALL_SEND_TS − dispatch` (client task-train duration, the
dispatch-relative completion matching sim's `sct`); sim is unchanged (`vclock−sct`, vclock is
already advanced to the barrier). **Why only feddance, not oort/refl:** oort/refl use the
oort overlay's STREAMING commit — `_oort_sim_recv` pops in `sct` order and `_advance_sim_clock`
tracks each pop, so each update commits at `vclock≈own sct` → lag≈0 in BOTH modes (real commits
first-K-to-arrive near arrival too). The barrier wait only exists for a baseline that waits for
the slowest of its cohort (feddance). So oort's per-message helper stays correct and is left
untouched; only `syncfl._aggregate_weights` (feddance + fedavg base) is barrier-anchored.
**Validated against STORED real logs (no rerun):** recomputed lag mean 15.65/min 0/max 53 ≈ sim
15.48/0/52. Pure telemetry (no dynamics/staleness effect). Guard:
`test_sync_sim_ordering.py::test_barrier_anchored_lags_*`. Pending: confirming real feddance rerun.

## §5  Checker corrections (stochastic / observability classes)
Once dynamics match, some residual FAILs were the checker enforcing exact identity on
quantities a stochastic/in-memory sim can't reproduce (tell: byte-identical across runs despite
large dynamics changes). All principled, guarded, append-only; a future *deterministic*
selector still gets exact enforcement via `DETERMINISTIC_SELECTORS`.
- **P1 aggregation_sequence** → WARN for stochastic (S2 participation is the enforced invariant).
- **S2 participation** → enforce participation BY SPEED CLASS (`speed_class_tvd`, registry
  `speed_class`), NOT per-trainer identity, for stochastic selectors (Jun 24). The per-trainer
  `matched_count_ks` is path-dependent: a stochastic weighted-exploit selector builds a persistent
  core whose SIZE/concentration/speed-composition match but whose individual MEMBERS diverge
  (refl: 63 of ~120 shared, speed-matched). What the POLICY fixes is the speed-class distribution
  (refl TVD 0.026); per-SECOND buckets re-expose the identity noise (TVD 0.187, sign-alternating),
  so coarse `speed_class` is the right granularity. `matched_count_ks` kept as diagnostic;
  `DETERMINISTIC_SELECTORS` still get exact identity enforcement. Tell it's identity-not-bias:
  A2c/K8 pass and the mode-specific cores are speed-matched.
- **F1-3 utility** → pooled KS (per-trainer KS=1.0 was mechanical for n≤2; means identical).
- **phase_mqtt_fetch** → DIAG (in-mem cache wall time, deliberately off the virtual clock).
- **trainer_speed / eligible_speed / selection_bias** → integer-grid / metadata-pool.

## Discrepancy ledger — flame vs reference Oort
flame has ONE `OortSelector` for both `oort` (matches `third_party/Oort`) and `refl` (matches
`third_party/REFL` fork). The references differ on defaults, so each baseline's knobs are
config-driven (`selector.kwargs`), defaulting to the Oort paper (`OORT_PAPER_DEFAULTS`) with
refl overriding.

| # | discrepancy | resolution |
|---|---|---|
| D1 | `pref` not sorted | FIXED (sort added) — port bug; validated on oort |
| D2 | stat-utility not normalized/clipped | FIXED (`scoring.oort_normalize_reward`, config) |
| D3 | `round_threshold` | config: oort/felix=10 (paper), refl=30 (fork) |
| D4 | `cut_off_util` + cutoff-index | FIXED: config (0.7 paper / 0.05 refl); index thresholds the exploit-boundary score (was inert) |
| D5 | temporal time-base | **SUPERSEDED by D7 (Jun 23) for oort+refl.** Jun-16 fix stamped at selection (`_record_last_selected_round`) to dodge commit-order dependence; D7 found that was still unfaithful (reference keys the UCB term on last-RECEIPT round + registration-init, not dispatch round) and replaced it with `PROP_LAST_RETURNED_ROUND` — the last-selected machinery is removed. felix (`AsyncOortSelector`, separate class) keeps its own last-selected path, still deferred. |
| D6 | `clip_bound` | config: 0.98 paper / 0.9 fork |
| S | refl exploitation | FIXED: was deterministic top-k; now fork's cut_off_util-weighted `np.random.choice` |
| D7 | UCB temporal-uncertainty `time_stamp` | **FIXED (Jun 23, §S.temporal).** Reference Oort+REFL: `sc += sqrt(0.1·log(round)/time_stamp)`, `time_stamp=self.epoch` (agg round of last RECEIPT), init at registration → never None, always contributes, up-weights under-selected/slower-returning clients. flame bug: refl's term was DEAD (0/7513) — `refl_oort.select()` let it divide by a None `time_stamp`; oort used last-SELECTED (dispatch round) with a None→0 guard. **Fix:** `PROP_LAST_RETURNED_ROUND` stamped at every receipt (fresh+stale) in oort/top_aggregator = agg round; selector reads it, registration-init lazy to current round; both oort+refl. `enable_temporal` kwarg (default True; False = ablation only). The legacy last-SELECTED machinery (`_record_last_selected_round`, D5) is REMOVED from OortSelector (no baseline used it); D5's MODEL_VERSION value was for staleness, not this UCB term. felix AsyncOortSelector is a separate class — untouched. |
| D8 | `pacer()` round_threshold adaptation | **FIXED (Jun 24, §S.pacer).** Reference (`oort.py:184-199`) makes TWO symmetric moves on the exploited-utility trend: FLAT `|Δ|≤0.1·last` → `round_threshold += pacer_delta`, SHARP `|Δ|≥5·last` → `round_threshold = max(pacer_delta, −pacer_delta)`, keyed on `training_round`. flame's base `OortSelector.pacer()` raised on ANY dip (`last > curr`) with NO decrease branch → monotonic ratchet to 100, noise-sensitive → sim/real `round_threshold` diverged & back-half-compounded (the oort §S.pacer root). **Fix:** faithful both-branch port keyed on the current round, `pacer_step>0` guard; `REFLOortSelector.pacer` override (already faithful) REMOVED so oort+refl share the base; `round_threshold` added to selection telemetry. Guard `TestPacerFidelity`. **Cross-checked vs the REFL fork too** (`third_party/REFL/thirdparty/oort/oort.py`:176-201, byte-identical pacer, only round_threshold=30 default differs). **felix `AsyncOortSelector` (separate class) had the SAME bug + fired the pacer on its eval hand off a stale round → ALSO fixed** (faithful two-branch, train-gated; `test_async_oort_pacer_faithful`); changes felix dynamics → re-validate its 46/46 next run. |
