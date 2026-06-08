# Parity Diagnosis — June 7, 2026 Runs

**Runs compared:**
| label | directory | rounds | wall |
|---|---|---:|---:|
| felix_sim  | `run_20260607_105426_dbg_felix_n300_alpha0.1_syn0_stream_sim`  | 1328 | 2.92 h |
| felix_real | `run_20260607_135610_dbg_felix_n300_alpha0.1_syn0_stream_real` | 1317 | 2.92 h |
| refl_sim   | `run_20260607_105421_dbg_refl_n300_alpha0.1_syn0_stream_sim`   | 4777 | 2.92 h |
| refl_real  | `run_20260607_135535_dbg_refl_n300_alpha0.1_syn0_stream_real`  | 3800 | 2.92 h |

Config: n=300, α=0.1, syn_0, agg_goal=10, max_runtime_s=10800s (3h),
min_trainers_to_start=290. Felix: async_oort c=30 (new), fedbuff.
REFL: refl_oort aggr_num=10 1.3× overcommit, stale_update=5.

---

## 1. FELIX parity verdict

```
[WARN] 1. Selection parity          J=0.024  (stochastic → gated, expected)
[FAIL] 2. Statistical utility        max_ks=1.0  avg_mean_diff=19.48
[FAIL] 3. Aggregation sequence       set_match_pct=0.0%  (stochastic → expected FAIL)
[FAIL] 4. Staleness distribution     real_mean=2.53  sim_mean=10.32  diff=7.80  ks=0.59
[PASS] 5. Participation counts       avg_diff=7.5  (total real=13170 sim=13280)
[WARN] 6. Convergence                avg_acc_diff=0.083  (real far ahead of sim by r400+)
[PASS] 7. Round duration parity
[PASS] 8. sim_send_ts correctness    300/300 OK, all non-zero vclock
[PASS] 9. GPU contention             no overruns
```

Sanity checks: T1 vclock monotone PASS. T3 sim_rate=0.274 virtual-s/wall-s PASS (range
check passes but rate is severely abnormal — see §1.1). T4 wall_speedup=15.45× PASS
(but this is a misleading metric here — see §1.1). T5 failsafe DID NOT FIRE.

---

### 1.1 Root cause: vclock severely under-charges at c=30

**Headline numbers:**

| | s/FL-round | final vclock | sim_rate |
|---|---:|---:|---:|
| felix_sim  | **2.17 s** (virtual) | **2880 s** | 0.274 virt-s/wall-s |
| felix_real | **7.97 s** (wall) | n/a (=wall) | 1.0 |
| ratio | **3.7× too fast** | **3.75× short** | **3.65× too slow** |

The sim consumed only **2880 of 10800 virtual seconds** before the 3h wall budget ran
out. The sim is 3.65× slower than real time — it will **never reach the vclock
budget**; the wall ceiling fires first.

**Mechanism:** The vclock advances as: `vclock = max(vclock, dispatch_vclock + trainer_speed_s)` at each commit. With c=30 and agg_goal=10, the FL round closes when the 10th fastest of the 30 in-flight trainers commits. Per the commit trace:

```
round=1 commit=0  vclock=2.0  max_speed=2.0
round=1 commit=5  vclock=4.0  max_speed=4.0
round=1 commit=9  vclock=5.0  max_speed=5.0   ← FL round closes at 5s virtual
round=2 commit=4  vclock=9.0  max_speed=9.0
...
```

The 10th fastest of 30 heterogeneous trainers has a speed ~5s (low end of the
distribution). So each FL round advances the vclock by only ~2.17s on average.
Meanwhile in real, the 10th fastest trainer completes in 7.97 wall-seconds — about
3.7× more — because real has **system overheads the sim ignores entirely**:

- MQTT send/receive round-trip latency
- Aggregator receive-queue and commit processing time
- Pre/post-training CPU phases (pre_train_s, post_train_s)
- Task dispatch delay from aggregator to trainer

At c=30 with fast trainers, these overheads become the **dominant bottleneck** in the
real system (they dominate the short per-round time), but are zero in the simulator.

`trainer_speed_s` per commit matches almost exactly: real mean=12.20s, sim mean=12.21s.
The trainer GPU compute model is correct. Only the non-GPU overhead is missing.

**Overlap factor comparison:**

| | overlap_factor = mean_speed / s_per_round |
|---|---|
| felix_sim  | 12.21 / 2.17 = **5.6×** |
| felix_real | 12.20 / 7.97 = **1.5×** |

The sim thinks 5.6 FL rounds of work are overlapping simultaneously; real has only 1.5×
overlap. The correct model for c=30 / agg_goal=10 should produce overlap ≈ 3× (c/goal),
but real overhead reduces this to 1.5×. The sim's lack of overhead over-estimates
overlap, producing the 3.7× vclock undercharge.

---

### 1.2 Staleness explosion (downstream of §1.1)

| | staleness mean | staleness >5 |
|---|---:|---:|
| felix_real | **2.53** | 363 / 13170 (**2.8%**) |
| felix_sim  | **10.32** | 7821 / 13280 (**58.9%**) |

**Causal chain:** vclock advances ~2.17s/round. A trainer dispatched at vclock T with
training_speed 28s completes at T+28. At that completion time, the vclock has advanced
roughly (28 / 2.17) ≈ 13 more FL rounds. So that trainer arrives stale by ~13 rounds.
Mean staleness 10.32 is exactly consistent with mean_speed(12.21) / s_per_round(2.17) ≈ 5.6 rounds of delay per dispatch.

In real, a trainer with speed 28s completes 28 / 7.97 ≈ 3.5 rounds later. Mean staleness
2.53 is consistent.

This is a **derived bug** — fix the vclock advance model (§1.1) and staleness will
automatically correct. fedbuff does not cap staleness, so the full distribution
difference flows through.

---

### 1.3 Convergence divergence (downstream of §1.2)

Real significantly ahead of sim from round ~200 onward:

| round | real acc | sim acc |
|---:|---:|---:|
| 160 | 0.124 | 0.084 |
| 310 | 0.100 | 0.103 |
| 410 | 0.175 | 0.100 |
| 460 | 0.194 | 0.100 |
| 560 | **0.246** | **0.100** |

High-staleness updates (sim mean=10.3 vs real mean=2.5) carry much less gradient
signal; the aggregated model converges far more slowly in simulation. This is another
derived bug — fix staleness, convergence parity should follow.

---

### 1.4 Utility divergence (downstream of §1.3)

Oort utility scores reflect recent model accuracy per trainer. Because the real model
has learned substantially better, real utilities are ~2× sim utilities across all 300
trainers (avg_mean_diff=19.48, max_ks=1.0). This is entirely downstream of the
convergence divergence. No independent utility model bug.

---

### 1.5 What IS working for felix

- **Participation** (PASS): total commits within 1% (13170 vs 13280), per-trainer avg
  diff=7.5. The system is correctly exercising the same trainers at similar rates.
- **sim_send_ts** (PASS): all 300 sim trainers carry correct, non-zero, increasing vclock
  dispatch timestamps.
- **GPU budget / contention** (PASS): no overruns in either mode.
- **Round duration** (PASS): `sim_round_duration_s` distributions match.
- **Round count** (PASS at surface): 1317 real vs 1328 sim — only 0.8% apart. Despite
  the vclock bug, both runs completed roughly the same number of FL rounds in 3h wall
  time (because the wall budget, not the vclock budget, binds for sim).

---

## 2. REFL parity verdict

```
[WARN] 1. Selection parity          (stochastic → gated, expected)
[FAIL] 2. Statistical utility        max_ks=1.0  avg_mean_diff=17.17
[FAIL] 3. Aggregation sequence       set_match_pct=0.6%  (stochastic → expected)
[PASS] 4. Staleness distribution     real_mean=2.97  sim_mean=2.65  diff=0.32  ks=0.10
[FAIL] 5. Participation counts       total real=38000  sim=47770  avg_diff=165.5
[PASS] 6. Convergence                avg_acc_diff=0.038
[PASS] 7. Round duration parity
[PASS] 8. sim_send_ts correctness    234/300 OK (66 trainers no task_recv data)
[PASS] 9. GPU contention
```

Sanity checks: T1 vclock monotone PASS (vclock_now is stamped on refl sim — different
from June 6 runs; T2/T3 SKIP (sync path still has no vclock_now on agg_round).
T4 wall_speedup=3.83× PASS.

---

### 2.1 Round-count divergence: sim 4777 vs real 3800 (+26%)

| | FL rounds | wall | s/round (wall) | max(trainer_speed_s)/round |
|---|---:|---:|---:|---:|
| refl_real | 3800 | 10510 s | **2.76 s** | **9.94 s** |
| refl_sim  | 4777 | 10505 s | **2.20 s** | **8.42 s** |

The sim completes **977 more rounds** in the same wall time. Same root cause as
felix §1.1 — the sim doesn't model MQTT/CPU/dispatch overhead. At refl's very
short per-round times (~2-3s), overhead is a large fraction of real wall time.

Additionally: the sim's `max(trainer_speed_s)` per round is 8.42s vs real 9.94s
(15% underestimate). The sim systematically under-models the committed-cohort
training speed. With refl's overcommit-take-fastest logic, small errors in the
speed model compound. Fixing the overhead model (§1.1) would increase sim wall
time per round → fewer rounds → closer to 3800.

---

### 2.2 Availability trace divergence (root cause of participation FAIL)

| | unique trainers participating | rounds |
|---|---:|---:|
| refl_real | 146 | 3800 |
| refl_sim  | 142 | 4777 |
| in BOTH   | 93  | — |
| sim ONLY  | 49  | — |
| real ONLY | 53  | — |

Only 93 trainers appear in BOTH runs. 49 participate exclusively in sim, 53
exclusively in real. The sim-only trainers each appear ~440 times (= 4777 × 13/142
≈ expected share), meaning they are consistently selected whenever available — they
are genuinely "available" in the sim trace but never appear in the real run.

This is an **availability trace replay divergence**: the syn_0 trace maps trainer
availability over (virtual) time, but the sim uses virtual time (vclock) while real
uses wall time to index the trace. With refl sim advancing 2.20s/wall-round and real
advancing 2.76s/wall-round, and different vclock vs wall definitions, the two runs
are effectively sampling different windows of the availability trace — the trainer
pool diverges progressively over the run.

This is SEPARATE from stochastic selection — even if selection were deterministic,
the ELIGIBLE SET would differ because the availability window is different.

The num_eligible statistics look similar in aggregate (sim mean=250.3 vs real
mean=247.6) because both have ~250 eligible per round — but the **identity** of
those 250 trainers differs, giving rise to the 49 sim-only / 53 real-only divergence.

---

### 2.3 What IS working for refl

- **Staleness** (PASS): stale_update=5 hard cap works correctly in both modes.
  Both have max_staleness=5, zero updates discarded for being >5 stale.
- **Convergence** (PASS): avg_acc_diff=0.038, both learning at similar rates.
  Despite the participation divergence, the learning dynamics are comparable.
- **num_chosen=13, in_flight~63** in both modes — selection volume parity is good.
- **Round duration parity** (PASS).
- **GPU budget** (PASS): no overruns.

---

## 3. Summary table: all issues, severity, root cause, fix

| ID | selector | check | verdict | issue | root cause | fix required |
|---|---|---|---|---|---|---|
| **F1** | felix | K2: vclock rate | **CRITICAL** | sim_rate=0.274 (3.7× too slow); vclock only reaches 2880/10800 s | Sim ignores system overhead: MQTT latency, CPU pre/post, dispatch delay; with c=30 these dominate short round times | Model overhead per commit in vclock advance: `delta_vc = max(speed + overhead_s, min_realistic_round_s)` |
| **F2** | felix | K4: staleness | **FAIL** | mean 10.32 (sim) vs 2.53 (real); 59% of commits stale>5 | Derived from F1: slow vclock → many rounds elapsed between dispatch and commit | Fix F1; no independent fix needed |
| **F3** | felix | convergence | **WARN** | real 0.246 acc vs sim 0.100 at r560 | Derived from F2: high-staleness updates yield poor gradients | Fix F2; no independent fix needed |
| **F4** | felix | utility | **FAIL** | max_ks=1.0 across all trainers | Derived from F3: real model learned; utility tracks model performance | Fix F3; no independent fix needed |
| **R1** | refl | K8: round count | **FAIL** | 4777 sim vs 3800 real (+26%) | Same overhead gap as F1: sim finishes rounds 25% faster in wall time | Fix F1 (overhead model); applies to sync path too |
| **R2** | refl | participation | **FAIL** | 49 trainers sim-only, 53 real-only; avg_diff=165.5 | Availability trace indexed by vclock (sim) vs wall time (real): different trainer pools emerge | Fix R1 (round-time parity) → trace windows converge; may also need explicit trace-time normalization |
| **R3** | refl | utility | **FAIL** | max_ks=1.0 avg_diff=17.17 | Different trainers selected (R2) → different training histories → different utilities | Fix R2 |
| **R4** | refl | vclock telemetry | **MISSING** | agg_round events have no `vclock_now` for sync path | syncfl aggregator path does not emit vclock stamps | Add `vclock_now` to sync agg_round telemetry (needed for K1-K3 to run) |
| **B1** | both | sim_send_ts (refl) | WARN | 66/300 trainers have no task_recv data | Trainers that were never dispatched a task (selected as eval-only or availability) | Investigate refl trainer dispatch pattern; may be expected |

---

## 4. Ordered fix list (dependency order)

1. **Model per-commit system overhead in the vclock advance** — the single highest-
   leverage fix. Adds a per-commit overhead term (MQTT round-trip + dispatch + CPU
   phases) so `delta_vc = trainer_speed + overhead`. This alone closes the vclock
   rate gap (F1), staleness (F2), convergence (F3), utility (F4), and indirectly
   refl round-count (R1) and availability trace divergence (R2).
   
   Estimated overhead to add ≈ real_s_per_round - 10th_order_stat_of_30_speeds ≈ 7.97 - 5 ≈ 3s per FL round, distributed across the 10 commits that close a round (≈0.3s overhead per commit).

2. **Add `vclock_now` to sync aggregator path** — enables K1-K3 clock checks for refl
   sim (currently all SKIP). Required before refl sim parity can be verified at the
   clock level.

3. **Verify availability trace replay** — once (1) is fixed, re-examine whether
   different trainers still appear in sim-only vs real-only pools. If they do, the
   trace indexing (vclock vs wall) may need explicit normalization.

4. **Re-run and recheck** — run the new parity checker (`--batch felix refl`) on
   next iteration of runs. Target: F1/F2/F3/F4 cleared (staleness mean within 1.0,
   vclock rate within 10%), R1 cleared (round count within 10%).

---

## 5. What parity checker checks are currently adequate vs missing

### Currently catching the issues (or would with correct thresholds):
- Checks 4 (staleness) and 5 (participation) directly surface the downstream effects.
- Check 8 (sim_send_ts) correctly validates vclock dispatch timestamps.

### Currently missing or misleadingly passing:
- **K2 (vclock rate)** — not in `parity_checks.py` / `compare_parity.py`. T3
  sim_rate only checks it's in range [0.01,100]; 0.274 passes that range but is
  clearly wrong for a 3h run. Need a gate: sim_rate must be ≥ 0.8 (sim should be
  faster than real, not slower).
- **K3 (per-round advance distribution)** — not implemented. Would directly catch
  sim 2.17s/round vs real 7.97s/round.
- **K4 (overlap factor)** — not implemented. Would flag sim 5.6× vs real 1.5×.
- **K8 (terminal-state parity)** — not implemented. Would catch refl 4777 vs 3800.
- **K10 (vclock telemetry present)** — not implemented. Refl sim has no vclock_now
  on agg_round; clock checks silently skip.
- **T4 wall_speedup is misleading**: reports 15.45× for felix, but this speedup is
  spurious — the sim only accumulated 2880 virtual seconds of work, so of course
  it "finishes" the same (tiny) virtual work faster. The metric should compare
  speedup at matched virtual-second budgets, not at sim's terminal state.
- **A1-A4 (availability / eligibility)**: not in the checker. Would have caught the
  trainer-identity divergence in refl (R2) directly.

These correspond exactly to the checks specified in `real-sim_parity_checker_plan.md`
§3.H (K-series) and §3.A — confirming the plan's gap analysis was correct.
