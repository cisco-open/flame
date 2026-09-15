# Flame and FMoW: fixes and design suggestions

The fixes address incorrect behavior found while implementing FMoW, listed in suggested
priority order. The design suggestions below describe how to organize these
changes and scale satellite simulations. None of these changes are implemented yet.

## Fixes, in priority order

| Fix | Why it happens and why it matters | Suggested approach |
| --- | --- | --- |
| **1. Unfinished tasks are tracked inconsistently** | Sync can use 175 of 200 results without clearing the discarded trainers' busy status. Too few free trainers triggers a search for future availability. If none exists, the run stops, even with trainers connected. | Give tasks unique IDs and shared tracking across sync and async. Clear their busy status when accepting, discarding, or failing results. Before stopping, check pending work as well as availability. Lowering the aggregation goal only masks the problem. |
| **2. Missing results can cause unlimited waits** | Sync collects the selected cohort before sorting by simulated completion time, so a goal of 175 can still require all 200 responses. A first response may have no timeout. Missing results can therefore freeze the run. | Add task failure reporting, retries, and cancelable waits. Keep host health deadlines separate from simulated completion times. Give chunk transfers unique IDs and validate them: incomplete transfers can currently mix with later messages without reporting task failure. |
| **3. Aggregation can exhaust GPU memory** | Accepted updates remain on the server GPU until aggregation. Trainers and evaluation may share that GPU, increasing memory pressure. A saved run crashed this way at round 139. | Combine updates incrementally where the algorithm permits it, preserving weighting rules. Bound storage for retained results and reserve memory for aggregation and evaluation. |
| **4. Connected trainers may not be ready to train** | Trainers connect before finishing data loading. The server can count 200 connections and start while some trainers are still initializing. Waiting for their results then looks like a training hang. | Send an explicit “ready to train” signal after initialization. Wait for ready trainers, and report startup failures separately from missing training results. |
| **5. A stop request does not end every wait** | Some loops wait for enough updates without checking whether the run has been told to stop. The run can remain stuck in a loop even after its stop flag is set. | Make every blocking wait and training loop respond to cancellation, so stopping the run also unblocks pending waits. |
| **6. Finished evaluation can still cause a five-minute wait** | FMoW clears its “evaluation running” flag but does not notify the completion event used by a waiting evaluation. The next evaluation can wait for the five-minute timeout even though the earlier one has finished. | Update both completion signals whenever evaluation exits, including after an error. Put that notification in a `finally` block so a failure cannot leave the waiter stuck. |
| **7. Cleanup can kill another run's processes** | Cleanup matches processes using broad script-name patterns. Trainers from another run can match the same pattern and be killed too. | Track the processes started by each run and stop only those processes during cleanup. |

Logs confirm an early stop with 200 connected trainers, 17 marked busy, and only
183 eligible against a goal of 185. They do not establish why all 17 remained busy.
The memory crash is also confirmed; the other mechanisms come from source audits.

## Design suggestions

**1. Separate task and result tracking from aggregation**

Currently, channels, selectors, and aggregators keep overlapping records of
unfinished work. Receiving a result and using it in a model update are different
events, but their bookkeeping is coupled. That allows a discarded result to leave
a trainer marked busy and makes reliability fixes dependent on the chosen stack.

Introduce a shared coordinator that owns task IDs, dispatch, receipt, failures,
and completion. Selectors choose participants; aggregation policies decide which
results to accept, retain, or discard and when to update the model. Those decisions
return to the coordinator so every task has a consistent status, even when its
result is unused. Sync, async, and FedBuff would use the same tracking mechanism
while retaining their own aggregation rules.

This is the architectural part of the first fixes above. Establish these contracts
first, then migrate the stacks to them, so the same tracking bugs do not need
separate repairs in every aggregator.

**2. Use one worker process per GPU instead of one per satellite**

With 200 satellites on eight GPUs, the current setup runs 25 trainer processes per
GPU, each with a model and CUDA context. A shared worker pool would reduce this
memory and startup overhead. Satellites would remain distinct clients with their
own data and state; workers would execute their queued training tasks. Start with
one worker per GPU and measure whether a few workers improve throughput.

Sequential GPU execution can still represent overlapping satellite training. If A
and B start at simulated time 0 and take 10 and 20 modeled seconds, their results
would become available at times 10 and 20 even if the GPU calculates them one after
the other. Each task must retain its original model, data cutoff, and seed. The
coordinator would process events in simulated order, pausing if a required result
is not ready. Host queueing and execution time must stay separate from simulated
duration, replacing the current timing rule that mixes them.

The main benefit is choosing the number of processes based on GPU capacity
instead of satellite count. One worker per GPU uses less memory, but may leave
some GPU capacity unused. Two or four workers could improve throughput if memory
allows, at the cost of more memory use and competition for the GPU. For example,
two workers on each of eight GPUs would give 16 workers serving all 200 satellites.
Each worker would take the next assigned task, rather than belong to a particular
satellite. Adding satellites would then add tasks and client state, without
requiring more worker processes. The worker count can be tuned independently,
leaving room for aggregation and evaluation.
