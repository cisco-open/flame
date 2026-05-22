# FedDance on async_cifar10 — TODO

FedDance has a `feddance` baseline in `examples/_metadata/baselines.yaml`, but it
is **not yet runnable as an async_cifar10 smoke test**. The other five baselines
(felix, fedbuff, fedavg, oort, refl) have working smoke YAMLs; FedDance is
deferred until the items below are resolved.

## Why it doesn't run yet

1. **Missing required kwarg.** `FedDanceSelector.__init__` does
   `self.aggr_num = kwargs["aggr_num"]` (required) —
   [`flame/selector/feddance.py`](../../flame/selector/feddance.py). The
   `feddance` baseline's `selector.kwargs` does not set `aggr_num`, so init
   raises `KeyError`.

2. **Tracking mismatch.** The baseline sets `trackTrainerAvail.type: ORACULAR`,
   but FedDance derives availability from *observed check-ins* via
   `FedDancePredictor` (`predictor.record_checkin(...)`), not from an oracular
   trace. The oracular setting is unused/misleading here.

3. **Stack mismatch.** The baseline currently points at `main_fedavg_agg.py`
   (base syncfl). That stack's `get_curr_unavail_trainers()` expects a
   `trainer_event_dict`, which `main_fedavg_agg.py` never populates. Need to
   confirm the correct sync stack for the FedDance selector + FedAvg optimizer.

4. **FedDance has its own example.** `examples/feddance_cifar10/` ships its own
   aggregator main and data pipeline. It may be cleaner to smoke-test FedDance
   there than to retrofit it into async_cifar10.

## Decision needed

Pick one:

- **(A) Wire FedDance into async_cifar10.** Add `aggr_num` to the baseline,
  drop/replace the oracular tracking, and point `example.aggregator_main` at a
  stack that (a) drives the `feddance` selector hooks
  (`on_update_received` / `on_round_completed`) and (b) doesn't require an
  oracular `trainer_event_dict`. Then add `feddance_n10_*_smoke.yaml`.
- **(B) Smoke-test in `feddance_cifar10`.** Build the launcher path there
  instead, and leave the async_cifar10 `feddance` baseline out (or documented as
  unsupported).

Until then, the `feddance` baseline entry should be treated as a placeholder.
