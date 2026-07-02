# fwdllm: migration status

All of fwdllm's migration content — how it works, what was fixed, and every
durable lesson (positive and negative) from doing the migration and
hardening it since — lives in
[`../MIGRATING_TO_LAUNCHER.md`](../MIGRATING_TO_LAUNCHER.md): the common
patterns are in its core sections (§2 aggregator gotchas, §5 telemetry, §8
launcher/spawner gotchas), and fwdllm's own specifics are in §9. That doc is
the one to read.

**No fwdllm-specific blockers remain as of 2026-07-02.** The previously
open item — fwdllm's round-cached reselection stalling when a trainer got
stuck but not formally departed — is fixed and GPU-validated (§9's "fwdllm's
own reselection-cache design"). A second bug found in the same validation
pass — trainers crashing with an uncaught `KeyError` on the aggregator's
end-of-training broadcast — is also fixed (§9's "Lessons from
smoke-testing fwdllm"). Validation evidence: three fresh 2h n=100 runs, one
per baseline family (`fwdllm`, `fwdllm_plus`, `fluxtune`), run 2026-07-02.

For the full investigation history (evidence trails, exact log lines,
commit-by-commit narrative) behind everything that used to be tracked here,
see `git log -- lib/python/examples/fwdllm/MIGRATION_TO_LAUNCHER_FWDLLM.md`.

The one remaining fwdllm follow-up is **not** a migration item: once
`launcher-script-fwdllm` merges, open the legacy-code deletion PR per
[`DELETION_CANDIDATES.md`](DELETION_CANDIDATES.md).
