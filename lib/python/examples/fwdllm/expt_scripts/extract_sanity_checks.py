#!/usr/bin/env python
"""Extract fwdllm smoke-test sanity-check signals into one log file.

Four sections: (1) selected clients per selection event, (2) data_id
sequence, (3) eval accuracy + iterations taken per data_id, (4) per-client
data partition (client_idx, hash, sample count).

Usage:
    python extract_sanity_checks.py <run_dir> [run_dir ...]

Writes <run_dir>/sanity_checks.log, auto-discovering that run's
*_aggregator.log / *_trainers.log.
"""
import argparse
import re
import sys
from pathlib import Path

SELECTED_ENDS_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2},\d{2}:\d{2}:\d{2}\.\d{3}) - "
    r"\{random\.py \(\d+\)\} - select\(\): new selected ends: (?P<ends>\{.*\})"
)
OORT_HEADER_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2},\d{2}:\d{2}:\d{2}\.\d{3}) - "
    r"\{async_oort\.py \(\d+\)\} - _select_candidates_using_default\(\): "
    r"Candidates selected with utilities"
)
OORT_CANDIDATE_RE = re.compile(
    r"\{async_oort\.py \(\d+\)\} - _select_candidates_using_default\(\): "
    r"(?P<end_id>[0-9a-f]+), (?P<utility>[\d.eE+-]+)"
)

DATA_ID_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2},\d{2}:\d{2}:\d{2}\.\d{3}) - .*"
    r"(?:_distribute_weights_(?:a?sync)|_prepare_distribution_payload).*?"
    r"data_id[=:]\s*(?P<data_id>\d+)"
)

EVAL_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2},\d{2}:\d{2}:\d{2}\.\d{3}) - "
    r"\{fwdllm_aggregator\.py \(\d+\)\} - eval_model\(\): results after eval are: "
    r"(?P<results>\{.*?\}), len\(wrong\) is: (?P<n_wrong>\d+), "
    r"'data_id_iterations': (?P<iters>\d+)"
)
ACC_IN_RESULTS_RE = re.compile(r"'acc': ([\d.eE+-]+)")
MCC_IN_RESULTS_RE = re.compile(r"'mcc': ([\d.eE+-]+)")

TRAINER_PID_RE = re.compile(
    r"\{main\.py \(\d+\)\} - <module>\(\): \[Trainer (?P<trainer_id>[0-9a-f]+)\] "
    r"PID: (?P<pid>\d+)"
)
CLIENT_HASH_RE = re.compile(
    r"(?P<pid>\d+) (?P<ts>\d{4}-\d{2}-\d{2},\d{2}:\d{2}:\d{2}\.\d{3}) - "
    r"\{base_data_manager\.py \(\d+\)\} - _load_federated_data_local\(\): "
    r"CLIENT (?P<client_idx>\d+) DATA HASH: (?P<data_hash>[0-9a-f]+)"
)
CLIENT_SAMPLES_RE = re.compile(
    r"(?P<pid>\d+) (?P<ts>\d{4}-\d{2}-\d{2},\d{2}:\d{2}:\d{2}\.\d{3}) - "
    r"\{FedSgdTrainer\.py \(\d+\)\} - _write_client_data_to_file\(\): "
    r"Successfully wrote (?P<n_samples>\d+) samples to .*?flame_client_(?P<client_idx>\d+)_"
)
PID_PREFIX_RE = re.compile(r"(?P<pid>\d+) ")


def find_logs(run_dir: Path):
    agg = sorted(run_dir.glob("*_aggregator.log"))
    trn = sorted(run_dir.glob("*_trainers.log"))
    if not agg:
        raise FileNotFoundError(f"no *_aggregator.log under {run_dir}")
    if not trn:
        raise FileNotFoundError(f"no *_trainers.log under {run_dir}")
    return agg[0], trn[0]


def check_selected_clients(agg_text: str) -> list[str]:
    """Section 1: who was picked to train, per selection event, in order."""
    events = []
    for m in SELECTED_ENDS_RE.finditer(agg_text):
        ends = sorted(re.findall(r"[0-9a-f]{30,}", m.group("ends")))
        events.append((m.group("ts"), "random(sync)", ends))

    lines = agg_text.splitlines()
    for i, line in enumerate(lines):
        m = OORT_HEADER_RE.search(line)
        if not m:
            continue
        ends = []
        for follow in lines[i + 1 : i + 12]:
            cm = OORT_CANDIDATE_RE.search(follow)
            if cm:
                ends.append((cm.group("end_id"), cm.group("utility")))
            elif ends:
                break
        events.append((m.group("ts"), "async_oort", ends))

    events.sort(key=lambda e: e[0])
    out = [f"Total selection events: {len(events)}", ""]
    for ts, kind, ends in events:
        if kind == "async_oort":
            ends_str = ", ".join(f"{e}(u={u})" for e, u in ends)
        else:
            ends_str = ", ".join(ends)
        out.append(f"[{ts}] ({kind}, n={len(ends)}) selected: {ends_str}")
    return out


def check_data_bins(agg_text: str) -> list[str]:
    """Section 2: data_id sequence over the run, collapsed to transitions."""
    seq = [(m.group("ts"), m.group("data_id")) for m in DATA_ID_RE.finditer(agg_text)]
    out = []
    last_id = None
    transitions = []
    for ts, data_id in seq:
        if data_id != last_id:
            transitions.append((ts, data_id))
            last_id = data_id
    out.append(f"Total distribute events referencing a data_id: {len(seq)}")
    out.append(f"Distinct data_id transitions: {len(transitions)}")
    out.append("")
    out.append("data_id sequence (first-seen timestamp -> data_id):")
    for ts, data_id in transitions:
        out.append(f"  [{ts}] -> data_id={data_id}")
    return out


def check_accuracy(agg_text: str) -> list[str]:
    """Section 3: eval accuracy + iterations taken per data_id eval."""
    out = []
    n = 0
    for m in EVAL_RE.finditer(agg_text):
        n += 1
        results = m.group("results")
        acc_m = ACC_IN_RESULTS_RE.search(results)
        mcc_m = MCC_IN_RESULTS_RE.search(results)
        acc = acc_m.group(1) if acc_m else "?"
        mcc = mcc_m.group(1) if mcc_m else "?"
        out.append(
            f"[{m.group('ts')}] acc={acc} mcc={mcc} "
            f"data_id_iterations={m.group('iters')} n_wrong={m.group('n_wrong')}"
        )
    out.insert(0, f"Total eval events: {n}")
    out.insert(1, "")
    return out


def check_client_partitions(trainer_text: str) -> list[str]:
    """Section 4: per-client data partition (client_idx, hash, sample count)."""
    pid_to_trainer = {}
    for m in TRAINER_PID_RE.finditer(trainer_text):
        pid_to_trainer[m.group("pid")] = m.group("trainer_id")

    hashes = {}  # pid -> (client_idx, data_hash, ts)
    samples = {}  # pid -> (client_idx, n_samples, ts)
    for line in trainer_text.splitlines():
        m = CLIENT_HASH_RE.search(line)
        if m:
            hashes[m.group("pid")] = (m.group("client_idx"), m.group("data_hash"), m.group("ts"))
            continue
        m = CLIENT_SAMPLES_RE.search(line)
        if m:
            samples[m.group("pid")] = (m.group("client_idx"), m.group("n_samples"), m.group("ts"))

    pids = sorted(set(pid_to_trainer) | set(hashes) | set(samples))
    out = [f"Total trainer processes found: {len(pids)}", ""]
    for pid in pids:
        trainer_id = pid_to_trainer.get(pid, "?")
        client_idx, data_hash, _ = hashes.get(pid, ("?", "?", None))
        _, n_samples, _ = samples.get(pid, ("?", "?", None))
        out.append(
            f"trainer_id={trainer_id} pid={pid} client_idx={client_idx} "
            f"n_samples={n_samples} data_hash={data_hash}"
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        agg_path, trn_path = find_logs(run_dir)
        agg_text = agg_path.read_text(errors="replace")
        trainer_text = trn_path.read_text(errors="replace")

        out_path = run_dir / "sanity_checks.log"
        sections = [
            ("1. SELECTED CLIENTS PER SELECTION EVENT", check_selected_clients(agg_text)),
            ("2. SELECTED DATA BIN (data_id) SEQUENCE", check_data_bins(agg_text)),
            ("3. EVAL ACCURACY + ITERATIONS TAKEN PER DATA_ID", check_accuracy(agg_text)),
            ("4. DATA PARTITION LOADED PER CLIENT", check_client_partitions(trainer_text)),
        ]
        with out_path.open("w") as f:
            f.write(f"Sanity checks for: {run_dir.name}\n")
            f.write(f"aggregator log: {agg_path.name}\n")
            f.write(f"trainer log:    {trn_path.name}\n")
            for title, lines in sections:
                f.write("\n" + "=" * len(title) + "\n")
                f.write(title + "\n")
                f.write("=" * len(title) + "\n")
                f.write("\n".join(lines) + "\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
