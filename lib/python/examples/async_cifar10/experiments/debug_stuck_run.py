#!/usr/bin/env python3
"""
Debug script to analyze stuck runs by checking message counts.
Checks for discrepancies between sent/received messages in trainer and aggregator logs.
Traces the first trainer that sent updates but aggregator didn't receive them.
"""

import sys
import re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime


def analyze_aggregator_log(log_path):
    """Analyze aggregator log for message sending and receiving patterns."""
    print(f"\n{'='*80}")
    print(f"ANALYZING AGGREGATOR LOG: {log_path}")
    print(f"{'='*80}\n")
    
    weights_sent_to_trainers = []  # List of (round, trainer_id) tuples
    model_versions_received = []   # List of (round, trainer_id) tuples
    aggregation_events = []        # List of rounds where aggregation happened
    distribute_events = []         # List of rounds where distribution happened
    
    # Track last seen activities
    last_lines = []
    stuck_indicators = []
    
    # Track queue operations
    queue_operations = []
    channel_awaiting = []
    message_types_received = Counter()
    last_model_version_time = None
    last_send_time = None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    total_lines = len(lines)
    print(f"Total lines in aggregator log: {total_lines}")
    
    for i, line in enumerate(lines):
        # Track last 100 lines for stuck analysis
        if i >= total_lines - 100:
            last_lines.append(line)
        
        # Extract timestamp
        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        timestamp = timestamp_match.group(1) if timestamp_match else None
        
        # Aggregator sending weights to trainers
        # Pattern: "_distribute_weights | sending weights to 505f9fc483cf4df68a2409257b5fad7d3c580414 with model_version: 107 for task: train"
        if "_distribute_weights" in line and "sending weights to" in line and "task: train" in line:
            # Extract trainer_id and model_version
            match = re.search(r'sending weights to\s+(\w{40})\s+with model_version:\s+(\d+)', line)
            if match:
                trainer_id = match.group(1)
                model_version = int(match.group(2))
                weights_sent_to_trainers.append((model_version, trainer_id, timestamp))
                last_send_time = timestamp
        
        # Aggregator receiving model updates from trainers
        # For SYNCFL/REFL: Look for "received data from" which appears in aggregate_weights
        # Pattern: "received data from 505f9fc483cf4df68a2409257b5fad7d3c580439"
        if "received data from" in line and len(line) > 100:  # Filter out short lines
            match = re.search(r'received data from\s+(\w{40})', line)
            if match:
                trainer_id = match.group(1)
                # Look for surrounding context - usually logged near "Agg weights" or model_version
                # For syncfl, we don't have explicit round in this line, try to infer from logs
                # Use -1 as placeholder, will aggregate all together
                round_num = -1  # SYNCFL doesn't log round in this message
                model_versions_received.append((round_num, trainer_id, timestamp))
                last_model_version_time = timestamp
        
        # Alternative: count MODEL_VERSION messages (more accurate for all modes)
        # Pattern: "channel.py:481 | INFO | MainThread | recv_fifo | msg of type MODEL_VERSION recvd for end 505f9fc483cf4df68a2409257b5fad7d3c580389, model_version=9"
        elif "msg of type MODEL_VERSION recvd for end" in line:
            # Extract trainer_id (end) and model_version
            match = re.search(r'msg of type MODEL_VERSION recvd for end\s+(\w{40}),\s*model_version=(\d+)', line)
            if match:
                trainer_id = match.group(1)
                model_version = int(match.group(2))
                # Use model_version as the round number for tracking
                round_num = model_version
                # Only add if we haven't already counted this via "received data from"
                if not any(tid == trainer_id and ts == timestamp for _, tid, ts in model_versions_received[-10:]):
                    model_versions_received.append((round_num, trainer_id, timestamp))
                    last_model_version_time = timestamp
        
        # Track all message types received
        if "msg of type" in line and "recvd for end" in line:
            msg_type_match = re.search(r'msg of type\s+(\w+)\s+recvd', line)
            if msg_type_match:
                msg_type = msg_type_match.group(1)
                message_types_received[msg_type] += 1
        
        # Track channel awaiting operations
        if "awaiting get" in line or "awaiting put" in line:
            channel_awaiting.append((timestamp, line.strip()[-120:]))
        
        # Track queue operations
        if "Drained" in line and "messages from end_id" in line:
            queue_operations.append((timestamp, line.strip()[-100:]))
        
        # Check for aggregation events
        if "aggregat" in line.lower() and ("complete" in line.lower() or "finished" in line.lower()):
            round_match = re.search(r'round[:\s]+(\d+)', line, re.IGNORECASE)
            if round_match:
                aggregation_events.append(int(round_match.group(1)))
        
        # Check for distribution events
        if "distribut" in line.lower() and ("complete" in line.lower() or "finished" in line.lower()):
            round_match = re.search(r'round[:\s]+(\d+)', line, re.IGNORECASE)
            if round_match:
                distribute_events.append(int(round_match.group(1)))
        
        # Check for stuck indicators
        if "timeout" in line.lower() or "waiting" in line.lower() or "blocked" in line.lower():
            stuck_indicators.append(line.strip())
    
    # Analysis
    print(f"\n--- MESSAGE COUNTS ---")
    print(f"Weights sent to trainers:     {len(weights_sent_to_trainers)}")
    print(f"Model versions received:      {len(model_versions_received)}")
    print(f"Discrepancy:                  {abs(len(weights_sent_to_trainers) - len(model_versions_received))}")
    
    # Per-trainer statistics at aggregator
    trainer_agg_stats = defaultdict(lambda: {'sent': 0, 'received': 0})
    
    for model_version, trainer_id, timestamp in weights_sent_to_trainers:
        trainer_agg_stats[trainer_id]['sent'] += 1
    
    for round_num, trainer_id, timestamp in model_versions_received:
        trainer_agg_stats[trainer_id]['received'] += 1
    
    print(f"\n--- PER-TRAINER STATISTICS AT AGGREGATOR ---")
    print(f"{'Trainer ID':<12} {'Sent':>6} {'Received':>8} {'Difference':>11}")
    print(f"{'-'*12} {'-'*6} {'-'*8} {'-'*11}")
    
    # Sort by difference (descending) to show problematic trainers first
    sorted_trainers = sorted(
        trainer_agg_stats.items(),
        key=lambda x: abs(x[1]['sent'] - x[1]['received']),
        reverse=True
    )
    
    total_sent_check = 0
    total_received_check = 0
    total_diff = 0
    
    for trainer_id, stats in sorted_trainers:
        tid_short = trainer_id[-10:] if trainer_id else "unknown"
        sent = stats['sent']
        received = stats['received']
        diff = sent - received
        total_sent_check += sent
        total_received_check += received
        total_diff += diff
        
        status = "" if diff == 0 else " ⚠"
        print(f"{tid_short:<12} {sent:>6} {received:>8} {diff:>11}{status}")
    
    print(f"{'-'*12} {'-'*6} {'-'*8} {'-'*11}")
    print(f"{'TOTAL':<12} {total_sent_check:>6} {total_received_check:>8} {total_diff:>11}")
    
    if total_diff != 0:
        print(f"\n⚠ Total difference: {total_diff} (should be 0)")
        trainers_with_diff = len([t for t, s in sorted_trainers if s['sent'] != s['received']])
        print(f"   {trainers_with_diff} trainers have send/receive mismatches")
    else:
        print(f"\n✓ All per-trainer differences sum to 0")
    
    # Timing analysis
    if last_send_time and last_model_version_time:
        print(f"\nLast weight sent:             {last_send_time}")
        print(f"Last MODEL_VERSION received:  {last_model_version_time}")
    
    # Count by round
    sent_by_round = Counter([r for r, _, _ in weights_sent_to_trainers if r != -1])
    recv_by_round = Counter([r for r, _, _ in model_versions_received if r != -1])
    
    # Count all received (including those without round info)
    recv_all = len(model_versions_received)
    recv_with_round = len([r for r, _, _ in model_versions_received if r != -1])
    recv_without_round = recv_all - recv_with_round
    
    if sent_by_round or recv_by_round:
        print(f"\n--- BY ROUND ---")
        print(f"Note: {recv_without_round} received messages don't have round info (common in SYNCFL/REFL)")
        all_rounds = sorted(set(list(sent_by_round.keys()) + list(recv_by_round.keys())))
        
        # Find where receiving stopped
        last_recv_round = max(recv_by_round.keys()) if recv_by_round else -1
        first_stuck_round = None
        for round_num in all_rounds:
            sent = sent_by_round.get(round_num, 0)
            recv = recv_by_round.get(round_num, 0)
            if sent > 0 and recv == 0 and first_stuck_round is None:
                first_stuck_round = round_num
        
        print(f"Last round with received messages: {last_recv_round}")
        if first_stuck_round:
            print(f"First round with 0 received (stuck): {first_stuck_round}")
            print(f"Stuck for {len([r for r in all_rounds if r >= first_stuck_round])} rounds")
        
        for round_num in all_rounds[-10:]:  # Show last 10 rounds
            sent = sent_by_round.get(round_num, 0)
            recv = recv_by_round.get(round_num, 0)
            status = "✓" if sent == recv else "✗ MISMATCH"
            print(f"  Round {round_num:3d}: Sent={sent:3d}, Recv={recv:3d}  {status}")
    
    print(f"\n--- AGGREGATION/DISTRIBUTION EVENTS ---")
    print(f"Aggregation events: {len(aggregation_events)}")
    print(f"Distribution events: {len(distribute_events)}")
    if aggregation_events:
        print(f"Last aggregation round: {max(aggregation_events)}")
    if distribute_events:
        print(f"Last distribution round: {max(distribute_events)}")
    
    # Message types analysis
    if message_types_received:
        print(f"\n--- MESSAGE TYPES RECEIVED ---")
        for msg_type, count in message_types_received.most_common():
            print(f"  {msg_type:20s}: {count:5d}")
    
    # Queue operations analysis
    if queue_operations:
        print(f"\n--- QUEUE DRAIN OPERATIONS (last 30) ---")
        # Count non-zero drains
        non_zero_drains = [(ts, op) for ts, op in queue_operations if "Drained 0" not in op]
        zero_drains = [(ts, op) for ts, op in queue_operations if "Drained 0" in op]
        print(f"Total drain operations: {len(queue_operations)}")
        print(f"Non-zero drains: {len(non_zero_drains)}")
        print(f"Zero drains: {len(zero_drains)}")
        if non_zero_drains:
            print(f"\nLast non-zero drains:")
            for ts, op in non_zero_drains[-10:]:
                print(f"  [{ts}] {op}")
        print(f"\nLast drain operations (all):")
        for ts, op in queue_operations[-30:]:
            print(f"  [{ts}] {op}")
    
    # Channel awaiting analysis
    if channel_awaiting:
        print(f"\n--- CHANNEL AWAITING OPERATIONS (last 20) ---")
        print(f"Total awaiting operations: {len(channel_awaiting)}")
        for ts, op in channel_awaiting[-20:]:
            print(f"  [{ts}] {op}")
    
    if stuck_indicators:
        print(f"\n--- STUCK INDICATORS (last 10) ---")
        for indicator in stuck_indicators[-10:]:
            print(f"  {indicator[:120]}")
    
    print(f"\n--- LAST 20 LINES OF LOG ---")
    for line in last_lines[-20:]:
        print(f"  {line.rstrip()}")
    
    return {
        'weights_sent': len(weights_sent_to_trainers),
        'weights_received': len(model_versions_received),
        'sent_by_round': sent_by_round,
        'recv_by_round': recv_by_round,
        'message_types': message_types_received,
        'last_send_time': last_send_time,
        'last_recv_time': last_model_version_time,
        'weights_sent_list': weights_sent_to_trainers,
        'weights_received_list': model_versions_received,
    }


def analyze_trainer_log(log_path):
    """Analyze trainer log for message sending and receiving patterns."""
    print(f"\n{'='*80}")
    print(f"ANALYZING TRAINER LOG: {log_path}")
    print(f"{'='*80}\n")
    
    fetch_weights_complete = []  # List of (trainer_id, round) tuples
    weights_sent = []            # List of (trainer_id, round) tuples
    
    # Track per-trainer stats
    trainer_stats = defaultdict(lambda: {'fetched': 0, 'sent': 0, 'last_round': -1})
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Total lines in trainer log: {total_lines}")
    
    current_trainer = None
    
    for i, line in enumerate(lines):
        # Identify trainer from log line prefix or trainer_id mentions
        trainer_match = re.search(r'Trainer id:\s*(\w{40})', line)
        if trainer_match:
            current_trainer = trainer_match.group(1)
        
        # Pattern: "### FETCH WEIGHTS complete for trainer_id 505f9fc483cf4df68a2409257b5fad7d3c580389, round: 11, model_version: 11, task_to_perform (can be default): train"
        if "FETCH WEIGHTS complete for trainer_id" in line and "task_to_perform" in line and "train" in line:
            # Extract trainer_id, round, and model_version
            match = re.search(r'FETCH WEIGHTS complete for trainer_id\s+(\w{40}),\s+round:\s+(\d+),\s+model_version:\s+(\d+)', line)
            if match:
                trainer_id = match.group(1)
                round_num = int(match.group(2))
                model_version = int(match.group(3))
                fetch_weights_complete.append((trainer_id, round_num))
                trainer_stats[trainer_id]['fetched'] += 1
                trainer_stats[trainer_id]['last_round'] = max(trainer_stats[trainer_id]['last_round'], round_num)
        
        # Pattern: "[TRAINER_SEND_WEIGHTS] Sent weights for trainer_id: 505f9fc483cf4df68a2409257b5fad7d3c580389, model_version: 11, _updates_returned_upto_round: 11"
        if "[TRAINER_SEND_WEIGHTS]" in line or ("Sent weights for trainer_id:" in line and "_updates_returned_upto_round" in line):
            # Extract trainer_id, model_version, and round
            match = re.search(r'Sent weights for trainer_id:\s+(\w{40}),\s+model_version:\s+(\d+),\s+_updates_returned_upto_round[:\s]+(\d+)', line)
            if not match:
                # Try old format for backward compatibility
                match = re.search(r'Sent weights for trainer_id:\s+(\w{40}),\s+_updates_returned_upto_round[:\s]+(\d+)', line)
                if match:
                    trainer_id = match.group(1)
                    round_num = int(match.group(2))
                    weights_sent.append((trainer_id, round_num))
                    trainer_stats[trainer_id]['sent'] += 1
            else:
                trainer_id = match.group(1)
                model_version = int(match.group(2))
                round_num = int(match.group(3))
                weights_sent.append((trainer_id, round_num))
                trainer_stats[trainer_id]['sent'] += 1
    
    # Analysis
    print(f"\n--- MESSAGE COUNTS ---")
    print(f"Fetch weights complete:       {len(fetch_weights_complete)}")
    print(f"Weights sent to aggregator:   {len(weights_sent)}")
    print(f"Discrepancy:                  {abs(len(fetch_weights_complete) - len(weights_sent))}")
    
    # Count by round
    fetch_by_round = Counter([r for _, r in fetch_weights_complete if r != -1])
    sent_by_round = Counter([r for _, r in weights_sent if r != -1])
    
    if fetch_by_round or sent_by_round:
        print(f"\n--- BY ROUND ---")
        all_rounds = sorted(set(list(fetch_by_round.keys()) + list(sent_by_round.keys())))
        for round_num in all_rounds[-10:]:  # Show last 10 rounds
            fetched = fetch_by_round.get(round_num, 0)
            sent = sent_by_round.get(round_num, 0)
            status = "✓" if fetched == sent else "✗ MISMATCH"
            print(f"  Round {round_num:3d}: Fetched={fetched:3d}, Sent={sent:3d}  {status}")
    
    # Per-trainer stats
    print(f"\n--- PER-TRAINER STATS (showing trainers with discrepancies) ---")
    mismatched_trainers = []
    for trainer_id, stats in trainer_stats.items():
        if stats['fetched'] != stats['sent']:
            mismatched_trainers.append((trainer_id, stats))
    
    if mismatched_trainers:
        print(f"Found {len(mismatched_trainers)} trainers with fetch/send mismatches:")
        for trainer_id, stats in sorted(mismatched_trainers, key=lambda x: abs(x[1]['fetched'] - x[1]['sent']), reverse=True)[:20]:
            tid_short = trainer_id[-8:] if trainer_id else "unknown"
            print(f"  {tid_short}: Fetched={stats['fetched']:3d}, Sent={stats['sent']:3d}, ",
                  f"Diff={stats['fetched'] - stats['sent']:3d}, LastRound={stats['last_round']}")
    else:
        print("  All trainers have matching fetch/send counts ✓")
    
    return {
        'fetch_complete': len(fetch_weights_complete),
        'weights_sent': len(weights_sent),
        'trainer_stats': trainer_stats,
        'fetch_complete_list': fetch_weights_complete,
        'weights_sent_list': weights_sent,
    }


def check_mqtt_issues(log_paths):
    """Check for MQTT-related issues in logs."""
    print(f"\n{'='*80}")
    print(f"CHECKING MQTT ISSUES")
    print(f"{'='*80}\n")
    
    mqtt_errors = []
    chunk_issues = []
    connection_issues = []
    
    for log_path in log_paths:
        if not Path(log_path).exists():
            continue
            
        with open(log_path, 'r') as f:
            for line in f:
                if "mqtt" in line.lower():
                    if "error" in line.lower() or "exception" in line.lower():
                        mqtt_errors.append(line.strip()[:150])
                    if "chunk" in line.lower() and ("miss" in line.lower() or "lost" in line.lower()):
                        chunk_issues.append(line.strip()[:150])
                    if "disconnect" in line.lower() or "reconnect" in line.lower():
                        connection_issues.append(line.strip()[:150])
    
    if mqtt_errors:
        print(f"--- MQTT ERRORS ({len(mqtt_errors)}) ---")
        for err in mqtt_errors[-10:]:
            print(f"  {err}")
    else:
        print("No MQTT errors found ✓")
    
    if chunk_issues:
        print(f"\n--- CHUNK ISSUES ({len(chunk_issues)}) ---")
        for issue in chunk_issues[-10:]:
            print(f"  {issue}")
    else:
        print("No chunk issues found ✓")
    
    if connection_issues:
        print(f"\n--- CONNECTION ISSUES ({len(connection_issues)}) ---")
        for issue in connection_issues[-10:]:
            print(f"  {issue}")
    else:
        print("No connection issues found ✓")


def trace_missing_messages(agg_log, trainer_log, agg_stats, trainer_stats):
    """
    Trace the first trainer that sent updates but aggregator didn't receive them.
    Cross-references aggregator sends, trainer fetches/sends, and aggregator receives.
    """
    print(f"\n{'='*80}")
    print(f"TRACING MISSING MESSAGES")
    print(f"{'='*80}\n")
    
    # Build sets for quick lookup
    # Format: {trainer_id: [(round, timestamp), ...]}
    agg_sent = defaultdict(list)  # Aggregator sent to trainer
    agg_received = defaultdict(list)  # Aggregator received from trainer
    trainer_fetched = defaultdict(list)  # Trainer fetched weights
    trainer_sent = defaultdict(list)  # Trainer sent back updates
    
    # Process aggregator sends
    for round_num, trainer_id, timestamp in agg_stats['weights_sent_list']:
        agg_sent[trainer_id].append((round_num, timestamp))
    
    # Process aggregator receives
    for round_num, trainer_id, timestamp in agg_stats['weights_received_list']:
        agg_received[trainer_id].append((round_num, timestamp))
    
    # Process trainer fetches and sends
    for trainer_id, round_num in trainer_stats['fetch_complete_list']:
        trainer_fetched[trainer_id].append(round_num)
    
    for trainer_id, round_num in trainer_stats['weights_sent_list']:
        trainer_sent[trainer_id].append(round_num)
    
    print(f"Unique trainers that aggregator sent to: {len(agg_sent)}")
    print(f"Unique trainers that aggregator received from: {len(agg_received)}")
    print(f"Unique trainers that fetched: {len(trainer_fetched)}")
    print(f"Unique trainers that sent back: {len(trainer_sent)}")
    
    # Find trainers that were sent to but never responded
    missing_trainers = []
    for trainer_id in agg_sent:
        num_sent = len(agg_sent[trainer_id])
        num_received = len(agg_received.get(trainer_id, []))
        if num_sent > num_received:
            missing_count = num_sent - num_received
            missing_trainers.append((trainer_id, missing_count, num_sent, num_received))
    
    missing_trainers.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n--- TRAINERS WITH MISSING RESPONSES ---")
    print(f"Found {len(missing_trainers)} trainers with missing responses")
    
    if not missing_trainers:
        print("No missing trainer responses found - all sends have matching receives ✓")
        return
    
    print(f"\nTop 20 trainers by missing response count:")
    for trainer_id, missing_count, sent, received in missing_trainers[:20]:
        tid_short = trainer_id[-8:]
        print(f"  {tid_short}: Sent={sent:3d}, Received={received:3d}, Missing={missing_count:3d}")
    
    # Find the FIRST occurrence of a missing message (chronologically)
    print(f"\n--- FINDING FIRST MISSING MESSAGE ---")
    
    # Get all sends with their timestamps, sorted chronologically
    all_sends = []
    for trainer_id, sends in agg_sent.items():
        for round_num, timestamp in sends:
            all_sends.append((timestamp, trainer_id, round_num))
    
    all_sends.sort()  # Sort by timestamp
    
    # Check each send to see if there's a corresponding receive
    first_missing = None
    for timestamp, trainer_id, round_num in all_sends:
        # Check if this trainer_id has a receive around this time
        # For REFL/SYNCFL, rounds might not match exactly, so we check if ANY receive exists
        receives = agg_received.get(trainer_id, [])
        
        # Count how many sends happened before this one for this trainer
        sends_before = len([ts for r, ts in agg_sent[trainer_id] if ts <= timestamp])
        # Count how many receives happened (for this trainer)
        receives_count = len(receives)
        
        # If more sends than receives at this point, this might be the first missing
        if sends_before > receives_count:
            first_missing = (timestamp, trainer_id, round_num, sends_before, receives_count)
            break
    
    if first_missing:
        timestamp, trainer_id, round_num, sends_before, receives_count = first_missing
        tid_short = trainer_id[-8:]
        print(f"\n🔍 FIRST MISSING MESSAGE FOUND:")
        print(f"   Timestamp:   {timestamp}")
        print(f"   Trainer ID:  {trainer_id}")
        print(f"   Round:       {round_num}")
        print(f"   Aggregator had sent {sends_before} messages to this trainer")
        print(f"   Aggregator had received {receives_count} messages from this trainer")
        print(f"   This is the first send without a corresponding receive")
        
        # Now check trainer side - did this trainer fetch and send back?
        trainer_fetched_rounds = sorted(trainer_fetched.get(trainer_id, []))
        trainer_sent_rounds = sorted(trainer_sent.get(trainer_id, []))
        
        print(f"\n   Trainer side:")
        print(f"   - Fetched {len(trainer_fetched_rounds)} times: {trainer_fetched_rounds[:10]}{'...' if len(trainer_fetched_rounds) > 10 else ''}")
        print(f"   - Sent back {len(trainer_sent_rounds)} times: {trainer_sent_rounds[:10]}{'...' if len(trainer_sent_rounds) > 10 else ''}")
        
        # Check if this specific round was fetched and sent
        if round_num in trainer_fetched_rounds:
            print(f"   ✓ Trainer DID fetch weights for round {round_num}")
        else:
            print(f"   ✗ Trainer did NOT fetch weights for round {round_num}")
        
        if round_num in trainer_sent_rounds:
            print(f"   ✓ Trainer DID send back updates for round {round_num}")
        else:
            print(f"   ✗ Trainer did NOT send back updates for round {round_num}")
        
        # Extract detailed logs around this timestamp
        print(f"\n--- DETAILED LOGS AROUND MISSING MESSAGE ---")
        extract_logs_around_time(agg_log, trainer_log, timestamp, trainer_id, round_num)
    else:
        print("Could not identify a specific first missing message")


def extract_logs_around_time(agg_log, trainer_log, timestamp_str, trainer_id, round_num):
    """
    Extract log lines from both aggregator and trainer logs around a specific timestamp.
    Looks for relevant entries ±2 minutes from the timestamp.
    """
    from datetime import datetime, timedelta
    
    try:
        target_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except:
        print(f"Could not parse timestamp: {timestamp_str}")
        return
    
    time_window = timedelta(minutes=2)
    start_time = target_time - time_window
    end_time = target_time + time_window
    
    tid_short = trainer_id[-8:]
    
    # Search aggregator log
    print(f"\n📄 AGGREGATOR LOG (±2 min around {timestamp_str}):")
    print(f"   Looking for trainer {tid_short} and round {round_num}")
    
    agg_lines = []
    with open(agg_log, 'r') as f:
        for line in f:
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                try:
                    line_time = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                    if start_time <= line_time <= end_time:
                        # Check if line is relevant (mentions trainer or MODEL_VERSION)
                        if tid_short in line or trainer_id in line or "MODEL_VERSION" in line:
                            agg_lines.append(line.rstrip())
                except:
                    pass
    
    if agg_lines:
        print(f"   Found {len(agg_lines)} relevant lines:")
        for line in agg_lines[:50]:  # Show first 50
            print(f"   {line}")
    else:
        print(f"   No relevant lines found in time window")
    
    # Search trainer log
    print(f"\n📄 TRAINER LOG (±2 min around {timestamp_str}):")
    print(f"   Looking for trainer {tid_short} and round {round_num}")
    
    trainer_lines = []
    with open(trainer_log, 'r') as f:
        for line in f:
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                try:
                    line_time = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                    if start_time <= line_time <= end_time:
                        # Check if line mentions this trainer
                        if tid_short in line or trainer_id in line:
                            trainer_lines.append(line.rstrip())
                except:
                    pass
    
    if trainer_lines:
        print(f"   Found {len(trainer_lines)} relevant lines:")
        for line in trainer_lines[:50]:  # Show first 50
            print(f"   {line}")
    else:
        print(f"   No relevant lines found in time window")
    
    # Additional analysis: Look for MQTT transmission logs
    print(f"\n📡 CHECKING FOR MESSAGE TRANSMISSION LOGS:")
    print(f"   Searching for 'put' or 'send' operations for trainer {tid_short}...")
    
    mqtt_sends = []
    with open(trainer_log, 'r') as f:
        for line in f:
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                try:
                    line_time = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                    if start_time <= line_time <= end_time:
                        if (tid_short in line or trainer_id in line) and ("put" in line.lower() or "send" in line.lower() or "tx" in line.lower()):
                            mqtt_sends.append(line.rstrip())
                except:
                    pass
    
    if mqtt_sends:
        print(f"   Found {len(mqtt_sends)} transmission-related lines:")
        for line in mqtt_sends[:30]:
            print(f"   {line}")
    else:
        print(f"   No transmission logs found")
    
    # Check aggregator for message reception logs
    print(f"\n📥 CHECKING AGGREGATOR MESSAGE RECEPTION:")
    print(f"   Searching for 'recv' or 'received' for trainer {tid_short}...")
    
    mqtt_recvs = []
    with open(agg_log, 'r') as f:
        for line in f:
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                try:
                    line_time = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                    if start_time <= line_time <= end_time:
                        if (tid_short in line or trainer_id in line) and ("recv" in line.lower() or "received" in line.lower() or "rx" in line.lower()):
                            mqtt_recvs.append(line.rstrip())
                except:
                    pass
    
    if mqtt_recvs:
        print(f"   Found {len(mqtt_recvs)} reception-related lines:")
        for line in mqtt_recvs[:30]:
            print(f"   {line}")
    else:
        print(f"   No reception logs found - MESSAGE MAY HAVE BEEN LOST IN TRANSIT")


def trace_specific_trainer(agg_log, trainer_log, trainer_id_suffix):
    """
    Trace a specific trainer through the entire message flow.
    Shows when agg sent weights, trainer received, trainer sent back, agg received,
    and whether agg processed or discarded the update.
    
    Args:
        agg_log: Path to aggregator log
        trainer_log: Path to trainer log
        trainer_id_suffix: Last characters of trainer ID to search for
    """
    print(f"\n{'='*80}")
    print(f"TRACING SPECIFIC TRAINER: *{trainer_id_suffix}")
    print(f"{'='*80}\n")
    
    # Find full trainer ID
    full_trainer_id = None
    with open(agg_log, 'r') as f:
        for line in f:
            if trainer_id_suffix in line and 'sending weights to' in line:
                match = re.search(r'sending weights to\s+(\w{40})', line)
                if match:
                    candidate = match.group(1)
                    if candidate.endswith(trainer_id_suffix):
                        full_trainer_id = candidate
                        break
    
    if not full_trainer_id:
        print(f"❌ Could not find trainer ending with {trainer_id_suffix}")
        return
    
    print(f"Found trainer: {full_trainer_id}")
    print(f"Short ID: ...{trainer_id_suffix}\n")
    
    # Collect all events for this trainer
    agg_sends = []  # (timestamp, round, model_version)
    agg_receives = []  # (timestamp, round)
    agg_processes = []  # (timestamp, round, action) - action: "used" or "discarded"
    trainer_fetches = []  # (timestamp, round)
    trainer_sends = []  # (timestamp, round)
    
    # Parse aggregator log
    print("Parsing aggregator log...")
    with open(agg_log, 'r') as f:
        for line in f:
            if full_trainer_id not in line:
                continue
            
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            timestamp = timestamp_match.group(1) if timestamp_match else None
            
            # Aggregator sending weights
            if "sending weights to" in line and full_trainer_id in line:
                match = re.search(r'sending weights to\s+\w{40}\s+with model_version:\s+(\d+)', line)
                if match:
                    model_version = int(match.group(1))
                    agg_sends.append((timestamp, model_version, "SEND"))
            
            # Aggregator receiving MODEL_VERSION
            if "msg of type MODEL_VERSION recvd for end" in line and full_trainer_id in line:
                # Extract model_version from new format: model_version=9
                version_match = re.search(r'model_version=(\d+)', line)
                if version_match:
                    model_version = int(version_match.group(1))
                    agg_receives.append((timestamp, model_version, "RECV"))
                else:
                    # Try to extract round if available (old format)
                    round_match = re.search(r'round[:\s]+(\d+)', line, re.IGNORECASE)
                    round_num = int(round_match.group(1)) if round_match else None
                    agg_receives.append((timestamp, round_num, "RECV"))
            
            # Check if update was used or discarded
            if "received data from" in line and full_trainer_id in line:
                agg_processes.append((timestamp, None, "PROCESSED"))
            
            # Check for stale/discarded updates
            if ("stale" in line.lower() or "discard" in line.lower()) and full_trainer_id in line:
                round_match = re.search(r'round[:\s]+(\d+)', line, re.IGNORECASE)
                round_num = int(round_match.group(1)) if round_match else None
                agg_processes.append((timestamp, round_num, "DISCARDED"))
    
    # Parse trainer log
    print("Parsing trainer log...")
    with open(trainer_log, 'r') as f:
        for line in f:
            if full_trainer_id not in line:
                continue
            
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            timestamp = timestamp_match.group(1) if timestamp_match else None
            
            # Trainer fetching weights
            if "FETCH WEIGHTS complete for trainer_id" in line and full_trainer_id in line:
                # Try to extract model_version first, fall back to round
                match = re.search(r'model_version:\s+(\d+)', line)
                if match:
                    model_version = int(match.group(1))
                    trainer_fetches.append((timestamp, model_version, "FETCH"))
                else:
                    match = re.search(r'round:\s+(\d+)', line)
                    if match:
                        round_num = int(match.group(1))
                        trainer_fetches.append((timestamp, round_num, "FETCH"))
            
            # Trainer sending weights back
            if ("[TRAINER_SEND_WEIGHTS]" in line or "Sent weights for trainer_id:" in line) and full_trainer_id in line:
                # Try new format first with model_version
                match = re.search(r'model_version:\s+(\d+)', line)
                if match:
                    model_version = int(match.group(1))
                    trainer_sends.append((timestamp, model_version, "SEND"))
                else:
                    # Fall back to old format
                    match = re.search(r'_updates_returned_upto_round[:\s]+(\d+)', line)
                    if match:
                        round_num = int(match.group(1))
                        trainer_sends.append((timestamp, round_num, "SEND"))
    
    # Display timeline
    print(f"\n{'='*80}")
    print(f"MESSAGE FLOW TIMELINE")
    print(f"{'='*80}\n")
    print(f"Total events:")
    print(f"  Aggregator sends:     {len(agg_sends)}")
    print(f"  Trainer fetches:      {len(trainer_fetches)}")
    print(f"  Trainer sends back:   {len(trainer_sends)}")
    print(f"  Aggregator receives:  {len(agg_receives)}")
    print(f"  Aggregator processes: {len(agg_processes)}")
    
    missing_count = len(agg_sends) - len(agg_receives)
    if missing_count > 0:
        print(f"\n⚠️  MISSING: {missing_count} messages sent but not received by aggregator")
    
    # Merge and sort all events chronologically
    all_events = []
    for ts, mv, action in agg_sends:
        all_events.append((ts, "AGG_SEND", f"Agg sent weights (model_version={mv})"))
    for ts, rnd, action in trainer_fetches:
        all_events.append((ts, "TRAINER_FETCH", f"Trainer fetched weights (round={rnd})"))
    for ts, rnd, action in trainer_sends:
        all_events.append((ts, "TRAINER_SEND", f"Trainer sent back updates (round={rnd})"))
    for ts, rnd, action in agg_receives:
        all_events.append((ts, "AGG_RECV", f"Agg received MODEL_VERSION (round={rnd if rnd else 'unknown'})"))
    for ts, rnd, action in agg_processes:
        all_events.append((ts, "AGG_PROCESS", f"Agg {action.lower()} update"))
    
    # Define event priority for sorting when timestamps are identical
    # Lower number = happens first in causal chain
    event_priority = {
        "AGG_SEND": 1,       # Aggregator sends weights first
        "TRAINER_FETCH": 2,  # Trainer fetches what was sent
        "TRAINER_SEND": 3,   # Trainer sends back after training
        "AGG_RECV": 4,       # Aggregator receives what trainer sent
        "AGG_PROCESS": 5     # Aggregator processes after receiving
    }
    
    # Sort by timestamp first, then by causal priority for same timestamps
    all_events.sort(key=lambda x: (x[0], event_priority.get(x[1], 99)))
    
    # Show timeline
    print(f"\n{'='*80}")
    print(f"CHRONOLOGICAL EVENT TIMELINE (showing all {len(all_events)} events)")
    print(f"{'='*80}\n")
    print(f"{'Timestamp':<20} {'Event Type':<18} {'Details'}")
    print(f"{'-'*20} {'-'*18} {'-'*50}")
    
    for ts, event_type, details in all_events:
        print(f"{ts:<20} {event_type:<18} {details}")
    
    # Analyze patterns - match sends with receives
    print(f"\n{'='*80}")
    print(f"MATCHING ANALYSIS")
    print(f"{'='*80}\n")
    
    # For each send, try to find corresponding fetch -> send -> receive
    print("Analyzing send-receive pairs:\n")
    
    unmatched_sends = []
    for i, (send_ts, send_mv, _) in enumerate(agg_sends):
        # Find if trainer fetched this
        fetch_found = None
        for fetch_ts, fetch_round, _ in trainer_fetches:
            if fetch_ts >= send_ts and abs(fetch_round - send_mv) <= 2:  # Allow some round mismatch
                fetch_found = (fetch_ts, fetch_round)
                break
        
        # Find if trainer sent back
        send_back_found = None
        if fetch_found:
            for send_ts_t, send_round, _ in trainer_sends:
                if send_ts_t >= fetch_found[0] and send_round == fetch_found[1]:
                    send_back_found = (send_ts_t, send_round)
                    break
        
        # Find if aggregator received
        recv_found = None
        if send_back_found:
            for recv_ts, recv_round, _ in agg_receives:
                if recv_ts >= send_back_found[0]:
                    recv_found = (recv_ts, recv_round)
                    break
        
        # Check if processed or discarded
        process_status = None
        if recv_found:
            for proc_ts, proc_round, action in agg_processes:
                if proc_ts >= recv_found[0] and (proc_round is None or proc_round == recv_found[1]):
                    process_status = action
                    break
        
        # Report
        status_icon = "✓" if recv_found else "✗"
        print(f"{status_icon} Send #{i+1} at {send_ts} (mv={send_mv}):")
        if fetch_found:
            print(f"   ✓ Trainer fetched at {fetch_found[0]} (round={fetch_found[1]})")
        else:
            print(f"   ✗ No trainer fetch found")
        
        if send_back_found:
            print(f"   ✓ Trainer sent back at {send_back_found[0]} (round={send_back_found[1]})")
        else:
            print(f"   ✗ No trainer send back found")
        
        if recv_found:
            print(f"   ✓ Aggregator received at {recv_found[0]}")
            if process_status:
                if process_status == "PROCESSED":
                    print(f"   ✓✓ Update was PROCESSED/USED")
                else:
                    print(f"   ⚠️  Update was {process_status}")
            else:
                print(f"   ? Processing status unknown")
        else:
            print(f"   ✗ Aggregator never received")
            unmatched_sends.append((send_ts, send_mv))
        print()
    
    if unmatched_sends:
        print(f"\n⚠️  {len(unmatched_sends)} unmatched sends (trainer sent but agg didn't receive):")
        for send_ts, send_mv in unmatched_sends[:10]:
            print(f"   - {send_ts} (model_version={send_mv})")


def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_stuck_run.py <aggregator_log> <trainer_log> [trainer_id_suffix]")
        print("\nOptional: Provide last characters of trainer ID to trace specific trainer")
        print("Example: python debug_stuck_run.py agg.log trainer.log 7d3c580389")
        sys.exit(1)
    
    agg_log = Path(sys.argv[1])
    trainer_log = Path(sys.argv[2])
    trainer_id_suffix = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not agg_log.exists():
        print(f"Error: Aggregator log not found: {agg_log}")
        sys.exit(1)
    
    if not trainer_log.exists():
        print(f"Error: Trainer log not found: {trainer_log}")
        sys.exit(1)
    
    # Analyze logs
    agg_stats = analyze_aggregator_log(agg_log)
    trainer_stats = analyze_trainer_log(trainer_log)
    
    # Check MQTT issues
    check_mqtt_issues([agg_log, trainer_log])
    
    # Trace missing messages in detail
    trace_missing_messages(agg_log, trainer_log, agg_stats, trainer_stats)
    
    # If specific trainer ID provided, trace that trainer in detail
    if trainer_id_suffix:
        trace_specific_trainer(agg_log, trainer_log, trainer_id_suffix)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")
    
    agg_discrepancy = abs(agg_stats['weights_sent'] - agg_stats['weights_received'])
    trainer_discrepancy = abs(trainer_stats['fetch_complete'] - trainer_stats['weights_sent'])
    
    print(f"Aggregator discrepancy: {agg_discrepancy}")
    print(f"Trainer discrepancy:    {trainer_discrepancy}")
    
    if agg_discrepancy > 0:
        print(f"\n⚠ ISSUE: Aggregator has {agg_discrepancy} message discrepancy!")
        print(f"   Sent: {agg_stats['weights_sent']}, Received: {agg_stats['weights_received']}")
        if agg_stats['weights_sent'] > agg_stats['weights_received']:
            print(f"   → Aggregator is waiting for {agg_stats['weights_sent'] - agg_stats['weights_received']} trainer responses")
    
    if trainer_discrepancy > 0:
        print(f"\n⚠ ISSUE: Trainers have {trainer_discrepancy} message discrepancy!")
        print(f"   Fetched: {trainer_stats['fetch_complete']}, Sent: {trainer_stats['weights_sent']}")
        if trainer_stats['fetch_complete'] > trainer_stats['weights_sent']:
            print(f"   → {trainer_stats['fetch_complete'] - trainer_stats['weights_sent']} trainers fetched but didn't send")
    
    if agg_discrepancy == 0 and trainer_discrepancy == 0:
        print(f"\n✓ All message counts match - issue may be elsewhere")
        print(f"  Consider checking:")
        print(f"  - Deadlocks in aggregation logic")
        print(f"  - Waiting conditions that never trigger")
        print(f"  - Resource exhaustion (memory, connections)")


if __name__ == "__main__":
    main()
