#!/usr/bin/env python3
"""
Quick test for resource monitoring functionality.
"""
import time
from pathlib import Path
from launch.resource_monitor import ResourceMonitor

def test_monitor():
    """Test resource monitor with default settings."""
    log_file = Path("/tmp/test_resource_monitor.log")
    
    print("Testing resource monitor...")
    print(f"Log file: {log_file}")
    
    monitor = ResourceMonitor(
        log_file=log_file,
        check_interval=5,  # 5 seconds for quick testing
        ram_warning_threshold=50.0,  # Lower for testing
        ram_critical_threshold=70.0,
    )
    
    print("Starting monitor...")
    monitor.start()
    
    print("Monitor running for 30 seconds...")
    print("Check the log file for output:")
    print(f"  tail -f {log_file}")
    
    time.sleep(30)
    
    print("\nStopping monitor...")
    monitor.stop()
    
    print("\nTest complete!")
    print(f"Check full log: cat {log_file}")

if __name__ == "__main__":
    test_monitor()
