#!/usr/bin/env python3
"""Quick test to verify memory profiler works."""

import sys
import torch
from memory_profiler import MemoryProfiler

def test_profiler():
    """Test basic profiler functionality."""
    print("Testing MemoryProfiler...")
    
    # Initialize profiler
    profiler = MemoryProfiler(trainer_id="test_trainer", log_interval_rounds=1)
    print("✓ Profiler initialized")
    
    # Test basic memory stats
    stats = profiler.get_memory_stats()
    print(f"✓ Got memory stats: RSS={stats['rss_mb']:.1f}MB, "
          f"{stats['total_objects']} objects, "
          f"{stats['torch_tensors']['total_count']} tensors")
    
    # Test round logging
    profiler.log_memory_before_round()
    print("✓ Logged memory before round")
    
    # Create some tensors to change memory
    tensors = [torch.randn(100, 100) for _ in range(10)]
    
    profiler.log_memory_after_round()
    print("✓ Logged memory after round")
    
    # Test component logging
    profiler.log_component_memory("test_component", "BEFORE")
    print("✓ Component memory logging works")
    
    # Test report generation
    profiler.log_memory_before_round()
    profiler.log_memory_after_round()
    
    report = profiler.generate_report()
    print("\n" + report)
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_profiler()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
