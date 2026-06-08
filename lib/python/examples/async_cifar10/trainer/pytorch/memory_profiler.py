"""Memory profiling utilities for identifying CPU memory leaks in trainers."""

import gc
import logging
import sys
import time
from collections import defaultdict
from typing import Dict, Any

import torch
import psutil

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Track memory usage and identify leaks across training rounds."""

    def __init__(self, trainer_id: str, log_interval_rounds: int = 1,
                 enabled: bool = False):
        """Initialize memory profiler.

        Args:
            trainer_id: Identifier for this trainer
            log_interval_rounds: Log detailed memory stats every N rounds
            enabled: When False (default), the per-round heap-walk methods
                (`log_memory_before_round`, `log_memory_after_round`,
                `log_component_memory`) are no-ops. These walk *all* Python
                objects (3x) with a per-object `torch.is_tensor()` check and
                call `gc.collect()`; at high trainer-per-host concurrency that
                dominates per-round wall time (it is the bulk of the observed
                "excess" trainer time). Leave off in production runs; turn on
                only when actively chasing a leak.
        """
        self.trainer_id = trainer_id
        self.log_interval_rounds = log_interval_rounds
        self.enabled = enabled
        self.round_count = 0
        self.process = psutil.Process()
        
        # Track memory across rounds
        self.round_memory = []
        self.initial_memory = None
        
        # Track object counts
        self.object_counts_history = []
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {}
        
        # System memory
        mem_info = self.process.memory_info()
        stats['rss_mb'] = mem_info.rss / 1024 / 1024
        stats['vms_mb'] = mem_info.vms / 1024 / 1024
        
        # Python object counts
        stats['total_objects'] = len(gc.get_objects())
        
        # Count objects by type
        type_counts = defaultdict(int)
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            type_counts[obj_type] += 1
        
        # Get top 10 object types
        top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        stats['top_object_types'] = dict(top_types)
        
        # PyTorch tensor stats
        stats['torch_tensors'] = {
            'total_count': 0,
            'cpu_tensors': 0,
            'gpu_tensors': 0,
            'cpu_memory_mb': 0,
            'gpu_memory_mb': 0,
        }
        
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                stats['torch_tensors']['total_count'] += 1
                tensor_size_mb = obj.element_size() * obj.nelement() / 1024 / 1024
                
                if obj.is_cuda:
                    stats['torch_tensors']['gpu_tensors'] += 1
                    stats['torch_tensors']['gpu_memory_mb'] += tensor_size_mb
                else:
                    stats['torch_tensors']['cpu_tensors'] += 1
                    stats['torch_tensors']['cpu_memory_mb'] += tensor_size_mb
        
        # Add PyTorch CUDA memory stats if available
        if torch.cuda.is_available():
            stats['cuda_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            stats['cuda_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return stats
    
    def log_memory_before_round(self):
        """Log memory state before training round starts."""
        self.round_count += 1
        if not self.enabled:
            return

        # Force garbage collection before measuring
        gc.collect()
        
        stats = self.get_memory_stats()
        
        if self.initial_memory is None:
            self.initial_memory = stats['rss_mb']
        
        self.round_memory.append({
            'round': self.round_count,
            'timestamp': time.time(),
            'stats': stats
        })
        
        # Log summary every round
        logger.info(
            f"[MEMORY] Trainer {self.trainer_id} Round {self.round_count} START: "
            f"RSS={stats['rss_mb']:.1f}MB (+{stats['rss_mb']-self.initial_memory:.1f}MB from start), "
            f"Tensors: {stats['torch_tensors']['total_count']} "
            f"(CPU: {stats['torch_tensors']['cpu_tensors']}/{stats['torch_tensors']['cpu_memory_mb']:.1f}MB, "
            f"GPU: {stats['torch_tensors']['gpu_tensors']}/{stats['torch_tensors']['gpu_memory_mb']:.1f}MB), "
            f"Total objects: {stats['total_objects']}"
        )
        
        # Detailed logging at intervals
        if self.round_count % self.log_interval_rounds == 0:
            self._log_detailed_stats(stats)
    
    def log_memory_after_round(self):
        """Log memory state after training round completes."""
        if not self.enabled:
            return
        gc.collect()
        
        stats = self.get_memory_stats()
        
        logger.info(
            f"[MEMORY] Trainer {self.trainer_id} Round {self.round_count} END: "
            f"RSS={stats['rss_mb']:.1f}MB (+{stats['rss_mb']-self.initial_memory:.1f}MB from start), "
            f"Tensors: {stats['torch_tensors']['total_count']} "
            f"(CPU: {stats['torch_tensors']['cpu_tensors']}/{stats['torch_tensors']['cpu_memory_mb']:.1f}MB, "
            f"GPU: {stats['torch_tensors']['gpu_tensors']}/{stats['torch_tensors']['gpu_memory_mb']:.1f}MB)"
        )
        
        self._detect_leaks()
    
    def _log_detailed_stats(self, stats: Dict[str, Any]):
        """Log detailed memory statistics."""
        logger.info(
            f"[MEMORY DETAIL] Trainer {self.trainer_id} Round {self.round_count}:"
        )
        logger.info(f"  RSS: {stats['rss_mb']:.1f} MB")
        logger.info(f"  VMS: {stats['vms_mb']:.1f} MB")
        logger.info(f"  Total Python objects: {stats['total_objects']}")
        logger.info(f"  Top object types:")
        for obj_type, count in stats['top_object_types'].items():
            logger.info(f"    {obj_type}: {count}")
        
        logger.info(f"  PyTorch tensors:")
        logger.info(f"    Total: {stats['torch_tensors']['total_count']}")
        logger.info(f"    CPU: {stats['torch_tensors']['cpu_tensors']} ({stats['torch_tensors']['cpu_memory_mb']:.1f} MB)")
        logger.info(f"    GPU: {stats['torch_tensors']['gpu_tensors']} ({stats['torch_tensors']['gpu_memory_mb']:.1f} MB)")
        
        if 'cuda_allocated_mb' in stats:
            logger.info(f"  CUDA memory:")
            logger.info(f"    Allocated: {stats['cuda_allocated_mb']:.1f} MB")
            logger.info(f"    Reserved: {stats['cuda_reserved_mb']:.1f} MB")
    
    def _detect_leaks(self):
        """Detect potential memory leaks by comparing recent rounds."""
        if len(self.round_memory) < 5:
            return
        
        # Compare last 5 rounds
        recent = self.round_memory[-5:]
        
        # Check if memory is consistently growing
        memory_trend = [r['stats']['rss_mb'] for r in recent]
        
        # Simple leak detection: memory grows every round
        is_growing = all(memory_trend[i] < memory_trend[i+1] 
                        for i in range(len(memory_trend)-1))
        
        if is_growing:
            growth_per_round = (memory_trend[-1] - memory_trend[0]) / 4
            logger.warning(
                f"[MEMORY LEAK WARNING] Trainer {self.trainer_id}: "
                f"Memory growing consistently: {growth_per_round:.1f} MB/round"
            )
            
            # Check which object types are growing
            if len(self.round_memory) >= 2:
                prev_types = recent[0]['stats']['top_object_types']
                curr_types = recent[-1]['stats']['top_object_types']
                
                logger.warning(f"  Object type growth over last 5 rounds:")
                for obj_type in curr_types:
                    prev_count = prev_types.get(obj_type, 0)
                    curr_count = curr_types[obj_type]
                    if curr_count > prev_count:
                        growth = curr_count - prev_count
                        logger.warning(f"    {obj_type}: +{growth} ({prev_count} -> {curr_count})")
    
    def log_component_memory(self, component_name: str, before_after: str = ""):
        """Log memory for a specific component (e.g., dataloader, model, optimizer)."""
        if not self.enabled:
            return
        stats = self.get_memory_stats()
        prefix = f"{before_after} " if before_after else ""
        
        logger.info(
            f"[MEMORY COMPONENT] Trainer {self.trainer_id} {prefix}{component_name}: "
            f"RSS={stats['rss_mb']:.1f}MB, "
            f"CPU Tensors={stats['torch_tensors']['cpu_tensors']}/{stats['torch_tensors']['cpu_memory_mb']:.1f}MB"
        )
    
    def get_dataloader_memory(self, dataloader) -> Dict[str, Any]:
        """Analyze DataLoader memory usage."""
        memory_info = {
            'dataset_size': len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 0,
            'num_workers': dataloader.num_workers if hasattr(dataloader, 'num_workers') else 0,
            'batch_size': dataloader.batch_size if hasattr(dataloader, 'batch_size') else 0,
        }
        
        # Count tensors held by dataloader
        dataloader_tensors = 0
        try:
            for obj in gc.get_referents(dataloader):
                if torch.is_tensor(obj):
                    dataloader_tensors += 1
        except:
            pass
        
        memory_info['held_tensors'] = dataloader_tensors
        
        return memory_info
    
    def analyze_model_memory(self, model) -> Dict[str, Any]:
        """Analyze model memory usage."""
        info = {
            'total_params': 0,
            'trainable_params': 0,
            'param_memory_mb': 0,
        }
        
        for param in model.parameters():
            param_count = param.numel()
            info['total_params'] += param_count
            if param.requires_grad:
                info['trainable_params'] += param_count
            info['param_memory_mb'] += param.element_size() * param_count / 1024 / 1024
        
        return info
    
    def generate_report(self) -> str:
        """Generate a comprehensive memory analysis report."""
        if len(self.round_memory) < 2:
            return "Insufficient data for report"
        
        first = self.round_memory[0]['stats']
        last = self.round_memory[-1]['stats']
        
        report_lines = [
            f"\n{'='*80}",
            f"MEMORY ANALYSIS REPORT - Trainer {self.trainer_id}",
            f"{'='*80}",
            f"Rounds analyzed: {self.round_count}",
            f"",
            f"Memory Growth:",
            f"  Initial RSS: {first['rss_mb']:.1f} MB",
            f"  Final RSS: {last['rss_mb']:.1f} MB",
            f"  Growth: {last['rss_mb'] - first['rss_mb']:.1f} MB",
            f"  Growth per round: {(last['rss_mb'] - first['rss_mb']) / self.round_count:.2f} MB",
            f"",
            f"Object Counts:",
            f"  Initial: {first['total_objects']}",
            f"  Final: {last['total_objects']}",
            f"  Growth: {last['total_objects'] - first['total_objects']}",
            f"",
            f"Tensor Usage:",
            f"  Initial CPU tensors: {first['torch_tensors']['cpu_tensors']} ({first['torch_tensors']['cpu_memory_mb']:.1f} MB)",
            f"  Final CPU tensors: {last['torch_tensors']['cpu_tensors']} ({last['torch_tensors']['cpu_memory_mb']:.1f} MB)",
            f"  Growth: +{last['torch_tensors']['cpu_tensors'] - first['torch_tensors']['cpu_tensors']} tensors "
            f"(+{last['torch_tensors']['cpu_memory_mb'] - first['torch_tensors']['cpu_memory_mb']:.1f} MB)",
            f"",
            f"Top Growing Object Types:",
        ]
        
        # Calculate object type growth
        first_types = first['top_object_types']
        last_types = last['top_object_types']
        
        growth_by_type = []
        all_types = set(first_types.keys()) | set(last_types.keys())
        for obj_type in all_types:
            first_count = first_types.get(obj_type, 0)
            last_count = last_types.get(obj_type, 0)
            growth = last_count - first_count
            if growth > 0:
                growth_by_type.append((obj_type, growth, first_count, last_count))
        
        growth_by_type.sort(key=lambda x: x[1], reverse=True)
        
        for obj_type, growth, first_count, last_count in growth_by_type[:10]:
            report_lines.append(
                f"  {obj_type}: +{growth} ({first_count} -> {last_count})"
            )
        
        report_lines.append(f"{'='*80}\n")
        
        return "\n".join(report_lines)
