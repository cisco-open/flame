# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""
REFL (Resource-Efficient Federated Learning) optimizer.

This implementation follows the original REFL paper and codebase:
https://github.com/ahmedcs/REFL
Paper: "REFL: Resource-Efficient Federated Learning" (EuroSys 2023)

Key features:
- Deadline-based straggler identification  
- Stale update caching and lifecycle management
- Multiple staleness weighting strategies (Equal, AdaSGD, DynSGD, REFL)
- Direct in-place model parameter updates
- Optional gradient policies: FedAvg (default), YoGi, QFedAvg

Note: Despite the historical filename, this is NOT FedAvg. It implements REFL's
staleness-aware aggregation, which can optionally use YoGi or QFedAvg for
post-aggregation gradient adjustment.
"""

import logging
import math
from typing import Dict, List, Optional
import numpy as np
import gc

from diskcache import Cache

from ..common.typing import ModelWeights
from ..common.util import MLFramework, get_ml_framework_in_use, valid_frameworks
from .abstract import AbstractOptimizer
from .regularizer.default import Regularizer
from .train_result import TrainResult

logger = logging.getLogger(__name__)


class REFL(AbstractOptimizer):
    """
    REFL (Resource-Efficient Federated Learning) optimizer.

    Implements REFL's staleness-aware aggregation with optional gradient policies:
    - Deadline-based filtering to identify slow trainers (stragglers)
    - Stale update caching and lifecycle management
    - Multiple staleness weighting strategies (Equal, AdaSGD, DynSGD, REFL)
    """

    def __init__(self, **kwargs):
        """
        Initialize REFL optimizer.
        
        Parameters:
        -----------
        Deadline & Staleness:
        - deadline: Fixed deadline in seconds (0 = use moving average)
        - initial_deadline: Initial value for moving average deadline
        - target_ratio: Target fraction of clients to complete (for moving avg)
        - stale_update: Max staleness in rounds (-1 = unlimited, 0 = no stale updates)
        - stale_factor: Staleness weighting (1=equal, -2=AdaSGD, -3=DynSGD, -4=REFL)
        - stale_beta: REFL beta parameter (default 0.35 per paper)
        - scale_coff: REFL scaling coefficient (default 10.0)
        
        Gradient Policies (applied after aggregation):
        - gradient_policy: None (default), "yogi", or "qfedavg"
        - yogi_eta: YoGi learning rate (default 0.003)
        - yogi_tau: YoGi tau parameter (default 0.001)
        - yogi_beta: YoGi beta momentum (default 0.9)
        - yogi_beta2: YoGi beta2 momentum (default 0.99)
        - qfed_q: QFedAvg q parameter (default 0.2)
        """
        super().__init__(**kwargs)

        self.agg_weights = None
        self.model_in_update = []  # Track if first update for initialization

        ml_framework_in_use = get_ml_framework_in_use()
        if ml_framework_in_use == MLFramework.PYTORCH:
            self.aggregate_fn = self._aggregate_pytorch
        elif ml_framework_in_use == MLFramework.TENSORFLOW:
            self.aggregate_fn = self._aggregate_tensorflow
        else:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}"
            )

        self.regularizer = Regularizer()

        # REFL deadline parameters
        self.deadline = kwargs.get("deadline", 0)  # 0 = use moving average
        self.mov_avg_deadline = kwargs.get("initial_deadline", 30.0)
        self.target_ratio = kwargs.get(
            "target_ratio", 0.8
        )  # Target fraction to wait for

        # Stale update parameters
        self.stale_update_max = kwargs.get("stale_update", -1)  # -1 = no limit
        self.stale_factor = kwargs.get(
            "stale_factor", 1
        )  # 1=equal, -2=AdaSGD, -3=DynSGD, -4=REFL
        # NOTE: REFL paper uses beta=0.35, not 0.9!
        self.stale_beta = kwargs.get("stale_beta", 0.35)  # REFL beta parameter
        self.scale_coff = kwargs.get("scale_coff", 10.0)  # REFL scaling coefficient

        # Gradient policy (applied after aggregation)
        self.gradient_policy = kwargs.get("gradient_policy", None)  # None, "yogi", "qfedavg"
        
        # YoGi parameters (if gradient_policy == "yogi")
        # NOTE: tau=1e-8 matches third_party/REFL (NOT 1e-3)
        self.yogi_eta = kwargs.get("yogi_eta", 0.003)
        self.yogi_tau = kwargs.get("yogi_tau", 1e-8)  # CRITICAL: Changed from 0.001 to 1e-8
        self.yogi_beta = kwargs.get("yogi_beta", 0.9)
        self.yogi_beta2 = kwargs.get("yogi_beta2", 0.99)
        
        # QFedAvg parameters (if gradient_policy == "qfedavg")
        self.qfed_q = kwargs.get("qfed_q", 0.2)
        self.qfed_learning_rate = kwargs.get("learning_rate", 0.1)  # For QFedAvg
        
        # Gradient controller state
        self.gradient_controller = None
        self.last_global_model = None
        
        if self.gradient_policy == "yogi":
            self.gradient_controller = self._init_yogi_controller()

        # Stale update storage
        self.stale_weights: Dict[str, ModelWeights] = {}
        self.stale_remain_duration: Dict[str, float] = {}
        self.stale_rounds: Dict[str, int] = {}
        self.stale_stat_utility: Dict[str, float] = {}
        self.stale_count: Dict[str, int] = {}

        # Statistics
        self.round_deadline_history = []
        self.stale_applied_count = 0
        self.stale_discarded_count = 0

        logger.info(
            f"REFL optimizer initialized (matching original REFL implementation): "
            f"deadline={self.deadline}, stale_update_max={self.stale_update_max}, "
            f"stale_factor={self.stale_factor}, stale_beta={self.stale_beta}, "
            f"scale_coff={self.scale_coff}, gradient_policy={self.gradient_policy}"
        )

    def do(
        self,
        base_weights: ModelWeights,
        cache: Cache,
        *,
        total: int = 0,
        version: int = 0,
        **kwargs,
    ) -> ModelWeights:
        """
        Aggregate models using REFL's approach: direct in-place parameter updates.
        
        This matches the original REFL implementation where:
        1. Each trainer's weights are multiplied by their importance weight
        2. Updates are accumulated directly into param.data
        3. Final normalization by sum of importance weights
        
        Parameters
        ----------
        base_weights: Base weights for aggregation
        cache: Cache containing training results
        total: Total number of data samples
        version: Model version number
        **kwargs: Additional arguments including:
            - round_duration: Duration of the round (for deadline filtering)
            - cur_time: Current virtual time (for stale lifecycle)

        Returns
        -------
        Aggregated model weights
        """
        logger.debug("Calling REFL FedAvg aggregation")

        assert base_weights is not None

        # Initialize aggregated weights from base_weights
        # CRITICAL: Trainers send DELTA weights (current - previous)
        # We must accumulate weighted deltas separately, then add to base
        self.agg_weights = self._copy_weights(base_weights)
        
        # Accumulate weighted deltas separately (don't modify base yet)
        self.weighted_deltas = self._zero_weights(base_weights)

        if len(cache) == 0 or total == 0:
            return base_weights

        # Get round metadata
        round_duration = kwargs.get("round_duration", 0)
        cur_time = kwargs.get("cur_time", 0)

        # Collect all training results from cache
        all_results = []
        for k in list(cache.iterkeys()):
            tres = cache.pop(k)
            all_results.append(tres)

        if not all_results:
            return base_weights

        # Log staleness distribution
        staleness_dist = [tres.staleness if tres.staleness else 0 for tres in all_results]
        logger.info(
            f"[REFL_AGG] Round aggregation with {len(all_results)} trainers: "
            f"staleness: min={min(staleness_dist)}, max={max(staleness_dist)}, "
            f"avg={sum(staleness_dist)/len(staleness_dist):.2f}"
        )

        # Apply deadline filtering
        fast_results, slow_results = self.filter_by_deadline(
            all_results, round_duration
        )

        logger.info(
            f"[REFL_DEADLINE] {len(fast_results)} fast, {len(slow_results)} slow "
            f"(deadline={self.get_effective_deadline():.2f}s)"
        )

        # Cache stale updates from slow trainers
        self.cache_stale_updates(slow_results, round_duration)

        # Get applicable stale updates from previous rounds
        applicable_stale = self.get_applicable_stale_updates(round_duration)

        logger.info(
            f"[REFL_STALE] {len(applicable_stale)} applicable stale updates, "
            f"{len(self.stale_weights)} still cached"
        )

        # Combine fast trainers + applicable stale updates
        all_trainers = fast_results + applicable_stale
        tasks_round = len(fast_results)  # Number of new (non-stale) trainers

        if not all_trainers:
            logger.warning("No trainers to aggregate (all filtered or cached)")
            return base_weights

        # Compute importance weights for all trainers
        # This must happen BEFORE aggregation for REFL method (needs current model state)
        importance_weights = self.compute_importance_weights(
            all_trainers, total, base_weights, tasks_round
        )

        # Log a sample weight BEFORE aggregation for debugging
        sample_key = next(iter(base_weights.keys())) if base_weights else None
        if sample_key:
            base_sample = base_weights[sample_key].flatten()[:3] if hasattr(base_weights[sample_key], 'flatten') else base_weights[sample_key][:3]
            logger.debug(f"[REFL_DEBUG] BEFORE aggregation - base_weights[{sample_key}][:3] = {base_sample}")

        # Aggregate using REFL's staleness-aware approach
        # CRITICAL: Accumulate weighted deltas, then normalize, then add to base
        importance_sum = 0.0
        for tres in all_trainers:
            end_id = tres.end_id if tres.end_id else "unknown"
            importance = importance_weights.get(end_id, 1.0)
            
            # Accumulate weighted deltas (NOT into base model yet!)
            self.aggregate_fn(tres, importance)
            importance_sum += importance

        # Log a sample weight AFTER accumulation BEFORE normalization
        if sample_key:
            delta_sample = self.weighted_deltas[sample_key].flatten()[:3] if hasattr(self.weighted_deltas[sample_key], 'flatten') else self.weighted_deltas[sample_key][:3]
            logger.debug(f"[REFL_DEBUG] Accumulated weighted deltas[{sample_key}][:3] = {delta_sample}, importance_sum={importance_sum:.4f}")

        # Normalize ONLY the deltas (not the base model!)
        self._normalize_by_importance(self.weighted_deltas, importance_sum)
        
        # Log normalized deltas
        if sample_key:
            norm_delta_sample = self.weighted_deltas[sample_key].flatten()[:3] if hasattr(self.weighted_deltas[sample_key], 'flatten') else self.weighted_deltas[sample_key][:3]
            logger.debug(f"[REFL_DEBUG] Normalized deltas[{sample_key}][:3] = {norm_delta_sample}")
        
        # Add normalized deltas to base model to get final weights
        # This is the correct formula: new_model = base + normalized_deltas
        self._add_deltas_to_base(self.agg_weights, self.weighted_deltas)
        
        # Log final weights
        if sample_key:
            final_sample = self.agg_weights[sample_key].flatten()[:3] if hasattr(self.agg_weights[sample_key], 'flatten') else self.agg_weights[sample_key][:3]
            logger.debug(f"[REFL_DEBUG] Final weights[{sample_key}][:3] = {final_sample}")

        # Apply gradient policy if specified (YoGi or QFedAvg)
        if self.gradient_policy and self.last_global_model is not None:
            self.agg_weights = self._apply_gradient_policy(
                self.last_global_model, self.agg_weights, all_trainers
            )
        
        # Save current model for next round's gradient policy
        self.last_global_model = self._copy_weights(self.agg_weights)

        # Update moving average deadline if using dynamic deadline
        if self.deadline <= 0:
            self.update_moving_avg_deadline(fast_results)

        logger.info(
            f"[REFL_AGG] Completed aggregation: {len(all_trainers)} trainers "
            f"({tasks_round} new + {len(applicable_stale)} stale), "
            f"importance_sum={importance_sum:.4f}, gradient_policy={self.gradient_policy}"
        )

        return self.agg_weights

    def filter_by_deadline(
        self, results: List[TrainResult], round_duration: float
    ) -> tuple[List[TrainResult], List[TrainResult]]:
        """
        Filter training results into fast and slow based on deadline.

        Args:
            results: List of all training results
            round_duration: Actual duration of the round

        Returns:
            Tuple of (fast_results, slow_results)
        """
        if self.deadline <= 0 and self.mov_avg_deadline <= 0:
            # No deadline filtering
            return results, []

        effective_deadline = self.get_effective_deadline()

        fast = []
        slow = []

        for tres in results:
            # Use completion_time if available, otherwise use round_duration as estimate
            trainer_duration = (
                tres.round_duration if tres.round_duration else round_duration
            )

            if trainer_duration <= effective_deadline:
                fast.append(tres)
            else:
                slow.append(tres)

        return fast, slow

    def get_effective_deadline(self) -> float:
        """Get the effective deadline (fixed or moving average)."""
        if self.deadline > 0:
            return self.deadline
        elif self.mov_avg_deadline > 0:
            return self.mov_avg_deadline
        else:
            return float("inf")

    def cache_stale_updates(
        self, slow_results: List[TrainResult], round_duration: float
    ) -> None:
        """
        Cache stale updates from slow trainers for potential future use.

        Args:
            slow_results: Training results from slow trainers
            round_duration: Duration of the current round
        """
        for tres in slow_results:
            end_id = tres.end_id if tres.end_id else "unknown"

            # Store stale update
            self.stale_weights[end_id] = tres.weights

            # Calculate remaining duration based on how late trainer was
            trainer_duration = (
                tres.round_duration if tres.round_duration else round_duration
            )
            self.stale_remain_duration[end_id] = (
                trainer_duration - self.get_effective_deadline()
            )

            # Initialize staleness counter
            self.stale_rounds[end_id] = 0

            # Store statistical utility for REFL weighting
            self.stale_stat_utility[end_id] = (
                tres.stat_utility if tres.stat_utility else 0.0
            )

            # Store sample count
            self.stale_count[end_id] = tres.count

            logger.debug(
                f"Cached stale update from {end_id}: "
                f"remaining_duration={self.stale_remain_duration[end_id]:.2f}s, "
                f"stat_utility={self.stale_stat_utility[end_id]:.4f}"
            )

    def get_applicable_stale_updates(self, round_duration: float) -> List[TrainResult]:
        """
        Get stale updates that are now ready to apply.

        Updates their staleness and removes expired updates.

        Args:
            round_duration: Duration of current round

        Returns:
            List of TrainResult objects from stale cache
        """
        applicable = []
        expired_ids = []

        for end_id in list(self.stale_weights.keys()):
            # Decrement remaining duration
            self.stale_remain_duration[end_id] -= round_duration

            # Increment staleness counter
            self.stale_rounds[end_id] += 1

            # Check if ready to apply
            if self.stale_remain_duration[end_id] <= 0:
                # Check if not too stale
                if (
                    self.stale_update_max < 0
                    or self.stale_rounds[end_id] <= self.stale_update_max
                ):
                    # Apply this stale update
                    tres = TrainResult(
                        weights=self.stale_weights[end_id],
                        count=self.stale_count[end_id],
                        stat_utility=self.stale_stat_utility[end_id],
                        staleness=self.stale_rounds[end_id],
                        end_id=end_id,
                    )
                    applicable.append(tres)
                    self.stale_applied_count += 1

                    logger.debug(
                        f"Applying stale update from {end_id}: "
                        f"staleness={self.stale_rounds[end_id]} rounds"
                    )
                else:
                    # Too stale, discard
                    self.stale_discarded_count += 1
                    logger.debug(
                        f"Discarding stale update from {end_id}: "
                        f"staleness={self.stale_rounds[end_id]} > {self.stale_update_max}"
                    )

                # Remove from cache
                expired_ids.append(end_id)
            else:
                logger.debug(
                    f"Stale update from {end_id} still cached: "
                    f"remaining={self.stale_remain_duration[end_id]:.2f}s, "
                    f"staleness={self.stale_rounds[end_id]} rounds"
                )

        # Clean up expired entries
        for end_id in expired_ids:
            del self.stale_weights[end_id]
            del self.stale_remain_duration[end_id]
            del self.stale_rounds[end_id]
            del self.stale_stat_utility[end_id]
            del self.stale_count[end_id]

        # Force garbage collection to free memory
        if expired_ids:
            gc.collect()

        return applicable

    def compute_importance_weights(
        self,
        trainers: List[TrainResult],
        total: int,
        current_model: ModelWeights,
        tasks_round: int,
    ) -> Dict[str, float]:
        """
        Compute importance weights for trainers based on staleness strategy.
        
        Matches REFL's implementation including the complex client_ratio calculation
        for the REFL method (stale_factor == -4).

        Args:
            trainers: List of training results (fast + applicable stale)
            total: Total number of samples
            current_model: Current global model weights (needed for REFL method)
            tasks_round: Number of new (non-stale) trainers this round

        Returns:
            Dictionary mapping end_id to importance weight
        """
        weights = {}

        # Separate stale and non-stale trainers
        stale_trainers = [t for t in trainers if t.staleness and t.staleness > 0]
        
        # For REFL method: compute client_ratio for stale trainers
        client_ratios = {}
        if len(stale_trainers) > 0 and self.stale_factor == -4:
            client_ratios = self._compute_client_ratios(
                stale_trainers, current_model, tasks_round
            )
            
            max_ratio = max(client_ratios.values()) if client_ratios else 1.0
            logger.debug(f"[REFL_RATIO] Computed client ratios for {len(client_ratios)} stale trainers, max_ratio={max_ratio:.6f}")
        else:
            max_ratio = 1.0

        # Compute importance weight for each trainer
        for tres in trainers:
            end_id = tres.end_id if tres.end_id else "unknown"
            staleness = tres.staleness if tres.staleness else 0

            # Initialize with base weight of 1.0
            weight = 1.0

            # Apply staleness weighting only to stale trainers
            if staleness > 0:
                if self.stale_factor > 1:
                    # Divide by constant factor
                    weight = 1.0 / self.stale_factor

                elif self.stale_factor == 1:
                    # Equal weight (standard FedAvg)
                    weight = 1.0

                elif self.stale_factor == -1:
                    # Average: divide by average staleness
                    avg_staleness = np.mean([t.staleness for t in stale_trainers if t.staleness])
                    weight = 1.0 / max(avg_staleness, 1.0)

                elif self.stale_factor == -2:
                    # AdaSGD: divide by (staleness + 1)
                    weight = 1.0 / (staleness + 1)

                elif self.stale_factor == -3:
                    # DynSGD: multiply by exp(-(staleness + 1))
                    weight = math.exp(-(staleness + 1))

                elif self.stale_factor == -4:
                    # REFL: hybrid formula combining staleness and importance ratio
                    client_ratio = client_ratios.get(end_id, 0.0)
                    weight = (1 - self.stale_beta) / (staleness + 1) + self.stale_beta * (
                        1.0 - (math.exp(-client_ratio / max_ratio) / self.scale_coff)
                    )

                else:
                    logger.warning(
                        f"Unknown stale_factor={self.stale_factor}, using equal weight"
                    )
                    weight = 1.0

            weights[end_id] = weight

        # Log importance weights for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[REFL_WEIGHTS] Importance weights:")
            for end_id, weight in weights.items():
                staleness = next((t.staleness for t in trainers if t.end_id == end_id), 0)
                logger.debug(f"  ...{end_id[-8:]}: staleness={staleness}, weight={weight:.6f}")

        return weights

    def update_moving_avg_deadline(self, fast_results: List[TrainResult]) -> None:
        """
        Update moving average deadline based on fast trainers.

        Uses target_ratio percentile of completion times.

        Args:
            fast_results: Training results from fast trainers
        """
        if not fast_results:
            return

        # Get completion times
        durations = []
        for tres in fast_results:
            if tres.round_duration:
                durations.append(tres.round_duration)

        if not durations:
            return

        # Calculate target percentile
        durations.sort()
        target_idx = int(len(durations) * self.target_ratio)
        target_idx = min(target_idx, len(durations) - 1)

        new_deadline = durations[target_idx]

        # Update with exponential moving average (alpha = 0.3)
        alpha = 0.3
        self.mov_avg_deadline = (
            alpha * new_deadline + (1 - alpha) * self.mov_avg_deadline
        )

        self.round_deadline_history.append(self.mov_avg_deadline)

        logger.debug(
            f"Updated moving avg deadline: {self.mov_avg_deadline:.2f}s "
            f"(from {len(durations)} fast trainers, target_percentile={self.target_ratio})"
        )

    def _compute_client_ratios(
        self,
        stale_trainers: List[TrainResult],
        current_model: ModelWeights,
        tasks_round: int,
    ) -> Dict[str, float]:
        """
        Compute client ratios for REFL method (stale_factor == -4).
        
        This matches REFL's implementation:
        val1 = ||update / (tasks_round + 1) + param / (tasks_round + 1) - param / tasks_round||^2
        val2 = ||param / tasks_round||^2
        ratio = val1 / val2
        
        This measures how much the stale update changes the normalized model.
        
        Args:
            stale_trainers: List of stale training results
            current_model: Current global model weights
            tasks_round: Number of new (non-stale) trainers this round
            
        Returns:
            Dictionary mapping end_id to client ratio
        """
        ml_framework = get_ml_framework_in_use()
        client_ratios = {}
        
        if tasks_round == 0:
            # Avoid division by zero
            logger.warning("tasks_round is 0, using ratio of 1.0 for all stale trainers")
            return {t.end_id: 1.0 for t in stale_trainers if t.end_id}
        
        if ml_framework == MLFramework.PYTORCH:
            import torch
            
            for tres in stale_trainers:
                end_id = tres.end_id if tres.end_id else "unknown"
                val1 = 0.0
                val2 = 0.0
                
                for key in current_model.keys():
                    if key not in tres.weights:
                        continue
                        
                    param = current_model[key]
                    update = tres.weights[key]
                    
                    # Compute: update / (tasks_round + 1) + param / (tasks_round + 1) - param / tasks_round
                    term1 = update / (tasks_round + 1)
                    term2 = param / (tasks_round + 1)
                    term3 = param / tasks_round
                    diff = term1 + term2 - term3
                    
                    val1 += torch.norm(diff) ** 2
                    val2 += torch.norm(param / tasks_round) ** 2
                
                ratio = abs(float(val1 / val2)) if val2 > 0 else 1.0
                client_ratios[end_id] = ratio
                
        elif ml_framework == MLFramework.TENSORFLOW:
            import numpy as np
            
            for tres in stale_trainers:
                end_id = tres.end_id if tres.end_id else "unknown"
                val1 = 0.0
                val2 = 0.0
                
                for idx in range(min(len(current_model), len(tres.weights))):
                    param = current_model[idx]
                    update = tres.weights[idx]
                    
                    # Compute: update / (tasks_round + 1) + param / (tasks_round + 1) - param / tasks_round
                    term1 = update / (tasks_round + 1)
                    term2 = param / (tasks_round + 1)
                    term3 = param / tasks_round
                    diff = term1 + term2 - term3
                    
                    val1 += np.linalg.norm(diff) ** 2
                    val2 += np.linalg.norm(param / tasks_round) ** 2
                
                ratio = abs(float(val1 / val2)) if val2 > 0 else 1.0
                client_ratios[end_id] = ratio
        
        return client_ratios

    def _copy_weights(self, weights: ModelWeights) -> ModelWeights:
        """Create a deep copy of model weights."""
        ml_framework = get_ml_framework_in_use()
        
        if ml_framework == MLFramework.PYTORCH:
            import torch
            return {k: v.clone() for k, v in weights.items()}
        elif ml_framework == MLFramework.TENSORFLOW:
            import numpy as np
            return [np.copy(w) for w in weights]
        else:
            raise NotImplementedError(f"Unsupported framework: {ml_framework}")
    
    def _zero_weights(self, weights: ModelWeights) -> ModelWeights:
        """Create zero-initialized weights with same structure as input."""
        ml_framework = get_ml_framework_in_use()
        
        if ml_framework == MLFramework.PYTORCH:
            import torch
            return {k: torch.zeros_like(v) for k, v in weights.items()}
        elif ml_framework == MLFramework.TENSORFLOW:
            import numpy as np
            return [np.zeros_like(w) for w in weights]
        else:
            raise NotImplementedError(f"Unsupported framework: {ml_framework}")
    
    def _add_deltas_to_base(self, base: ModelWeights, deltas: ModelWeights) -> None:
        """Add deltas to base weights in-place. Formula: base += deltas"""
        ml_framework = get_ml_framework_in_use()
        
        if ml_framework == MLFramework.PYTORCH:
            for k in base.keys():
                if k in deltas:
                    base[k] += deltas[k]
        elif ml_framework == MLFramework.TENSORFLOW:
            for idx in range(len(base)):
                base[idx] += deltas[idx]
        else:
            raise NotImplementedError(f"Unsupported framework: {ml_framework}")

    def _normalize_by_importance(self, weights: ModelWeights, importance_sum: float) -> None:
        """
        Normalize model weights by sum of importance weights (REFL's approach).
        
        Args:
            weights: Model weights to normalize (modified in-place)
            importance_sum: Sum of all importance weights
        """
        if importance_sum == 0:
            logger.error("importance_sum is 0, cannot normalize")
            return
            
        ml_framework = get_ml_framework_in_use()
        
        if ml_framework == MLFramework.PYTORCH:
            for k in weights.keys():
                weights[k] = weights[k] / importance_sum
        elif ml_framework == MLFramework.TENSORFLOW:
            for idx in range(len(weights)):
                weights[idx] = weights[idx] / importance_sum
        else:
            raise NotImplementedError(f"Unsupported framework: {ml_framework}")

    def _aggregate_pytorch(self, tres: TrainResult, importance: float):
        """
        Accumulate PyTorch DELTA weights with importance weighting.
        
        CRITICAL: Trainers send DELTA weights (current - previous).
        We accumulate weighted deltas into self.weighted_deltas.
        The base model (self.agg_weights) remains unchanged until final step.
        
        Formula: weighted_deltas += delta * importance
        """
        # Accumulate weighted deltas (not into base model!)
        for k, v in tres.weights.items():
            if k in self.weighted_deltas:
                self.weighted_deltas[k] += v * importance
            else:
                logger.warning(f"Key {k} not found in weighted_deltas")

    def _aggregate_tensorflow(self, tres: TrainResult, importance: float):
        """
        Accumulate TensorFlow DELTA weights with importance weighting.
        
        CRITICAL: Trainers send DELTA weights (current - previous).
        We accumulate weighted deltas into self.weighted_deltas.
        The base model (self.agg_weights) remains unchanged until final step.
        
        Formula: weighted_deltas[idx] += delta[idx] * importance
        """
        # Accumulate weighted deltas (not into base model!)
        for idx in range(len(tres.weights)):
            self.weighted_deltas[idx] += tres.weights[idx] * importance

    def _init_yogi_controller(self):
        """Initialize YoGi controller for gradient policy."""
        return {
            'v_t': None,
            'delta_t': None,
            'eta': self.yogi_eta,
            'tau': self.yogi_tau,
            'beta': self.yogi_beta,
            'beta2': self.yogi_beta2,
        }

    def _apply_gradient_policy(
        self,
        last_model: ModelWeights,
        current_model: ModelWeights,
        trainers: List[TrainResult],
    ) -> ModelWeights:
        """
        Apply gradient policy (YoGi or QFedAvg) to model updates.
        
        This matches REFL's round_weight_handler behavior.
        
        Args:
            last_model: Model weights from previous round
            current_model: Aggregated model weights from current round
            trainers: List of training results (for QFedAvg)
            
        Returns:
            Adjusted model weights after applying gradient policy
        """
        if self.gradient_policy == "yogi":
            return self._apply_yogi(last_model, current_model)
        elif self.gradient_policy == "qfedavg":
            return self._apply_qfedavg(last_model, current_model, trainers)
        else:
            return current_model

    def _apply_yogi(
        self, last_model: ModelWeights, current_model: ModelWeights
    ) -> ModelWeights:
        """
        Apply YoGi adaptive learning rate to model update.
        
        Matches REFL's YoGi implementation:
        1. Compute diff = current_model - last_model
        2. Apply YoGi adaptive updates to diff
        3. Return last_model + adjusted_diff
        """
        ml_framework = get_ml_framework_in_use()
        
        if ml_framework == MLFramework.PYTORCH:
            import torch
            
            # Compute model difference
            diff = {k: current_model[k] - last_model[k] for k in current_model.keys()}
            
            # Initialize YoGi state if needed
            if self.gradient_controller['v_t'] is None:
                self.gradient_controller['v_t'] = {k: v ** 2 for k, v in diff.items()}
                self.gradient_controller['delta_t'] = {k: v.clone() for k, v in diff.items()}
                adjusted_diff = diff
            else:
                # Apply YoGi updates
                adjusted_diff = {}
                v_t = self.gradient_controller['v_t']
                delta_t = self.gradient_controller['delta_t']
                
                for k in diff.keys():
                    gradient = diff[k]
                    gradient_square = gradient ** 2
                    
                    # Update momentum
                    delta_t[k] = (
                        self.yogi_beta * delta_t[k] + (1.0 - self.yogi_beta) * gradient
                    )
                    
                    # Update adaptive learning rate (YoGi-specific)
                    v_t[k] = v_t[k] - (1.0 - self.yogi_beta2) * gradient_square * torch.sign(
                        v_t[k] - gradient_square
                    )
                    
                    # Apply adaptive learning rate
                    yogi_lr = self.yogi_eta / (torch.sqrt(v_t[k]) + self.yogi_tau)
                    adjusted_diff[k] = yogi_lr * delta_t[k]
            
            # Return last_model + adjusted_diff
            return {k: last_model[k] + adjusted_diff[k] for k in last_model.keys()}
            
        elif ml_framework == MLFramework.TENSORFLOW:
            import numpy as np
            
            # Compute model difference
            diff = [current_model[i] - last_model[i] for i in range(len(current_model))]
            
            # Initialize YoGi state if needed
            if self.gradient_controller['v_t'] is None:
                self.gradient_controller['v_t'] = [v ** 2 for v in diff]
                self.gradient_controller['delta_t'] = [np.copy(v) for v in diff]
                adjusted_diff = diff
            else:
                # Apply YoGi updates
                adjusted_diff = []
                v_t = self.gradient_controller['v_t']
                delta_t = self.gradient_controller['delta_t']
                
                for idx in range(len(diff)):
                    gradient = diff[idx]
                    gradient_square = gradient ** 2
                    
                    # Update momentum
                    delta_t[idx] = (
                        self.yogi_beta * delta_t[idx] + (1.0 - self.yogi_beta) * gradient
                    )
                    
                    # Update adaptive learning rate (YoGi-specific)
                    v_t[idx] = v_t[idx] - (1.0 - self.yogi_beta2) * gradient_square * np.sign(
                        v_t[idx] - gradient_square
                    )
                    
                    # Apply adaptive learning rate
                    yogi_lr = self.yogi_eta / (np.sqrt(v_t[idx]) + self.yogi_tau)
                    adjusted_diff.append(yogi_lr * delta_t[idx])
            
            # Return last_model + adjusted_diff
            return [last_model[i] + adjusted_diff[i] for i in range(len(last_model))]
        
        else:
            raise NotImplementedError(f"Unsupported framework: {ml_framework}")

    def _apply_qfedavg(
        self, last_model: ModelWeights, current_model: ModelWeights, trainers: List[TrainResult]
    ) -> ModelWeights:
        """
        Apply QFedAvg gradient adjustment.
        
        Matches REFL's QFedAvg implementation.
        """
        ml_framework = get_ml_framework_in_use()
        
        if ml_framework == MLFramework.PYTORCH:
            import torch
            
            Deltas = None
            hs = 0.0
            
            for tres in trainers:
                # Compute gradients from weight updates
                grads = [
                    (current_model[k] - tres.weights[k]) / self.qfed_learning_rate
                    for k in current_model.keys()
                ]
                
                loss = tres.loss if hasattr(tres, 'loss') else 1.0
                
                if Deltas is None:
                    Deltas = [
                        np.float_power(loss + 1e-10, self.qfed_q) * grad for grad in grads
                    ]
                else:
                    for idx, grad in enumerate(grads):
                        Deltas[idx] += np.float_power(loss + 1e-10, self.qfed_q) * grad
                
                # Lipschitz constant estimation
                hs += (
                    self.qfed_q * np.float_power(loss + 1e-10, self.qfed_q - 1)
                    * torch.sum(torch.stack([torch.square(grad).sum() for grad in grads]))
                    + (1.0 / self.qfed_learning_rate) * np.float_power(loss + 1e-10, self.qfed_q)
                )
            
            # Apply QFedAvg update
            result = {}
            for idx, k in enumerate(last_model.keys()):
                result[k] = last_model[k] - Deltas[idx] / (hs + 1e-10)
            
            return result
            
        elif ml_framework == MLFramework.TENSORFLOW:
            # Similar implementation for TensorFlow
            logger.warning("QFedAvg not fully implemented for TensorFlow, returning current_model")
            return current_model
        
        else:
            raise NotImplementedError(f"Unsupported framework: {ml_framework}")

    def scale_add_agg_weights(
        self, base_weights: ModelWeights, agg_goal_weights: ModelWeights, agg_goal: int
    ) -> ModelWeights:
        """
        Return aggregated weights directly (pass-through for REFL).
        
        REFL performs complete aggregation in its do() method, including:
        - Staleness-aware weighting
        - Importance-based aggregation
        - Gradient policy application (if configured)
        
        Unlike FedBuff/FedAvg which need post-aggregation scaling, REFL's do() 
        method returns fully aggregated weights ready to use as the global model.
        
        Parameters
        ----------
        base_weights: Current global weights (unused, REFL's do() already computed final weights)
        agg_goal_weights: Fully aggregated weights from REFL's do() method
        agg_goal: Aggregation goal (unused, already accounted for in do())
        
        Returns
        -------
        The aggregated weights (pass-through from do() method)
        """
        logger.debug(
            f"[REFL] scale_add_agg_weights: Returning pre-aggregated weights "
            f"(REFL's do() method already completed full aggregation)"
        )
        # REFL's do() method already returns the final global model weights
        # No additional scaling or learning rate adjustment needed
        return agg_goal_weights

    def get_statistics(self) -> Dict:
        """
        Get statistics about REFL aggregation.

        Returns:
            Dictionary with statistics
        """
        return {
            "stale_cached_count": len(self.stale_weights),
            "stale_applied_total": self.stale_applied_count,
            "stale_discarded_total": self.stale_discarded_count,
            "moving_avg_deadline": self.mov_avg_deadline,
            "deadline_history": self.round_deadline_history[-10:],  # Last 10 rounds
        }