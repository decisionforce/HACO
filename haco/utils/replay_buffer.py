from ray.rllib.execution.replay_buffer import LocalReplayBuffer, ReplayBuffer
import collections
import logging
import numpy as np
import platform
import random
from typing import List

import ray
from ray.rllib.execution.segment_tree import SumSegmentTree, MinSegmentTree
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, \
    DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import DeveloperAPI
from ray.util.iter import ParallelIteratorWorker
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.window_stat import WindowStat
from ray.rllib.utils.typing import SampleBatchType

# Constant that represents all policies in lockstep replay mode.
_ALL_POLICIES = "__all__"

logger = logging.getLogger(__name__)


class MyReplayBuffer(LocalReplayBuffer):
    """
    Use normal replay buffer instead of PrioritizedReplay
    """

    def __init__(self,
                 num_shards,
                 learning_starts,
                 buffer_size,
                 replay_batch_size,
                 prioritized_replay_alpha=0.6,
                 prioritized_replay_beta=0.4,
                 prioritized_replay_eps=1e-6,
                 replay_mode="independent",
                 replay_sequence_length=1):
        self.replay_starts = learning_starts // num_shards
        self.buffer_size = buffer_size // num_shards
        self.replay_batch_size = replay_batch_size
        self.prioritized_replay_beta = prioritized_replay_beta
        self.prioritized_replay_eps = prioritized_replay_eps
        self.replay_mode = replay_mode
        self.replay_sequence_length = replay_sequence_length

        if replay_sequence_length > 1:
            self.replay_batch_size = int(
                max(1, replay_batch_size // replay_sequence_length))
            logger.info(
                "Since replay_sequence_length={} and replay_batch_size={}, "
                "we will replay {} sequences at a time.".format(
                    replay_sequence_length, replay_batch_size,
                    self.replay_batch_size))

        if replay_mode not in ["lockstep", "independent"]:
            raise ValueError("Unsupported replay mode: {}".format(replay_mode))

        def gen_replay():
            while True:
                yield self.replay()

        ParallelIteratorWorker.__init__(self, gen_replay, False)

        def new_buffer():
            return ReplayBuffer(self.buffer_size)

        self.replay_buffers = collections.defaultdict(new_buffer)
        self.human_buffers = collections.defaultdict(new_buffer)

        # Metrics
        self.add_batch_timer = TimerStat()
        self.replay_timer = TimerStat()
        self.update_priorities_timer = TimerStat()
        self.num_added = 0

        # Make externally accessible for testing.
        global _local_replay_buffer
        _local_replay_buffer = self
        # If set, return this instead of the usual data for testing.
        self._fake_batch = None

    def add_batch(self, batch):
        # Make a copy so the replay buffer doesn't pin plasma memory.
        batch = batch.copy()
        # Handle everything as if multiagent
        if isinstance(batch, SampleBatch):
            batch = MultiAgentBatch({DEFAULT_POLICY_ID: batch}, batch.count)
        with self.add_batch_timer:
            for policy_id, b in batch.policy_batches.items():
                buffer = self.human_buffers if b["takeover"][0] else self.replay_buffers
                for s in b.timeslices(self.replay_sequence_length):
                    if "weights" in s:
                        weight = np.mean(s["weights"])
                    else:
                        weight = None
                    buffer[policy_id].add(s, weight=weight)
        self.num_added += batch.count

    def replay(self):
        if self._fake_batch:
            fake_batch = SampleBatch(self._fake_batch)
            return MultiAgentBatch({
                DEFAULT_POLICY_ID: fake_batch
            }, fake_batch.count)

        if self.num_added < self.replay_starts:
            return None
        num_human = len(self.human_buffers["default_policy"])
        num_agent = len(self.replay_buffers["default_policy"])

        bs_human = min(int(self.replay_batch_size/2), num_human)
        bs_agent = self.replay_batch_size - bs_human
        with self.replay_timer:
            samples = {}
            for policy_id, replay_buffer in self.replay_buffers.items():
                samples[policy_id] = replay_buffer.sample(bs_agent)
            for policy_id, replay_buffer in self.human_buffers.items():
                samples[policy_id] = samples[policy_id].concat(replay_buffer.sample(bs_human))
            return MultiAgentBatch(samples, self.replay_batch_size)
