#!/usr/bin/env python3
"""
cacheflow_phase_a.py

Phase-A CacheFlow prototype:
- Extends a Sparse MoE model by injecting a CacheEngine.
- Adds a cache-aware TokenScheduler in the master process.
- Spawns multiple slave processes that host experts and execute forward passes.
- Benchmarks throughput and latency for synthetic concurrent requests.

Requirements:
- Keep model.py, expert.py and any weights in the same folder as this script
- Uses multiprocessing + asyncio to simulate distributed execution on one machine
"""

import asyncio
import multiprocessing as mp
import numpy as np
import time
import logging
import os
import statistics
from collections import OrderedDict as LinkedDict
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

# Try to import user model pieces. If they don't exist, we create minimal stubs.
try:
    from model import SparseMoELanguageModel
    from expert import Expert
    from config import N_EMBED, DEVICE, NUM_EXPERTS, MODEL_PATH
except Exception:
    # Minimal stubs to allow running without full repo (for demonstration).
    class Expert(nn.Module):
        def __init__(self, dim=128):
            super().__init__()
            self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

        def forward(self, x):
            return self.ffn(x)

    class SparseMoELanguageModel(nn.Module):
        def __init__(self, vocab_size=65, n_experts=8, hidden=128):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden = hidden
            # A simple embedding + "layer" with experts
            self.embed = nn.Embedding(vocab_size, hidden)
            # Experts stored in a ModuleList inside a higher-level container to be found by injector
            self.experts = nn.ModuleList([Expert(hidden) for _ in range(n_experts)])
            # Simple linear head
            self.head = nn.Linear(hidden, vocab_size)

        def forward(self, x_tokens):
            # x_tokens: (batch, seq)
            b, s = x_tokens.shape
            e = self.embed(x_tokens)  # (b, s, hidden)
            # call a simple router: randomly pick an expert per position for demo
            out = e
            for i in range(s):
                tok_vec = out[:, i, :]
                # simulate calling a single expert
                idx = int(torch.randint(0, len(self.experts), (1,)).item())
                tok_vec = self.experts[idx](tok_vec)
                out[:, i, :] = tok_vec
            # collapse seq dim
            last = out[:, -1, :]
            return self.head(last)

        def generate(self, context, max_new_tokens=10):
            # naive autoregressive generation for demo
            generated = []
            input_tokens = context
            for _ in range(max_new_tokens):
                logits = self.forward(input_tokens)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated.append(int(next_token.item()))
                input_tokens = torch.cat([input_tokens, next_token], dim=1)
            return generated

# --------------------------
# Logging
# --------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CacheFlowA")

# --------------------------
# Expert Cache Engine (improved)
# --------------------------
ACTIVE_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU_EXPERT_CAPACITY = int(os.environ.get("GPU_EXPERT_CAPACITY", "2"))

class ExpertCacheEngine:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cpu_registry: Dict[str, Dict[str, torch.Tensor]] = {}
        self.gpu_cache: LinkedDict = LinkedDict()
        self.hits = 0
        self.misses = 0
        self.lock = mp.Lock()

    def register_expert(self, expert_uid: str, expert_module: nn.Module):
        cpu_state = {}
        for k, v in expert_module.state_dict().items():
            t = v.detach().cpu()
            if torch.cuda.is_available():
                t = t.pin_memory()
            cpu_state[k] = t
        self.cpu_registry[expert_uid] = cpu_state
        # remove original params to reduce memory
        try:
            expert_module.to('meta')
        except Exception:
            expert_module.to('cpu')
            for p in expert_module.parameters():
                p.data = torch.empty(0)

    def get_expert_state(self, expert_uid: str):
        # return the CPU state_dict for sending to slave
        return self.cpu_registry[expert_uid]

    # Note: GPU-side module instances live in slave processes for simulation.
    def note_hit(self):
        with self.lock:
            self.hits += 1

    def note_miss(self):
        with self.lock:
            self.misses += 1

# --------------------------
# CacheFlow Expert Wrapper for master-side routing introspection (lightweight)
# --------------------------
class CacheFlowExpertWrapper:
    def __init__(self, expert_uid: str, engine: ExpertCacheEngine):
        self.expert_uid = expert_uid
        self.engine = engine

# --------------------------
# Token Scheduler (cache-aware)
# --------------------------
class TokenRequest:
    def __init__(self, req_id: str, token_tensor: torch.Tensor, timestamp: float):
        self.req_id = req_id
        self.token_tensor = token_tensor
        self.timestamp = timestamp
        # assigned expert will be set by router
        self.assigned_expert = None
        self.enqueue_time = time.perf_counter()

class TokenScheduler:
    """
    Priority queue that prefers tokens whose target expert is cached in GPU,
    uses wait-time bias to avoid starvation.
    """
    def __init__(self, engine: ExpertCacheEngine):
        self.engine = engine
        self.queue: List[TokenRequest] = []
        self.lock = asyncio.Lock()

    async def push(self, req: TokenRequest):
        async with self.lock:
            self.queue.append(req)

    async def pop_batch(self, max_batch: int = 8) -> List[TokenRequest]:
        async with self.lock:
            if not self.queue:
                return []
            # score tokens: cache_hit -> higher score; older -> higher
            scored = []
            now = time.perf_counter()
            for req in self.queue:
                # router assigned_expert should be set; if not, we pick random later
                uid = req.assigned_expert or "unknown"
                cache_score = 1.0 if uid in self.engine.cpu_registry and uid in self.engine.gpu_cache else 0.0
                wait_seconds = now - req.enqueue_time
                score = cache_score * 2.0 + min(wait_seconds, 2.0)
                scored.append((score, req))
            scored.sort(key=lambda x: -x[0])
            take = scored[:max_batch]
            selected = [t for s, t in take]
            # remove selected from queue
            remaining = [t for s, t in scored[max_batch:]]
            self.queue = [t for s, t in remaining]
            return selected

# --------------------------
# Master process: routing + scheduling + dispatch
# --------------------------
def master_process_main(num_slaves=2, num_experts=8, concurrency=8, max_tokens=10):
    logger.info("Master: starting")
    # Build model structure (master only keeps structure)
    model = SparseMoELanguageModel(vocab_size=65)
    # find experts inside model and register them into engine
    engine = ExpertCacheEngine(capacity=GPU_EXPERT_CAPACITY)
    expert_uids = []
    for i, ex in enumerate([m for m in model.experts]):
        uid = f"expert_{i}"
        engine.register_expert(uid, ex)
        expert_uids.append(uid)

    # Create inter-process queues
    manager = mp.Manager()
    task_queues = [manager.Queue(maxsize=1024) for _ in range(num_slaves)]
    result_queue = manager.Queue(maxsize=4096)

    # Spawn slave processes
    slaves = []
    slave_handles = []
    for si in range(num_slaves):
        p = mp.Process(target=slave_process_entrypoint, args=(si, task_queues[si], result_queue))
        p.start()
        slaves.append(p)

    # Cache-aware scheduler
    scheduler = TokenScheduler(engine)

    # simple router: picks most-likely expert (here round-robin for demo), but marks assigned_expert
    rr = 0

    async def produce_requests(n_requests=50):
        nonlocal rr
        for rid in range(n_requests):
            # simulate a short context token
            token = torch.zeros((1, 1), dtype=torch.long)
            # pick an expert id
            assigned = expert_uids[rr % len(expert_uids)]
            rr += 1
            req = TokenRequest(req_id=str(rid), token_tensor=token, timestamp=time.time())
            req.assigned_expert = assigned
            await scheduler.push(req)
            await asyncio.sleep(0.001)  # tiny inter-arrival

    async def dispatch_loop():
        latencies = []
        start_time = time.perf_counter()
        sent = 0
        completed = 0
        # dispatch until a number of completions
        target_completions = 200
        while completed < target_completions:
            batch = await scheduler.pop_batch(max_batch=8)
            if not batch:
                await asyncio.sleep(0.002)
                continue
            # for each request, choose a slave based on assigned_expert (simple hash)
            for req in batch:
                assigned = req.assigned_expert or expert_uids[0]
                # simple mapping: hash to slave index
                si = (int(assigned.split("_")[-1]) % num_slaves)
                # prepare payload: serialize token to numpy bytes + expert uid + req id + enqueue time
                token_np = req.token_tensor.numpy()
                payload = {
                    "req_id": req.req_id,
                    "expert_uid": assigned,
                    "token_shape": token_np.shape,
                    "token_bytes": token_np.tobytes(),
                    "enqueue_time": req.enqueue_time,
                }
                try:
                    task_queues[si].put(payload, block=False)
                    sent += 1
                except Exception:
                    # backpressure: push back into scheduler
                    await scheduler.push(req)
            # collect results available
            while not result_queue.empty():
                res = result_queue.get()
                completed += 1
                latency = time.perf_counter() - res["start_time"]
                latencies.append(latency)
                # update engine metrics
                # if result was cache hit, note it (slave reports)
                if res.get("cache_hit"):
                    engine.note_hit()
                else:
                    engine.note_miss()
            await asyncio.sleep(0.001)

        total_time = time.perf_counter() - start_time
        # compute metrics
        tokens_served = len(latencies)
        tps = tokens_served / total_time if total_time > 0 else 0.0
        p50 = np.percentile(latencies, 50) if latencies else 0.0
        p95 = np.percentile(latencies, 95) if latencies else 0.0
        p99 = np.percentile(latencies, 99) if latencies else 0.0

        logger.info("=== BENCHMARK RESULTS ===")
        logger.info(f"Total requests completed: {tokens_served}")
        logger.info(f"Total time: {total_time:.3f}s")
        logger.info(f"Throughput (req/s): {tps:.2f}")
        logger.info(f"P50 latency: {p50*1000:.2f} ms")
        logger.info(f"P95 latency: {p95*1000:.2f} ms")
        logger.info(f"P99 latency: {p99*1000:.2f} ms")
        logger.info(f"Cache hits: {engine.hits}, misses: {engine.misses}")
        # shutdown slaves
        for q in task_queues:
            q.put({"cmd": "shutdown"})
        for p in slaves:
            p.join(timeout=3.0)
        logger.info("Master: finished")

    async def main_async():
        # produce + dispatch concurrently
        prod = asyncio.create_task(produce_requests(n_requests=400))
        disp = asyncio.create_task(dispatch_loop())
        await prod
        await disp

    asyncio.run(main_async())

# --------------------------
# Slave process: loads experts on demand and executes
# --------------------------
def slave_process_entrypoint(slave_idx: int, task_queue, result_queue):
    # Each slave holds its own GPU cache dict mapping uid -> module
    logger = logging.getLogger(f"Slave-{slave_idx}")
    logger.setLevel(logging.INFO)
    # Local GPU cache simulation: map uid -> module instance on ACTIVE_DEVICE
    local_gpu_cache: Dict[str, nn.Module] = {}
    local_capacity = GPU_EXPERT_CAPACITY
    # simple LRU order
    lru_order: List[str] = []

    # minimal model structure for materialization of an expert
    # We assume the expert module class exists in expert.py or stub above
    def materialize_expert_from_state(state_dict):
        # create an Expert and load state
        e = Expert()
        try:
            e.to(ACTIVE_DEVICE)
            # convert pinned CPU tensors back
            st = {}
            for k, v in state_dict.items():
                st[k] = v.to(ACTIVE_DEVICE)
            e.load_state_dict(st, strict=False)
        except Exception:
            # fallback cpu
            e.to('cpu')
            st = {}
            for k, v in state_dict.items():
                st[k] = v
            e.load_state_dict(st, strict=False)
        e.eval()
        return e

    running = True
    # For this prototype we don't have a central engine in slave; master sends full state on first miss
    while running:
        task = task_queue.get()
        if not isinstance(task, dict):
            continue
        if task.get("cmd") == "shutdown":
            running = False
            break
        req_id = task["req_id"]
        expert_uid = task["expert_uid"]
        token_shape = task["token_shape"]
        token_bytes = task["token_bytes"]
        enqueue_time = task.get("enqueue_time", time.perf_counter())
        # quick token reconstruction
        token_np = np.frombuffer(token_bytes, dtype=np.int64).reshape(token_shape)
        token_tensor = torch.from_numpy(token_np).long()
        # Check cache
        cache_hit = expert_uid in local_gpu_cache
        start_proc = time.perf_counter()
        if not cache_hit:
            # For prototype, request master to send state could be added.
            # We'll simulate loading by constructing a new expert (random init) and materializing it.
            state = None
            # In a real integrated system, master would have sent CPU state; here we create on-the-fly
            e = Expert()
            try:
                e.to(ACTIVE_DEVICE)
            except Exception:
                e.to('cpu')
            local_gpu_cache[expert_uid] = e
            lru_order.append(expert_uid)
            # Evict if over capacity
            if len(local_gpu_cache) > local_capacity:
                evict = lru_order.pop(0)
                del local_gpu_cache[evict]
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        # Execute
        expert_mod = local_gpu_cache[expert_uid]
        # For simplicity, convert token to float feature vector
        token_feat = torch.randn((1, 128), device=expert_mod.ffn[0].weight.device)  # synthetic
        # run forward
        try:
            out = expert_mod(token_feat)
        except Exception:
            out = expert_mod(token_feat.cpu())
        # reply with timing + cache_hit flag
        result = {"req_id": req_id, "start_time": enqueue_time, "cache_hit": cache_hit}
        result_queue.put(result)
    logger.info(f"Slave-{slave_idx} exiting")

# --------------------------
# Entry point
# --------------------------
if __name__ == "__main__":
    # run the master process main (simulates the whole Phase-A prototype)
    master_process_main(num_slaves=2, num_experts=8, concurrency=8, max_tokens=10)
