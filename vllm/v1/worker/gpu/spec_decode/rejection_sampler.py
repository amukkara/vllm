# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np
import torch

from vllm.config import SpeculativeConfig
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    get_num_sampled_and_rejected,
)
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

# Cap on the FP32 target-logits buffer materialized by apply_sampling_params.
# TODO(mgoin): Chunking is a workaround. The rejection kernels already upcast
# per vocab block on load and apply ops like temperature and gumbel, so folding
# sampling-param application into those kernels would remove this buffer and
# its traffic entirely.
MAX_CHUNK_BYTES = 2**30  # 1GB
_FP32_BYTES = 4


def get_max_chunk_logits(vocab_size: int) -> int:
    """Largest number of logits rows one verification chunk may hold."""
    return max(1, MAX_CHUNK_BYTES // (vocab_size * _FP32_BYTES))


def _iter_request_chunks(
    cu_num_logits: np.ndarray, max_chunk_logits: int
) -> Iterator[tuple[int, int]]:
    """Yield maximally packed request ranges without splitting requests."""
    assert max_chunk_logits > 0
    num_reqs = cu_num_logits.size - 1
    start = 0
    while start < num_reqs:
        max_logit = int(cu_num_logits[start]) + max_chunk_logits
        end = int(np.searchsorted(cu_num_logits, max_logit, side="right") - 1)
        end = min(num_reqs, max(start + 1, end))
        yield start, end
        start = end


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


@triton.jit
def _gather_draft_sampled_kernel(
    # [num_logits]
    draft_sampled_ptr,
    # [num_logits]
    pos_ptr,
    # [num_tokens]
    input_ids_ptr,
    # [num_tokens]
    positions_ptr,
    # [num_logits]
    logits_indices_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    prefill_len_ptr,
    num_logits,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < num_logits
    token_idx = tl.load(logits_indices_ptr + block, mask=mask)
    token_id = tl.load(input_ids_ptr + token_idx, mask=mask)
    pos = tl.load(positions_ptr + token_idx, mask=mask)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + block, mask=mask)
    local_pos = tl.load(expanded_local_pos_ptr + block, mask=mask)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx, mask=mask)
    # Drafts are only proposed after sampling past the prefill, so any draft
    # slots in a step starting within the prefill are placeholders (e.g. the
    # padded first step after a P/D remote KV load).
    is_placeholder = (local_pos > 0) & (pos - local_pos < prefill_len)
    token_id = tl.where(is_placeholder, -1, token_id)
    tl.store(draft_sampled_ptr + block, token_id, mask=mask)
    tl.store(pos_ptr + block, pos, mask=mask)


def gather_draft_sampled(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    logits_indices: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the input token and position of each logits row.

    Draft rows of requests that have not yet sampled past their prefill are
    set to -1 so that the rejection kernels reject them.
    """
    num_logits = logits_indices.shape[0]
    draft_sampled = input_ids.new_empty(num_logits)
    pos = positions.new_empty(num_logits)
    BLOCK_SIZE = 1024
    _gather_draft_sampled_kernel[(triton.cdiv(num_logits, BLOCK_SIZE),)](
        draft_sampled,
        pos,
        input_ids,
        positions,
        logits_indices,
        expanded_idx_mapping,
        expanded_local_pos,
        prefill_len,
        num_logits,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return draft_sampled, pos


def vocab_parallel_top1(
    local_logits: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    vocab_size: int,
    all_gather: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy token and its logit per row from vocab-parallel logits.

    Same result as argmax / max over the all-gathered full-vocab logits (ties
    go to the lowest token id), but each rank reduces its own shard and only
    one (logit, token id) pair per row and rank is gathered.

    local_logits: [rows, vocab_per_rank], this rank's equal-size shard of the
    padded vocab. all_gather concatenates [rows, 2] along the last dim in
    rank order. Returns (token ids [rows] int64, logits [rows] float32).
    """
    shard = local_logits.shape[-1]
    vocab_start = tp_rank * shard
    num_valid = min(max(vocab_size - vocab_start, 0), shard)
    if num_valid < shard:
        # Vocab padding at the end of the last shard(s) is not a token.
        local_logits = local_logits.clone()
        local_logits[:, num_valid:] = float("-inf")
    values, indices = local_logits.max(dim=-1)
    values = values.float()
    token_ids = indices + vocab_start
    if tp_size == 1:
        return token_ids, values
    # fp32 holds token ids exactly (vocab < 2**24). The first rank holding
    # the max wins, i.e. the lowest token id, as argmax over the full row.
    pairs = all_gather(torch.stack([values, token_ids.float()], dim=-1))
    pairs = pairs.view(-1, tp_size, 2)
    best_rank = pairs[:, :, 0].argmax(dim=-1, keepdim=True)
    best = pairs.gather(1, best_rank.unsqueeze(-1).expand(-1, 1, 2)).squeeze(1)
    return best[:, 1].to(torch.int64), best[:, 0]


# Simulated block diffusion (synthetic_denoise_steps): per-request phase.
_DENOISING = tl.constexpr(0)
# The whole canvas is final; the request's next pass is its commit pass.
_FINALIZED = tl.constexpr(1)
# Committed this step; the next scheduler step starts a new block.
_COMMITTED = tl.constexpr(2)


@triton.jit
def _diffusion_random_tokens(seed, req_state_idx, seq_len, num_passes, offs, vocab_size):
    # Deterministic per (request slot, block, pass, position): the same
    # canvas noise whether or not the passes run in the runner loop.
    key = ((req_state_idx.to(tl.int64) * 1000003 + seq_len) * 4099 + num_passes) * 64
    rnd = tl.randint(seed, (key + offs).to(tl.uint32))
    return (rnd.to(tl.int64) & 0xFFFFFFFF) % vocab_size


@triton.jit
def _diffusion_pass_kernel(
    top_tokens_ptr,  # [num_logits] greedy token per logits row
    top_logits_ptr,  # [num_logits] its logit: the confidence
    cu_num_logits_ptr,
    idx_mapping_ptr,
    seq_lens_ptr,
    query_start_loc_ptr,
    input_ids_ptr,  # the decode graph's input-id buffer
    draft_tokens_ptr,  # [max_num_reqs, W - 1] next window's inputs 1..W-1
    draft_tokens_stride,
    sampled_ptr,  # [num_reqs, W] out
    sampled_stride,
    num_sampled_ptr,  # [num_reqs] out
    num_rejected_ptr,  # [num_reqs] out
    prefill_len_ptr,  # [max_num_reqs]
    commit_next_ptr,  # [num_reqs] out (host-mapped): 1 if the next pass commits
    block_seq_len_ptr,  # [max_num_reqs] seq_len of the request's block
    phase_ptr,  # [max_num_reqs]
    num_passes_ptr,  # [max_num_reqs] denoising passes run on the block
    canvas_ptr,  # [max_num_reqs, W]
    finalized_ptr,  # [max_num_reqs, W]
    vocab_size,
    seed,
    threshold,
    W: tl.constexpr,
    NUM_DENOISE_STEPS: tl.constexpr,
    USE_THRESHOLD: tl.constexpr,
    WRITE_INPUT_IDS: tl.constexpr,
):
    """One pass of the simulated block-diffusion decoder, per batch row.

    The window is [last committed token, canvas[0 : W - 1]] at positions
    L .. L + W - 1, and its logits rows predict canvas[0 : W] (positions
    L + 1 .. L + W). A denoising pass emits nothing: it finalizes canvas
    positions by confidence and writes the next pass's window inputs (final
    canvas tokens, random tokens elsewhere) into draft_tokens and, inside
    the runner loop, straight into the graph's input ids. A pass whose
    request was already FINALIZED ran the final canvas: it is the commit
    pass and emits the whole canvas (RecoverSSM then folds it in).
    """
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    tl.store(commit_next_ptr + batch_idx, 0)
    if req_state_idx < 0:
        tl.store(num_sampled_ptr + batch_idx, 0)
        tl.store(num_rejected_ptr + batch_idx, 0)
        return
    offs = tl.arange(0, W)
    in_window = offs < W - 1
    draft_ptrs = draft_tokens_ptr + req_state_idx * draft_tokens_stride + offs
    start = tl.load(cu_num_logits_ptr + batch_idx)
    # W, or fewer when the scheduler trims the drafts.
    num_logits = tl.load(cu_num_logits_ptr + batch_idx + 1) - start
    valid = offs < num_logits
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if num_logits <= 1:
        # Prefill (or a non-speculative step): sample as usual, nothing while
        # chunked prefilling. The first block's window starts fully masked.
        num_sampled = tl.where(
            seq_len < tl.load(prefill_len_ptr + req_state_idx), 0, 1
        )
        tl.store(sampled_ptr + batch_idx * sampled_stride, tl.load(top_tokens_ptr + start))
        tl.store(num_sampled_ptr + batch_idx, num_sampled)
        tl.store(num_rejected_ptr + batch_idx, 0)
        tl.store(block_seq_len_ptr + req_state_idx, -1)
        rnd = _diffusion_random_tokens(seed, req_state_idx, seq_len, 0, offs, vocab_size)
        tl.store(draft_ptrs, rnd, mask=in_window)
        return

    # Positions (so seq_len) only move at a commit: a new seq_len is a new
    # block, after a commit or for a new request.
    new_block = tl.load(block_seq_len_ptr + req_state_idx) != seq_len
    tl.store(block_seq_len_ptr + req_state_idx, seq_len)
    phase = tl.where(new_block, _DENOISING, tl.load(phase_ptr + req_state_idx))
    canvas_ptrs = canvas_ptr + req_state_idx * W + offs
    if phase == _FINALIZED:
        # Commit pass: the window held the final canvas.
        tl.store(
            sampled_ptr + batch_idx * sampled_stride + offs,
            tl.load(canvas_ptrs),
            mask=valid,
        )
        tl.store(num_sampled_ptr + batch_idx, num_logits)
        tl.store(num_rejected_ptr + batch_idx, 0)
        tl.store(phase_ptr + req_state_idx, _COMMITTED)
        # The next block starts fully masked.
        rnd = _diffusion_random_tokens(seed, req_state_idx, seq_len + W, 0, offs, vocab_size)
        tl.store(draft_ptrs, rnd, mask=in_window)
        return
    # A denoising pass emits nothing and rejects the whole window.
    tl.store(num_sampled_ptr + batch_idx, 0)
    tl.store(num_rejected_ptr + batch_idx, num_logits)
    if phase == _COMMITTED:
        # Not reached: the runner never re-runs a committed window.
        return

    finalized_ptrs = finalized_ptr + req_state_idx * W + offs
    num_passes = tl.load(num_passes_ptr + req_state_idx)
    num_passes = tl.where(new_block, 0, num_passes) + 1
    final = tl.where(new_block, 0, tl.load(finalized_ptrs)) != 0
    tokens = tl.load(top_tokens_ptr + start + offs, mask=valid, other=0)
    conf = tl.load(top_logits_ptr + start + offs, mask=valid, other=0.0)
    final = final & valid
    open_ = ~final & valid
    conf = tl.where(open_, conf, float("-inf"))
    # Rank of each open position by confidence (ties: lower position first).
    beats = (conf[None, :] > conf[:, None]) | (
        (conf[None, :] == conf[:, None]) & (offs[None, :] < offs[:, None])
    )
    rank = tl.sum((beats & open_[None, :]).to(tl.int32), axis=1)
    if USE_THRESHOLD:
        # Every confident enough position, and at least the most confident.
        pick = open_ & ((conf >= threshold) | (rank == 0))
    else:
        num_final = tl.sum(final.to(tl.int32), axis=0)
        target = tl.minimum(
            (num_logits * num_passes + NUM_DENOISE_STEPS - 1) // NUM_DENOISE_STEPS,
            num_logits,
        )
        pick = open_ & (rank < target - num_final)
    canvas = tl.where(pick, tokens, tl.load(canvas_ptrs))
    final = final | pick
    tl.store(canvas_ptrs, canvas)
    tl.store(finalized_ptrs, final.to(tl.int32))
    tl.store(num_passes_ptr + req_state_idx, num_passes)
    done = tl.sum(final.to(tl.int32), axis=0) == num_logits
    tl.store(phase_ptr + req_state_idx, tl.where(done, _FINALIZED, _DENOISING))
    tl.store(commit_next_ptr + batch_idx, done.to(tl.int32))

    # Next pass's window inputs 1 .. W - 1: canvas[0 : W - 1].
    rnd = _diffusion_random_tokens(seed, req_state_idx, seq_len, num_passes, offs, vocab_size)
    inputs = tl.where(final, canvas, rnd)
    tl.store(draft_ptrs, inputs, mask=in_window)
    if WRITE_INPUT_IDS:
        query_start = tl.load(query_start_loc_ptr + batch_idx)
        tl.store(
            input_ids_ptr + query_start + 1 + offs, inputs, mask=offs < num_logits - 1
        )


@triton.jit
def _diffusion_commit_next_kernel(
    commit_next_ptr,  # [num_reqs] out (host-mapped): 1 if the first pass commits
    cu_num_logits_ptr,
    idx_mapping_ptr,
    seq_lens_ptr,
    block_seq_len_ptr,
    phase_ptr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    commit = req_state_idx >= 0
    if commit:
        num_logits = tl.load(cu_num_logits_ptr + batch_idx + 1) - tl.load(
            cu_num_logits_ptr + batch_idx
        )
        commit = (
            (num_logits > 1)
            & (tl.load(block_seq_len_ptr + req_state_idx) == tl.load(seq_lens_ptr + batch_idx))
            & (tl.load(phase_ptr + req_state_idx) == _FINALIZED)
        )
    tl.store(commit_next_ptr + batch_idx, commit.to(tl.int32))


class _DiffusionFlags:
    """Per-row "next pass commits" flags in pinned host memory.

    The kernels write each pass's flags straight into a host slot through its
    device (UVA) view, so no copy is launched; an event recorded after the
    kernel marks them ready. The host reads a pass's flags only after it has
    queued that pass's forward, so waiting on the event does not idle the
    GPU."""

    NUM_SLOTS = 4

    def __init__(self, max_num_reqs: int):
        self.host = torch.zeros(
            self.NUM_SLOTS, max_num_reqs, dtype=torch.int32, pin_memory=True
        )
        self.uva = get_accelerator_view_from_cpu_tensor(self.host)
        self.events = [torch.cuda.Event(blocking=True) for _ in range(self.NUM_SLOTS)]
        self.slot = 0

    def next_slot(self) -> int:
        """The slot the next kernel writes (into self.uva[slot])."""
        return self.slot

    def record(self, num_reqs: int) -> tuple[int, int]:
        """Mark the flags written into next_slot() by the kernel just queued."""
        slot = self.slot
        self.slot = (slot + 1) % self.NUM_SLOTS
        self.events[slot].record()
        return slot, num_reqs

    def any(self, handle: tuple[int, int]) -> bool:
        slot, num_reqs = handle
        self.events[slot].synchronize()
        return bool(self.host[slot, :num_reqs].any())

    def get(self, handle: tuple[int, int]) -> list[int]:
        slot, num_reqs = handle
        self.events[slot].synchronize()
        return self.host[slot, :num_reqs].tolist()


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
        *,
        watermark_key: int | None = None,
        seed: int | None = None,
    ):
        self.sampler = sampler
        self.watermark_key = watermark_key
        if watermark_key is not None:
            assert isinstance(sampler, GPUWatermarkSampler)
        self.num_speculative_steps = spec_config.num_speculative_tokens
        self.enable_adaptive_verification = spec_config.enable_adaptive_verification
        rejection_sample_method = spec_config.rejection_sample_method
        self.use_block_verification: bool = False
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if rejection_sample_method == "synthetic":
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        elif rejection_sample_method == "block":
            self.use_block_verification = True
        # Simulated block diffusion (see sample_denoise_pass). Only the
        # greedy top-1 path implements it; this class's __call__ (used e.g.
        # by the structured-output warmup) samples as regular spec decoding.
        self.synthetic_denoise_steps = spec_config.synthetic_denoise_steps
        if self.synthetic_denoise_steps is not None:
            max_num_reqs = sampler.req_states.prefill_len.gpu.shape[0]
            # seq_len of each request's current block.
            self.block_seq_len = torch.full(
                (max_num_reqs,), -1, dtype=torch.int32, device=device
            )
            window = self.num_speculative_steps + 1
            assert window == triton.next_power_of_2(window), (
                "synthetic_denoise_steps needs num_speculative_tokens + 1 to "
                "be a power of 2"
            )
            self.denoise_threshold = spec_config.synthetic_denoise_threshold
            self.denoise_loop = spec_config.synthetic_denoise_loop
            # A block takes at most this many passes (denoising + commit).
            self.max_block_passes = (
                window if self.denoise_threshold is not None
                else self.synthetic_denoise_steps
            ) + 1
            self.denoise_seed = 0 if seed is None else seed
            self.denoise_phase = torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            )
            self.denoise_passes = torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            )
            self.denoise_canvas = torch.zeros(
                max_num_reqs, window, dtype=torch.int64, device=device
            )
            self.denoise_finalized = torch.zeros(
                max_num_reqs, window, dtype=torch.int32, device=device
            )
            self.denoise_flags = _DiffusionFlags(max_num_reqs)
            # Outputs of the runner loop's non-final passes, which emit
            # nothing and are never copied out (the final pass allocates).
            self._denoise_scratch = (
                torch.zeros(max_num_reqs, window, dtype=torch.int64, device=device),
                torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
                torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
            )

    def denoise_commit_next(self, input_batch: InputBatch) -> tuple[int, int]:
        """Flag which rows this step's first pass will commit.

        Call before launching the step's forward, so the flags are ready ahead
        of it; returns a handle for denoise_flags.any()."""
        num_reqs = input_batch.num_reqs
        flags = self.denoise_flags
        _diffusion_commit_next_kernel[(num_reqs,)](
            flags.uva[flags.next_slot()],
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            input_batch.seq_lens,
            self.block_seq_len,
            self.denoise_phase,
        )
        return self.denoise_flags.record(num_reqs)

    def sample_denoise_pass(
        self,
        top_tokens: torch.Tensor,
        top_logits: torch.Tensor,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
        vocab_size: int,
        write_input_ids: bool,
        emit: bool = True,
    ) -> tuple[SamplerOutput, tuple[int, int]]:
        """One simulated block-diffusion pass (synthetic_denoise_steps), greedy.

        top_tokens / top_logits: the greedy token and its logit (the
        confidence) for every logits row. Denoising passes emit nothing and
        update the canvas; a commit pass emits the canvas. Writes the next
        pass's window inputs into draft_tokens (and, with write_input_ids,
        into input_batch.input_ids) and the "next pass commits" flags; returns
        their denoise_flags handle. No host synchronization.

        emit=False: the caller knows no row commits or samples in this pass
        (a non-final pass of the runner loop) and discards the output, so it
        goes to persistent scratch buffers instead of fresh allocations.
        """
        assert self.synthetic_denoise_steps is not None
        num_reqs = input_batch.num_reqs
        window = self.num_speculative_steps + 1
        if emit:
            sampled = top_tokens.new_zeros(num_reqs, window)
            num_sampled = torch.empty(
                num_reqs, dtype=torch.int32, device=top_tokens.device
            )
            num_rejected = torch.empty_like(num_sampled)
        else:
            sampled, num_sampled, num_rejected = (
                t[:num_reqs] for t in self._denoise_scratch
            )
        flags = self.denoise_flags
        _diffusion_pass_kernel[(num_reqs,)](
            top_tokens,
            top_logits,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            input_batch.seq_lens,
            input_batch.query_start_loc,
            input_batch.input_ids,
            draft_tokens,
            draft_tokens.stride(0),
            sampled,
            sampled.stride(0),
            num_sampled,
            num_rejected,
            self.sampler.req_states.prefill_len.gpu,
            flags.uva[flags.next_slot()],
            self.block_seq_len,
            self.denoise_phase,
            self.denoise_passes,
            self.denoise_canvas,
            self.denoise_finalized,
            vocab_size,
            self.denoise_seed,
            float(self.denoise_threshold or 0.0),
            W=window,
            NUM_DENOISE_STEPS=self.synthetic_denoise_steps,
            USE_THRESHOLD=self.denoise_threshold is not None,
            WRITE_INPUT_IDS=write_input_ids,
        )
        output = SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
        return output, flags.record(num_reqs)

    def _get_logprobs_tensors(
        self,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        cu_num_logits_np: np.ndarray,
        max_num_logprobs: int,
    ) -> LogprobsTensors | None:
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            cu_num_logits,
            num_warps=1,
        )
        expanded_logits = num_logits != num_reqs
        cu_num_generated_tokens: list[int] | torch.Tensor | None = None
        if expanded_logits:
            if self.enable_adaptive_verification:
                # Adaptive verification keeps the true per-request boundaries
                # on device only; cu_num_logits_np holds the pre-compacted layout.
                cu_num_generated_tokens = cu_num_logits.clone()
            else:
                cu_num_generated_tokens = cu_num_logits_np.tolist()
        return compute_topk_scores(
            logits,
            max_num_logprobs,
            flat_sampled,
            cu_num_generated_tokens,
            logits_mode=self.sampler.logprobs_mode
            in ("raw_logits", "processed_logits"),
        )

    def _watermarking_kwargs(
        self,
        draft_sampled: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> dict[str, Any]:
        if self.watermark_key is None:
            return {}
        assert isinstance(self.sampler, GPUWatermarkSampler)
        contexts = self.sampler._get_contexts(
            expanded_idx_mapping, expanded_local_pos, draft_sampled
        )
        watermarking_skip_mask = None
        if self.sampler.deduplicate_contexts != "none":
            watermarking_skip_mask = self.sampler._get_repeated_contexts(
                expanded_idx_mapping, contexts, expanded_local_pos
            )
        return {
            "contexts": contexts,
            "watermarking": self.sampler.watermarking.gpu,
            "watermarking_skip_mask": watermarking_skip_mask,
            "watermark_key": self.watermark_key,
        }

    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        seq_lens_upper_bound_np: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
            seq_lens_upper_bound_np,
        )
        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
            **self._watermarking_kwargs(
                draft_sampled, expanded_idx_mapping, expanded_local_pos
            ),
        )
        return processed_logits, sampled, num_sampled

    def _verify_in_chunks(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        max_chunk_logits: int,
        max_num_logprobs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, LogprobsTensors | None]:
        cu_num_logits_np = input_batch.cu_num_logits_np
        use_processed_logits = self.sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
        num_reqs = input_batch.num_reqs

        if logits.shape[0] <= max_chunk_logits:
            # One chunk covers the batch. Adaptive verification compacts the logits
            # without updating cu_num_logits_np (it keeps the pre-compacted layout),
            # so the stale sums must not pick chunk boundaries; its budget cap
            # guarantees the compacted batch always lands here.
            request_chunks: Iterable[tuple[int, int]] = ((0, num_reqs),)
        else:
            assert not self.enable_adaptive_verification
            request_chunks = _iter_request_chunks(cu_num_logits_np, max_chunk_logits)

        sampled_chunks: list[torch.Tensor] = []
        num_sampled_chunks: list[torch.Tensor] = []
        logprobs_chunks: list[LogprobsTensors] = []

        for start, end in request_chunks:
            lo = int(cu_num_logits_np[start])
            hi = int(cu_num_logits_np[end])
            chunk_cu_num_logits_np = cu_num_logits_np[start : end + 1] - lo
            chunk_cu_num_logits = input_batch.cu_num_logits[start : end + 1] - lo
            # draft_logits uses persistent request-state indices and stays global.
            processed_logits, sampled, num_sampled = self._verify(
                logits[lo:hi],
                draft_logits,
                draft_sampled[lo:hi],
                pos[lo:hi],
                chunk_cu_num_logits,
                input_batch.idx_mapping[start:end],
                input_batch.idx_mapping_np[start:end],
                input_batch.expanded_idx_mapping[lo:hi],
                input_batch.expanded_local_pos[lo:hi],
                input_batch.seq_lens_cpu_upper_bound.numpy()[start:end],
            )
            chunk_logprobs = self._get_logprobs_tensors(
                sampled,
                num_sampled,
                processed_logits if use_processed_logits else logits[lo:hi],
                chunk_cu_num_logits,
                chunk_cu_num_logits_np,
                max_num_logprobs,
            )
            if chunk_logprobs is not None:
                logprobs_chunks.append(chunk_logprobs)
            del processed_logits
            sampled_chunks.append(sampled)
            num_sampled_chunks.append(num_sampled)

        if len(sampled_chunks) == 1:
            logprobs_tensors = logprobs_chunks[0] if logprobs_chunks else None
            return sampled_chunks[0], num_sampled_chunks[0], logprobs_tensors

        logprobs_tensors = None
        if logprobs_chunks:
            expanded_logits = logits.shape[0] != input_batch.num_reqs
            logprobs_tensors = LogprobsTensors.cat(
                logprobs_chunks,
                cu_num_generated_tokens=(
                    cu_num_logits_np.tolist() if expanded_logits else None
                ),
            )

        sampled = torch.cat(sampled_chunks)
        num_sampled = torch.cat(num_sampled_chunks)
        return sampled, num_sampled, logprobs_tensors

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        draft_sampled, pos = gather_draft_sampled(
            input_batch.input_ids,
            input_batch.positions,
            input_batch.logits_indices,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.sampler.req_states.prefill_len.gpu,
        )

        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        chunk_logit_limit = get_max_chunk_logits(logits.shape[1])
        sampled, num_sampled, logprobs_tensors = self._verify_in_chunks(
            logits,
            input_batch,
            draft_logits,
            draft_sampled,
            pos,
            chunk_logit_limit,
            max_num_logprobs,
        )

        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
