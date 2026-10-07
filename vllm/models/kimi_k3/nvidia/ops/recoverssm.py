# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 RecoverSSM speculative verify and accepted-state recovery."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@triton.jit
def _kda_gate(
    raw_g,
    dt_bias,
    A,
    lower_bound,
    USE_LOWER_BOUND: tl.constexpr,
):
    gate_input = raw_g + dt_bias
    if USE_LOWER_BOUND:
        return lower_bound * tl.sigmoid(A * gate_input)
    softplus_gate = tl.where(
        gate_input > 20.0,
        gate_input,
        tl.log(1.0 + tl.exp(gate_input)),
    )
    return -A * softplus_gate


@triton.jit
def _kda_recoverssm_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    raw_g_ptr,
    raw_beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    correction_cache_ptr,
    kg_cache_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    lower_bound,
    null_block_id,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_g_token,
    stride_beta_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_block,
    stride_correction_head,
    stride_correction_pos,
    stride_correction_dim,
    stride_kg_block,
    stride_kg_head,
    stride_kg_pos,
    stride_kg_dim,
    stride_out_token,
    stride_query_start_loc,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    HEADS_PER_QK_HEAD: tl.constexpr,
    SCALAR_GATE: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    # GDN shares each q/k head across HEADS_PER_QK_HEAD value heads (KDA: 1).
    pid_qk = pid_h // HEADS_PER_QK_HEAD

    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    if state_idx <= null_block_id:
        for token_offset in tl.static_range(SPEC_QUERY_LEN):
            token_valid = token_offset < query_len
            tl.store(
                out_ptr + (bos + token_offset) * stride_out_token + pid_h * V + offs_v,
                tl.zeros([BV], dtype=tl.float32),
                mask=token_valid & mask_v,
            )
        return

    state_ptrs = (
        state_ptr
        + state_idx * stride_state_block
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    state = tl.load(state_ptrs, mask=mask_state, other=0.0).to(tl.float32)
    A = tl.exp(tl.load(A_log_ptr + pid_h).to(tl.float32))
    if SCALAR_GATE:
        dt_bias = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    else:
        dt_bias = tl.load(dt_bias_ptr + pid_h * K + offs_k, mask=mask_k, other=0.0).to(
            tl.float32
        )
    correction_ptr = (
        correction_cache_ptr
        + state_idx * stride_correction_block
        + pid_h * stride_correction_head
        + offs_v * stride_correction_dim
    )
    kg_ptr = kg_cache_ptr + state_idx * stride_kg_block + pid_h * stride_kg_head

    # The recurrence is serial over the window, so only the request's real
    # tokens are visited (a runtime loop; no masked padding steps).
    for token_offset in range(0, tl.minimum(query_len, SPEC_QUERY_LEN)):
        token = bos + token_offset
        q = tl.load(
            q_ptr + token * stride_q_token + pid_qk * K + offs_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + token * stride_k_token + pid_qk * K + offs_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + token * stride_v_token + pid_h * V + offs_v,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        raw_beta = tl.load(raw_beta_ptr + token * stride_beta_token + pid_h).to(
            tl.float32
        )
        q *= tl.rsqrt(tl.sum(q * q) + 1e-6) * (K**-0.5)
        normalized_k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
        if SCALAR_GATE:
            # GDN: one gate per head, a scalar decay of the whole state.
            raw_g = tl.load(raw_g_ptr + token * stride_g_token + pid_h).to(
                tl.float32
            )
            gate = _kda_gate(raw_g, dt_bias, A, lower_bound, USE_LOWER_BOUND)
            state *= tl.exp(gate)
            # Broadcast over K so the records match KDA's per-channel layout
            # and the KDA commit kernel applies as is.
            raw_g = raw_g + tl.zeros([BK], dtype=tl.float32)
        else:
            raw_g = tl.load(
                raw_g_ptr + token * stride_g_token + pid_h * K + offs_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            gate = _kda_gate(raw_g, dt_bias, A, lower_bound, USE_LOWER_BOUND)
            state *= tl.exp(gate)[None, :]
        correction = v - tl.sum(state * normalized_k[None, :], axis=1)
        correction *= tl.sigmoid(raw_beta)
        state += correction[:, None] * normalized_k[None, :]

        out = tl.sum(state * q[None, :], axis=1)
        tl.store(
            out_ptr + token * stride_out_token + pid_h * V + offs_v,
            out,
            mask=mask_v,
        )
        tl.store(
            correction_ptr + token_offset * stride_correction_pos,
            correction,
            mask=mask_v,
        )
        if pid_v == 0:
            kg_token_ptr = kg_ptr + token_offset * stride_kg_pos
            tl.store(kg_token_ptr + offs_k * stride_kg_dim, k, mask=mask_k)
            tl.store(
                kg_token_ptr + (K + offs_k) * stride_kg_dim, raw_g, mask=mask_k
            )


@triton.jit
def _gdn_recoverssm_chunk_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    raw_g_ptr,
    raw_beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    correction_cache_ptr,
    kg_cache_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    lower_bound,
    null_block_id,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_g_token,
    stride_beta_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_block,
    stride_correction_head,
    stride_correction_pos,
    stride_correction_dim,
    stride_kg_block,
    stride_kg_head,
    stride_kg_pos,
    stride_kg_dim,
    stride_out_token,
    stride_query_start_loc,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    HEADS_PER_QK_HEAD: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """GDN (scalar gate) verify of a whole window at once, in chunk form.

    With G = cumsum(g), k^ and q^ the normalized keys/queries and S0 the
    checkpoint ([V, K]), the recurrence's per-token corrections C (the
    records the commit replays) and outputs O are
        (I + A) C = beta * V - (beta * exp(G)) * (K^ S0^T),
        A[t, j] = beta_t exp(G_t - G_j) k^_t . k^_j            (j < t)
        O = exp(G) * (Q^ S0^T) + (exp(G_t - G_j) q^_t . k^_j)_{j <= t} C
    so only the BT x BT triangular solve is serial over tokens.
    """
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_qk = pid_h // HEADS_PER_QK_HEAD

    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)

    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_t = offs_t < query_len
    mask_k = offs_k < K
    mask_v = offs_v < V
    tokens = bos + offs_t
    out_ptrs = out_ptr + tokens[:, None] * stride_out_token + pid_h * V + offs_v[None, :]
    mask_tv = mask_t[:, None] & mask_v[None, :]
    if state_idx <= null_block_id:
        tl.store(out_ptrs, tl.zeros([BT, BV], dtype=tl.float32), mask=mask_tv)
        return

    mask_tk = mask_t[:, None] & mask_k[None, :]
    q = tl.load(
        q_ptr + tokens[:, None] * stride_q_token + pid_qk * K + offs_k[None, :],
        mask=mask_tk,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_ptr + tokens[:, None] * stride_k_token + pid_qk * K + offs_k[None, :],
        mask=mask_tk,
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr + tokens[:, None] * stride_v_token + pid_h * V + offs_v[None, :],
        mask=mask_tv,
        other=0.0,
    ).to(tl.float32)
    raw_g = tl.load(
        raw_g_ptr + tokens * stride_g_token + pid_h, mask=mask_t, other=0.0
    ).to(tl.float32)
    raw_beta = tl.load(
        raw_beta_ptr + tokens * stride_beta_token + pid_h, mask=mask_t, other=0.0
    ).to(tl.float32)
    A = tl.exp(tl.load(A_log_ptr + pid_h).to(tl.float32))
    dt_bias = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    g = tl.where(
        mask_t, _kda_gate(raw_g, dt_bias, A, lower_bound, USE_LOWER_BOUND), 0.0
    )
    beta = tl.where(mask_t, tl.sigmoid(raw_beta), 0.0)
    G = tl.cumsum(g, axis=0)
    q_hat = q * (tl.rsqrt(tl.sum(q * q, axis=1) + 1e-6) * (K**-0.5))[:, None]
    k_hat = k * tl.rsqrt(tl.sum(k * k, axis=1) + 1e-6)[:, None]

    state = tl.load(
        state_ptr
        + state_idx * stride_state_block
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k,
        mask=mask_v[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)

    # exp(G_t - G_j) on and below the diagonal (G is non-increasing).
    lower = offs_t[:, None] >= offs_t[None, :]
    decay = tl.where(lower, tl.exp(tl.where(lower, G[:, None] - G[None, :], 0.0)), 0.0)
    strict = offs_t[:, None] > offs_t[None, :]
    a_mat = tl.where(
        strict,
        beta[:, None]
        * decay
        * tl.dot(k_hat, tl.trans(k_hat), input_precision=DOT_PRECISION),
        0.0,
    )
    # (I + A)^-1 by forward substitution: row i = e_i - A[i, :] @ inv.
    inv = tl.where(offs_t[:, None] == offs_t[None, :], 1.0, 0.0)
    for i in tl.static_range(1, BT):
        a_row = tl.sum(tl.where(offs_t[:, None] == i, a_mat, 0.0), axis=0)
        row = tl.sum(a_row[:, None] * inv, axis=0)
        inv = tl.where(offs_t[:, None] == i, inv - row[None, :], inv)

    k_state = tl.dot(k_hat, tl.trans(state), input_precision=DOT_PRECISION)
    rhs = beta[:, None] * v - (beta * tl.exp(G))[:, None] * k_state
    correction = tl.dot(inv, rhs, input_precision=DOT_PRECISION)

    q_state = tl.dot(q_hat, tl.trans(state), input_precision=DOT_PRECISION)
    qk = decay * tl.dot(q_hat, tl.trans(k_hat), input_precision=DOT_PRECISION)
    out = tl.exp(G)[:, None] * q_state + tl.dot(
        qk, correction, input_precision=DOT_PRECISION
    )
    tl.store(out_ptrs, out, mask=mask_tv)

    tl.store(
        correction_cache_ptr
        + state_idx * stride_correction_block
        + pid_h * stride_correction_head
        + offs_t[:, None] * stride_correction_pos
        + offs_v[None, :] * stride_correction_dim,
        correction,
        mask=mask_tv,
    )
    if pid_v == 0:
        kg_ptrs = (
            kg_cache_ptr
            + state_idx * stride_kg_block
            + pid_h * stride_kg_head
            + offs_t[:, None] * stride_kg_pos
            + offs_k[None, :] * stride_kg_dim
        )
        tl.store(kg_ptrs, k, mask=mask_tk)
        # Per-head gate broadcast over K, as in the recurrent verify.
        tl.store(
            kg_ptrs + K * stride_kg_dim,
            raw_g[:, None] + tl.zeros([BT, BK], dtype=tl.float32),
            mask=mask_tk,
        )


@triton.heuristics(
    {
        "HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None,
        "ALIGN_MODE": lambda args: args["block_table_ptr"] is not None,
    }
)
@triton.jit
def _prepare_commit_plan_kernel(
    num_accepted_ptr,
    request_indices_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    num_computed_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    mamba_block_size,
    block_table_width,
    stride_num_accepted,
    stride_request_indices,
    stride_state_indices,
    stride_query_start_loc,
    stride_block_table_row,
    stride_block_table_col,
    stride_num_computed,
    SPEC_QUERY_LEN: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    spec_idx = tl.program_id(0)
    source_state_idx = tl.load(state_indices_ptr + spec_idx * stride_state_indices).to(
        tl.int64
    )
    request_idx = spec_idx
    if HAS_REQUEST_INDICES:
        request_idx = tl.load(
            request_indices_ptr + spec_idx * stride_request_indices
        ).to(tl.int64)
    num_accepted = tl.load(num_accepted_ptr + request_idx * stride_num_accepted).to(
        tl.int32
    )
    bos = tl.load(query_start_loc_ptr + spec_idx * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (spec_idx + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = (eos - bos).to(tl.int32)
    commit_len = tl.minimum(tl.maximum(num_accepted, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)

    final_state_idx = source_state_idx
    boundary_state_idx = null_block_id
    boundary_recovery_len = 0
    if ALIGN_MODE:
        num_computed = tl.load(num_computed_ptr + request_idx * stride_num_computed).to(
            tl.int32
        )
        final_num_computed = num_computed + commit_len
        final_state_col = tl.minimum(
            final_num_computed // mamba_block_size, block_table_width - 1
        )
        final_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + final_state_col * stride_block_table_col
        ).to(tl.int64)
        next_boundary = (num_computed // mamba_block_size + 1) * mamba_block_size
        crosses_boundary = final_num_computed >= next_boundary
        boundary_recovery_len = next_boundary - num_computed
        boundary_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + (next_boundary // mamba_block_size - 1) * stride_block_table_col,
            mask=crosses_boundary,
            other=null_block_id,
        ).to(tl.int64)
    valid = (source_state_idx > null_block_id) & (commit_len > 0)
    tl.store(commit_lens_ptr + spec_idx, tl.where(valid, commit_len, 0))
    tl.store(
        final_state_indices_ptr + spec_idx,
        tl.where(valid, final_state_idx, null_block_id),
    )
    tl.store(
        boundary_state_indices_ptr + spec_idx,
        tl.where(valid, boundary_state_idx, null_block_id),
    )
    tl.store(
        boundary_recovery_lens_ptr + spec_idx,
        tl.where(valid, boundary_recovery_len, 0),
    )


@triton.heuristics(
    {"PER_LAYER_STATE_INDICES": lambda args: args["state_indices_addrs_ptr"] is not None}
)
@triton.jit
def _compact_conv_state_kernel(
    conv_state_ref_ptr,
    conv_state_base_addrs_ptr,
    conv_state_block_strides_ptr,
    conv_state_dim_strides_ptr,
    conv_state_token_strides_ptr,
    state_indices_ptr,
    state_indices_addrs_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    conv_dim,
    conv_history_len,
    stride_state_indices,
    BLOCK_D: tl.constexpr,
    BLOCK_HISTORY: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    PER_LAYER_STATE_INDICES: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_l = tl.program_id(2)
    if PER_LAYER_STATE_INDICES:
        state_indices_ptr = tl.load(state_indices_addrs_ptr + pid_l).to(
            tl.pointer_type(tl.int32)
        )
    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(
        tl.int64
    )
    if source_state_idx <= null_block_id:
        return

    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_state_idx = source_state_idx
    if ALIGN_MODE:
        final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

    if final_state_idx <= null_block_id:
        return

    base_addr = tl.load(conv_state_base_addrs_ptr + pid_l)
    block_stride = tl.load(conv_state_block_strides_ptr + pid_l)
    dim_stride = tl.load(conv_state_dim_strides_ptr + pid_l)
    token_stride = tl.load(conv_state_token_strides_ptr + pid_l)
    conv_state_ptr = base_addr.to(tl.pointer_type(conv_state_ref_ptr.dtype.element_ty))
    source_ptr = conv_state_ptr + source_state_idx * block_stride
    final_ptr = conv_state_ptr + final_state_idx * block_stride

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, BLOCK_HISTORY)
    mask = (offs_d[:, None] < conv_dim) & (offs_h[None, :] < conv_history_len)
    final_values = tl.load(
        source_ptr
        + offs_d[:, None] * dim_stride
        + (commit_len - 1 + offs_h[None, :]) * token_stride,
        mask=mask,
    )
    if ALIGN_MODE:
        boundary_values = tl.load(
            source_ptr
            + offs_d[:, None] * dim_stride
            + (boundary_recovery_len - 1 + offs_h[None, :]) * token_stride,
            mask=mask & (boundary_state_idx > null_block_id),
        )
        boundary_ptr = conv_state_ptr + boundary_state_idx * block_stride
        tl.store(
            boundary_ptr
            + offs_d[:, None] * dim_stride
            + offs_h[None, :] * token_stride,
            boundary_values,
            mask=mask & (boundary_state_idx > null_block_id),
        )
    tl.store(
        final_ptr + offs_d[:, None] * dim_stride + offs_h[None, :] * token_stride,
        final_values,
        mask=mask,
    )


@triton.heuristics(
    {"PER_LAYER_STATE_INDICES": lambda args: args["state_indices_addrs_ptr"] is not None}
)
@triton.jit
def _commit_kda_state_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    correction_cache_ref_ptr,
    correction_cache_base_addrs_ptr,
    correction_cache_block_strides_ptr,
    kg_cache_ref_ptr,
    kg_cache_base_addrs_ptr,
    kg_cache_block_strides_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_indices_ptr,
    state_indices_addrs_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    lower_bound,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_cache_head,
    stride_correction_cache_pos,
    stride_correction_cache_dim,
    stride_kg_cache_head,
    stride_kg_cache_pos,
    stride_kg_cache_dim,
    stride_A_layer,
    stride_A_head,
    stride_dt_bias_layer,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    PER_LAYER_STATE_INDICES: tl.constexpr,
    CHUNKED: tl.constexpr = False,
    BT: tl.constexpr = 16,
    DOT_PRECISION: tl.constexpr = "tf32x3",
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_lh = tl.program_id(2)
    pid_l = pid_lh // NUM_HEADS
    pid_h = pid_lh % NUM_HEADS

    if PER_LAYER_STATE_INDICES:
        state_indices_ptr = tl.load(state_indices_addrs_ptr + pid_l).to(
            tl.pointer_type(tl.int32)
        )
    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(
        tl.int64
    )
    if source_state_idx <= null_block_id:
        return
    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    # Without align mode the plan's final index is the source index.
    final_state_idx = source_state_idx
    if ALIGN_MODE:
        final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

    if final_state_idx <= null_block_id:
        return

    state_base_addr = tl.load(state_base_addrs_ptr + pid_l)
    state_block_stride = tl.load(state_block_strides_ptr + pid_l)
    state_ptr = state_base_addr.to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    source_state_ptr = (
        state_ptr + source_state_idx * state_block_stride + pid_h * stride_state_head
    )

    correction_cache_base_addr = tl.load(correction_cache_base_addrs_ptr + pid_l)
    correction_cache_block_stride = tl.load(correction_cache_block_strides_ptr + pid_l)
    correction_cache_ptr = correction_cache_base_addr.to(
        tl.pointer_type(correction_cache_ref_ptr.dtype.element_ty)
    )
    correction_cache_ptr += (
        source_state_idx * correction_cache_block_stride
        + pid_h * stride_correction_cache_head
    )
    kg_cache_base_addr = tl.load(kg_cache_base_addrs_ptr + pid_l)
    kg_cache_block_stride = tl.load(kg_cache_block_strides_ptr + pid_l)
    kg_cache_ptr = kg_cache_base_addr.to(
        tl.pointer_type(kg_cache_ref_ptr.dtype.element_ty)
    )
    kg_cache_ptr += (
        source_state_idx * kg_cache_block_stride + pid_h * stride_kg_cache_head
    )

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    state_ptrs = (
        source_state_ptr
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    initial_state = tl.load(state_ptrs, mask=mask_state, other=0.0).to(tl.float32)
    A = tl.exp(
        tl.load(A_log_ptr + pid_l * stride_A_layer + pid_h * stride_A_head).to(
            tl.float32
        )
    )

    dt_bias = tl.load(
        dt_bias_ptr
        + pid_l * stride_dt_bias_layer
        + pid_h * stride_dt_bias_head
        + offs_k * stride_dt_bias_dim,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    if CHUNKED:
        # Parallel fold of the whole window. With G = cumsum(gate) over the
        # committed tokens (inclusive), the serial replay below equals
        #   S = exp(G_end) * S0 + C^T @ (k_hat * exp(G_end - G)),
        # one [BV, BT] x [BT, BK] dot. The gate is <= 0, so the weights are
        # in (0, 1]. Rows past the commit length have zero weight.
        offs_t = tl.arange(0, BT)
        mask_t = offs_t < commit_len
        mask_tk = mask_t[:, None] & mask_k[None, :]
        kg_ptrs = (
            kg_cache_ptr
            + offs_t[:, None] * stride_kg_cache_pos
            + offs_k[None, :] * stride_kg_cache_dim
        )
        k = tl.load(kg_ptrs, mask=mask_tk, other=0.0).to(tl.float32)
        raw_g = tl.load(
            kg_ptrs + K * stride_kg_cache_dim, mask=mask_tk, other=0.0
        ).to(tl.float32)
        correction = tl.load(
            correction_cache_ptr
            + offs_t[:, None] * stride_correction_cache_pos
            + offs_v[None, :] * stride_correction_cache_dim,
            mask=mask_t[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        normalized_k = k * tl.rsqrt(tl.sum(k * k, axis=1) + 1e-6)[:, None]
        gate = tl.where(
            mask_tk,
            _kda_gate(raw_g, dt_bias[None, :], A, lower_bound, USE_LOWER_BOUND),
            0.0,
        )
        G = tl.cumsum(gate, axis=0)
        G_end = tl.sum(gate, axis=0)
        weighted_k = tl.where(
            mask_t[:, None], normalized_k * tl.exp(G_end[None, :] - G), 0.0
        )
        state = initial_state * tl.exp(G_end)[None, :] + tl.dot(
            tl.trans(correction), weighted_k, input_precision=DOT_PRECISION
        )
        if ALIGN_MODE:
            before_boundary = offs_t < boundary_recovery_len
            G_boundary = tl.sum(
                tl.where(before_boundary[:, None], gate, 0.0), axis=0
            )
            boundary_k = tl.where(
                before_boundary[:, None],
                normalized_k * tl.exp(G_boundary[None, :] - G),
                0.0,
            )
            boundary_state = initial_state * tl.exp(G_boundary)[None, :] + tl.dot(
                tl.trans(correction), boundary_k, input_precision=DOT_PRECISION
            )
    else:
        final_decay = tl.full([BK], 1.0, tl.float32)
        final_correction = tl.zeros([BV, BK], tl.float32)
        boundary_decay = tl.full([BK], 1.0, tl.float32)
        boundary_correction = tl.zeros([BV, BK], tl.float32)

        for reverse_offset in range(commit_len):
            token_offset = commit_len - reverse_offset - 1
            correction_ptr = (
                correction_cache_ptr + token_offset * stride_correction_cache_pos
            )
            kg_ptr = kg_cache_ptr + token_offset * stride_kg_cache_pos
            k = tl.load(
                kg_ptr + offs_k * stride_kg_cache_dim,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            correction = tl.load(
                correction_ptr + offs_v * stride_correction_cache_dim,
                mask=mask_v,
                other=0.0,
            ).to(tl.float32)
            raw_g = tl.load(
                kg_ptr + (K + offs_k) * stride_kg_cache_dim,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            normalized_k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
            gate = _kda_gate(
                raw_g,
                dt_bias,
                A,
                lower_bound,
                USE_LOWER_BOUND,
            )
            update = correction[:, None] * normalized_k[None, :]
            decay = tl.exp(gate)
            final_correction += update * final_decay[None, :]
            final_decay *= decay
            if ALIGN_MODE:
                before_boundary = token_offset < boundary_recovery_len
                boundary_correction += tl.where(
                    before_boundary,
                    update * boundary_decay[None, :],
                    0.0,
                )
                boundary_decay *= tl.where(before_boundary, decay, 1.0)

        state = initial_state * final_decay[None, :] + final_correction
        if ALIGN_MODE:
            boundary_state = (
                initial_state * boundary_decay[None, :] + boundary_correction
            )
    if ALIGN_MODE:
        boundary_ptrs = (
            state_ptr
            + boundary_state_idx * state_block_stride
            + pid_h * stride_state_head
            + offs_v[:, None] * stride_state_v
            + offs_k[None, :] * stride_state_k
        )
        tl.store(
            boundary_ptrs,
            boundary_state,
            mask=mask_state & (boundary_state_idx > null_block_id),
        )

    final_ptrs = (
        state_ptr
        + final_state_idx * state_block_stride
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    tl.store(final_ptrs, state, mask=mask_state)


def kda_recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kg_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    out: torch.Tensor | None = None,
    block_v: int = 32,
    chunked: bool = False,
    dot_precision: str = "tf32",
) -> torch.Tensor:
    """Verify a KDA speculative window without modifying its checkpoint.

    Also serves GDN: q/k may have fewer heads than v (GQA), and ``raw_g`` and
    ``dt_bias`` may hold one gate per head instead of one per key channel.
    ``chunked`` (GDN only) verifies the whole window in chunk form."""
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("KDA RecoverSSM q must have shape [1, tokens, heads, dim]")
    _, total_tokens, num_qk_heads, key_dim = q.shape
    num_heads, value_dim = v.shape[-2:]
    scalar_gate = raw_g.ndim == 3
    if (
        k.shape != q.shape
        or v.shape != (1, total_tokens, num_heads, value_dim)
        or num_heads % num_qk_heads != 0
    ):
        raise ValueError("KDA RecoverSSM q, k, and v shapes are incompatible")
    gate_shape = (1, total_tokens, num_heads) + (() if scalar_gate else (key_dim,))
    if raw_g.shape != gate_shape or raw_beta.shape != (1, total_tokens, num_heads):
        raise ValueError("KDA RecoverSSM gate or beta shape is incompatible")
    if any(tensor.stride()[2:] != (key_dim, 1) for tensor in (q, k)) or (
        raw_g.stride()[2:] != ((1,) if scalar_gate else (key_dim, 1))
    ):
        raise ValueError("KDA RecoverSSM q, k, and gate heads must be contiguous")
    if v.stride()[2:] != (value_dim, 1) or raw_beta.stride(2) != 1:
        raise ValueError("KDA RecoverSSM v and beta heads must be contiguous")
    num_blocks = checkpoint_state.shape[0]
    if checkpoint_state.shape[1:] != (
        num_heads,
        value_dim,
        key_dim,
    ):
        raise ValueError("KDA RecoverSSM checkpoint shape is incompatible")
    expected_correction_shape = (
        num_blocks,
        num_heads,
        spec_query_len,
        value_dim,
    )
    if correction_cache.shape != expected_correction_shape:
        raise ValueError(
            f"KDA RecoverSSM correction buffer needs shape {expected_correction_shape}"
        )
    expected_kg_shape = (num_blocks, num_heads, spec_query_len, 2 * key_dim)
    if kg_cache.shape != expected_kg_shape:
        raise ValueError(
            f"KDA RecoverSSM key/gate buffer needs shape {expected_kg_shape}"
        )
    if correction_cache.dtype != torch.float32:
        raise ValueError("KDA RecoverSSM correction buffer must use float32")
    if kg_cache.dtype != k.dtype:
        raise ValueError("KDA RecoverSSM key/gate buffer must match activation dtype")
    if A_log.shape != (num_heads,) or dt_bias.numel() != num_heads * (
        1 if scalar_gate else key_dim
    ):
        raise ValueError("KDA RecoverSSM gate parameters are incompatible")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("KDA RecoverSSM gate parameters must be contiguous")
    batch = state_indices.shape[0]
    if query_start_loc.shape[0] != batch + 1:
        raise ValueError("KDA RecoverSSM query metadata is incompatible")
    if total_tokens > batch * spec_query_len:
        raise ValueError(
            "KDA RecoverSSM speculative decode input exceeds its activation capacity"
        )
    if out is None:
        out = torch.empty_like(v)
    if out.shape != v.shape:
        raise ValueError("KDA RecoverSSM output shape is incompatible")
    if out.stride()[2:] != (value_dim, 1):
        raise ValueError("KDA RecoverSSM output heads must be contiguous")
    device = q.device
    if any(
        tensor.device != device
        for tensor in (
            k,
            v,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            checkpoint_state,
            correction_cache,
            kg_cache,
            query_start_loc,
            state_indices,
            out,
        )
    ):
        raise ValueError("KDA RecoverSSM inputs must be on the same device")
    if total_tokens == 0:
        return out

    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), block_v)
    grid = (triton.cdiv(value_dim, block_v), batch, num_heads)
    if chunked:
        if not scalar_gate:
            raise ValueError("Chunked RecoverSSM verify needs a per-head gate")
        block_t = max(triton.next_power_of_2(spec_query_len), 16)
        if block_v < 16:
            raise ValueError("Chunked RecoverSSM verify needs block_v >= 16")
        kernel = _gdn_recoverssm_chunk_verify_kernel
        meta = dict(
            BT=block_t,
            DOT_PRECISION=dot_precision,
            num_warps=4,
            num_stages=1,
        )
    else:
        kernel = _kda_recoverssm_verify_kernel
        meta = dict(
            SPEC_QUERY_LEN=spec_query_len,
            SCALAR_GATE=scalar_gate,
            num_warps=4,
            num_stages=2,
        )
    kernel[grid](
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        checkpoint_state,
        correction_cache,
        kg_cache,
        out,
        query_start_loc,
        state_indices,
        lower_bound or 0.0,
        NULL_BLOCK_ID,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        raw_g.stride(1),
        raw_beta.stride(1),
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        checkpoint_state.stride(3),
        correction_cache.stride(0),
        correction_cache.stride(1),
        correction_cache.stride(2),
        correction_cache.stride(3),
        kg_cache.stride(0),
        kg_cache.stride(1),
        kg_cache.stride(2),
        kg_cache.stride(3),
        out.stride(1),
        query_start_loc.stride(0),
        state_indices.stride(0),
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        USE_LOWER_BOUND=lower_bound is not None,
        HEADS_PER_QK_HEAD=num_heads // num_qk_heads,
        **meta,
    )
    return out


@dataclass
class KDARecoverSSMCommitContext:
    conv_states: tuple[torch.Tensor, ...]
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    checkpoints: tuple[torch.Tensor, ...]
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    correction_caches: tuple[torch.Tensor, ...]
    correction_cache_base_addrs: torch.Tensor
    correction_cache_block_strides: torch.Tensor
    kg_caches: tuple[torch.Tensor, ...]
    kg_cache_base_addrs: torch.Tensor
    kg_cache_block_strides: torch.Tensor
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    lower_bound: float | None
    spec_query_len: int
    # Per-layer persistent state-index buffers, for layers from KV cache
    # groups with different block tables (GDN: one group per layer). The
    # state_indices passed to commit() then only select the batch rows.
    state_indices_addrs: torch.Tensor | None = None
    # Fold the accepted tokens with one tensor-core dot per (layer, head,
    # request) instead of a serial replay over the window.
    chunked_commit: bool = False

    @classmethod
    def create(
        cls,
        layers: Sequence[Any],
        *,
        spec_query_len: int,
        max_num_reqs: int,
        state_indices: Sequence[torch.Tensor] | None = None,
        chunked_commit: bool = False,
    ) -> "KDARecoverSSMCommitContext":
        if not layers:
            raise ValueError("KDA RecoverSSM commit requires at least one layer")
        if any(len(layer.kv_cache) != 4 for layer in layers):
            raise ValueError(
                "KDA RecoverSSM pages must contain conv, state, correction, "
                "and key/gate"
            )

        conv_states = [layer.kv_cache[0] for layer in layers]
        if not is_conv_state_dim_first():
            conv_states = [state.transpose(-1, -2) for state in conv_states]
        checkpoints = [layer.kv_cache[1] for layer in layers]
        correction_caches = [layer.kv_cache[2] for layer in layers]
        kg_caches = [layer.kv_cache[3] for layer in layers]
        A_log = [layer.A_log for layer in layers]
        # GDN layers have no gate lower bound.
        lower_bounds = {getattr(layer, "gate_lower_bound", None) for layer in layers}
        if len(lower_bounds) != 1:
            raise ValueError("KDA RecoverSSM layers need matching gate bounds")

        state_ref = checkpoints[0]
        if state_ref.ndim != 4:
            raise ValueError("KDA RecoverSSM checkpoint must be four-dimensional")
        num_blocks, num_heads, value_dim, key_dim = state_ref.shape
        # GDN's per-head dt_bias is broadcast over K: its verify pass records
        # the per-head gate broadcast the same way, so the KDA commit applies.
        dt_bias = [
            layer.dt_bias.view(num_heads, -1).expand(num_heads, key_dim)
            for layer in layers
        ]
        for state in checkpoints:
            if (
                state.shape != state_ref.shape
                or state.dtype != state_ref.dtype
                or state.device != state_ref.device
                or state.stride()[1:] != state_ref.stride()[1:]
            ):
                raise ValueError(
                    "KDA RecoverSSM layers need matching checkpoint layout"
                )
        expected_correction_shape = (
            num_blocks,
            num_heads,
            spec_query_len,
            value_dim,
        )
        correction_ref = correction_caches[0]
        for correction_cache in correction_caches:
            if (
                correction_cache.shape != expected_correction_shape
                or correction_cache.dtype != torch.float32
                or correction_cache.device != state_ref.device
                or correction_cache.stride()[1:] != correction_ref.stride()[1:]
            ):
                raise ValueError(
                    "KDA RecoverSSM correction buffers need float32 shape "
                    f"{expected_correction_shape}"
                )
        expected_kg_shape = (num_blocks, num_heads, spec_query_len, 2 * key_dim)
        kg_ref = kg_caches[0]
        for kg_cache in kg_caches:
            if (
                kg_cache.shape != expected_kg_shape
                or kg_cache.dtype != kg_ref.dtype
                or kg_cache.device != state_ref.device
                or kg_cache.stride()[1:] != kg_ref.stride()[1:]
            ):
                raise ValueError(
                    f"KDA RecoverSSM key/gate buffers need shape {expected_kg_shape}"
                )
        if any(param.shape != (num_heads,) for param in A_log):
            raise ValueError("KDA RecoverSSM A_log shape is incompatible")
        if any(param.shape != (num_heads, key_dim) for param in dt_bias):
            raise ValueError("KDA RecoverSSM dt_bias shape is incompatible")

        conv_ref = conv_states[0]
        if conv_ref.ndim != 3:
            raise ValueError("KDA RecoverSSM conv state must be three-dimensional")
        conv_dim, conv_state_len = conv_ref.shape[1:]
        conv_history_len = conv_state_len - spec_query_len + 1
        if conv_history_len <= 0:
            raise ValueError("KDA RecoverSSM conv state is shorter than its window")
        for conv_state in conv_states:
            if (
                conv_state.shape != conv_ref.shape
                or conv_state.dtype != conv_ref.dtype
                or conv_state.device != state_ref.device
                or conv_state.shape[0] != num_blocks
            ):
                raise ValueError("KDA RecoverSSM layers need matching conv state")

        device = state_ref.device

        def _base_addrs(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [tensor.data_ptr() for tensor in tensors],
                dtype=torch.int64,
                device=device,
            )

        def _block_strides(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [tensor.stride(0) for tensor in tensors],
                dtype=torch.int64,
                device=device,
            )

        return cls(
            conv_states=tuple(conv_states),
            conv_state_base_addrs=_base_addrs(conv_states),
            conv_state_block_strides=_block_strides(conv_states),
            conv_state_dim_strides=torch.tensor(
                [state.stride(1) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_state_token_strides=torch.tensor(
                [state.stride(2) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_history_len=conv_history_len,
            checkpoints=tuple(checkpoints),
            state_base_addrs=_base_addrs(checkpoints),
            state_block_strides=_block_strides(checkpoints),
            correction_caches=tuple(correction_caches),
            correction_cache_base_addrs=_base_addrs(correction_caches),
            correction_cache_block_strides=_block_strides(correction_caches),
            kg_caches=tuple(kg_caches),
            kg_cache_base_addrs=_base_addrs(kg_caches),
            kg_cache_block_strides=_block_strides(kg_caches),
            commit_lens=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            final_state_indices=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            boundary_state_indices=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            boundary_recovery_lens=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            A_log=torch.stack(tuple(A_log)).contiguous(),
            dt_bias=torch.stack(tuple(dt_bias)).contiguous(),
            lower_bound=lower_bounds.pop(),
            spec_query_len=spec_query_len,
            chunked_commit=chunked_commit,
            state_indices_addrs=(
                _base_addrs(state_indices) if state_indices is not None else None
            ),
        )

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
        state_indices_addrs: torch.Tensor | None = None,
    ) -> None:
        """Fold accepted KDA and convolution inputs into every layer.

        ``state_indices_addrs`` overrides the per-layer state-index buffers
        for this call (same row stride as ``state_indices``)."""
        if state_indices_addrs is None:
            state_indices_addrs = self.state_indices_addrs
        batch = state_indices.shape[0]
        if batch == 0:
            return
        if batch > self.commit_lens.shape[0]:
            raise ValueError("KDA RecoverSSM commit batch exceeds its plan capacity")
        if query_start_loc.shape[0] != batch + 1:
            raise ValueError("KDA RecoverSSM commit metadata is incompatible")
        if request_indices is not None and request_indices.shape[0] < batch:
            raise ValueError("KDA RecoverSSM request mapping is too short")
        align_args = (block_table, num_computed_tokens, mamba_block_size)
        if any(arg is not None for arg in align_args) and any(
            arg is None for arg in align_args
        ):
            raise ValueError("KDA RecoverSSM align metadata is incomplete")
        if mamba_block_size is not None and mamba_block_size < self.spec_query_len:
            raise ValueError(
                "KDA RecoverSSM align block size must cover one speculative window"
            )
        if block_table is not None and block_table.ndim != 2:
            raise ValueError("KDA RecoverSSM block table must be two-dimensional")
        if state_indices_addrs is not None and block_table is not None:
            raise ValueError("KDA RecoverSSM per-layer state indices need no align")
        device = self.checkpoints[0].device
        if (
            any(
                tensor.device != device
                for tensor in (
                    num_accepted_tokens,
                    state_indices,
                    query_start_loc,
                )
            )
            or (request_indices is not None and request_indices.device != device)
            or (block_table is not None and block_table.device != device)
            or (
                num_computed_tokens is not None and num_computed_tokens.device != device
            )
        ):
            raise ValueError("KDA RecoverSSM commit inputs must be on the same device")

        block_table_stride = (0, 0) if block_table is None else block_table.stride()
        num_computed_stride = (
            0 if num_computed_tokens is None else num_computed_tokens.stride(0)
        )

        num_layers = len(self.checkpoints)
        conv_ref = self.conv_states[0]
        conv_dim = conv_ref.shape[1]
        block_history = triton.next_power_of_2(self.conv_history_len)
        _prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens,
            request_indices,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            mamba_block_size or 1,
            block_table.shape[1] if block_table is not None else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            query_start_loc.stride(0),
            block_table_stride[0],
            block_table_stride[1],
            num_computed_stride,
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )
        _compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), batch, num_layers)](
            conv_ref,
            self.conv_state_base_addrs,
            self.conv_state_block_strides,
            self.conv_state_dim_strides,
            self.conv_state_token_strides,
            state_indices,
            state_indices_addrs,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            conv_dim,
            self.conv_history_len,
            state_indices.stride(0),
            BLOCK_D=256,
            BLOCK_HISTORY=block_history,
            ALIGN_MODE=block_table is not None,
            num_warps=4,
        )

        state_ref = self.checkpoints[0]
        _, num_heads, value_dim, key_dim = state_ref.shape
        block_k = triton.next_power_of_2(key_dim)
        # tl.dot needs every dimension >= 16.
        block_v = min(
            triton.next_power_of_2(value_dim), 64 if self.chunked_commit else 32
        )
        grid = (
            triton.cdiv(value_dim, block_v),
            batch,
            num_layers * num_heads,
        )
        _commit_kda_state_kernel[grid](
            state_ref,
            self.state_base_addrs,
            self.state_block_strides,
            self.correction_caches[0],
            self.correction_cache_base_addrs,
            self.correction_cache_block_strides,
            self.kg_caches[0],
            self.kg_cache_base_addrs,
            self.kg_cache_block_strides,
            self.A_log,
            self.dt_bias,
            state_indices,
            state_indices_addrs,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            self.lower_bound or 0.0,
            NULL_BLOCK_ID,
            state_ref.stride(1),
            state_ref.stride(2),
            state_ref.stride(3),
            self.correction_caches[0].stride(1),
            self.correction_caches[0].stride(2),
            self.correction_caches[0].stride(3),
            self.kg_caches[0].stride(1),
            self.kg_caches[0].stride(2),
            self.kg_caches[0].stride(3),
            self.A_log.stride(0),
            self.A_log.stride(1),
            self.dt_bias.stride(0),
            self.dt_bias.stride(1),
            self.dt_bias.stride(2),
            state_indices.stride(0),
            K=key_dim,
            V=value_dim,
            BK=block_k,
            BV=block_v,
            NUM_HEADS=num_heads,
            USE_LOWER_BOUND=self.lower_bound is not None,
            ALIGN_MODE=block_table is not None,
            CHUNKED=self.chunked_commit,
            BT=max(triton.next_power_of_2(self.spec_query_len), 16),
            num_warps=4,
            num_stages=2,
        )


__all__ = ["KDARecoverSSMCommitContext", "kda_recoverssm_verify"]
