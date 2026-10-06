# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointBuilder,
    MambaPrefillCheckpointMetadata,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import MambaSpec

if TYPE_CHECKING:
    from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
        KDARecoverSSMCommitContext,
    )


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class GDNAttentionMetadata(RecoverSSMMetadata):
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    checkpoint: MambaPrefillCheckpointMetadata | None = None
    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_sequence_masks_cpu: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]
    uniform_spec_sequence_length: int | None = None  # None for ragged batches

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    # Chunk-kernel inputs for prefill
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_state_indices: torch.Tensor | None = None
    prefill_has_initial_state: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None

    # RecoverSSM: the spec verify pass leaves the checkpoint untouched; after
    # sampling, this group's commit folds the accepted tokens into it.
    recoverssm_context: "KDARecoverSSMCommitContext | None" = field(
        default=None, repr=False, compare=False
    )
    # Batch row of each spec decode when spec rows are not the leading rows.
    recoverssm_request_indices: torch.Tensor | None = None
    # Per-layer state-index buffer addresses for this step's commit, when the
    # context's persistent ones (the FULL graph staging buffers) don't apply.
    recoverssm_state_indices_addrs: torch.Tensor | None = None

    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        if self.recoverssm_context is None:
            return None
        assert self.spec_state_indices_tensor is not None
        assert self.spec_query_start_loc is not None
        self.recoverssm_context.commit(
            num_accepted_tokens,
            self.spec_state_indices_tensor[: self.num_spec_decodes, 0],
            self.spec_query_start_loc[: self.num_spec_decodes + 1],
            request_indices=self.recoverssm_request_indices,
            state_indices_addrs=self.recoverssm_state_indices_addrs,
        )
        return None


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    kv_cache_spec: MambaSpec
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.checkpoint_builder = MambaPrefillCheckpointBuilder(
            vllm_config, kv_cache_spec
        )
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        self.gdn_prefill_backend: Literal["triton", "flashinfer", "cutedsl"]
        _, self.gdn_prefill_backend = _resolve_gdn_prefill_backend(vllm_config)

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)
        # RecoverSSM verifies every draft off the checkpoint slot, so spec
        # decodes need one state slot instead of one per token.
        self.use_recoverssm = vllm_config.cache_config.use_kda_recoverssm
        self.spec_state_slots = 1 if self.use_recoverssm else self.num_spec + 1
        self.recoverssm_context: KDARecoverSSMCommitContext | None = None

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        # update_block_table() keeps the source group's batch-level FULL graph
        # buffers, so only MRV2, which also reuses metadata at capture, may use
        # it.
        self.supports_update_block_table = (
            vllm_config.use_v2_model_runner
            and device.type == "cuda"
            # Not isinstance: KDA's RecoverSSM/checkpoint metadata is per group.
            and type(self) is GDNAttentionMetadataBuilder
        )
        if self.supports_update_block_table:
            # Opts into MRV2's CUDA-only aligned-index precompute.
            self.mamba_aligned_state_indices: torch.Tensor | None = None

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.spec_state_slots),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    def _get_recoverssm_context(self) -> "KDARecoverSSMCommitContext":
        if self.recoverssm_context is None:
            from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
                KDARecoverSSMCommitContext,
            )

            forward_context = self.vllm_config.compilation_config.static_forward_context
            self.recoverssm_context = KDARecoverSSMCommitContext.create(
                [forward_context[layer_name] for layer_name in self.layer_names],
                spec_query_len=1 + self.num_spec,
                max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            )
        return self.recoverssm_context

    def _build_chunk_metadata(
        self,
        prefill_query_start_loc: torch.Tensor,
        prefill_query_start_loc_cpu: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

        if self.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            assert prefill_query_start_loc is not None
            assert prefill_query_start_loc_cpu is not None
            total_tokens = int(prefill_query_start_loc_cpu[-1].item())
            return prepare_metadata_cutedsl(
                prefill_query_start_loc,
                total_tokens,
                FLA_CHUNK_SIZE,
            )

        # Only prefill batches use FLA chunk ops.
        # Pre-compute on CPU and async-copy to GPU to avoid
        # GPU→CPU sync (.tolist()) in prepare_chunk_indices.
        from vllm.third_party.flash_linear_attention.ops.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )

        assert prefill_query_start_loc_cpu is not None
        return (
            async_tensor_h2d(
                prepare_chunk_indices(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
            async_tensor_h2d(
                prepare_chunk_offsets(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        aligned_state_indices = getattr(self, "mamba_aligned_state_indices", None)
        if aligned_state_indices is not None:
            block_table_tensor = aligned_state_indices[: m.num_reqs]
        else:
            block_table_tensor = mamba_get_block_table_tensor(
                m.block_table_tensor,
                m.seq_lens,
                self.kv_cache_spec,
                self.vllm_config.cache_config.mamba_cache_mode,
            )

        uniform_spec_sequence_length = None
        recoverssm_request_indices: torch.Tensor | None = None
        spec_sequence_masks_cpu: torch.Tensor | None = None
        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            if self.use_recoverssm and m.is_prefilling is not None:
                # As in KDA: RecoverSSM keeps the extended conv window, so
                # draft-less decodes also take the spec path.
                query_lens_cpu = query_start_loc_cpu.diff()
                spec_sequence_masks_cpu |= ~m.is_prefilling & (query_lens_cpu > 0)
                spec_sequence_masks_cpu &= query_lens_cpu <= self.num_spec + 1
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0 or (
                not self.use_recoverssm
                and num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum().item()
                == 0
            ):
                num_spec_decodes = 0
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = async_tensor_h2d(
                    spec_sequence_masks_cpu, device=query_start_loc.device
                )

        if spec_sequence_masks is None:
            # V2 already excludes prefills from full decode graphs via
            # has_prefill. Classify first chunks as prefills to mask recycled
            # state; resumed one-token chunks can still use the decode kernels.
            assert m.seq_lens_cpu_upper_bound is not None
            query_lens_cpu = query_start_loc_cpu.diff()
            no_prior_state = (query_lens_cpu > 0) & (
                m.seq_lens_cpu_upper_bound <= query_lens_cpu
            )
            # Capture batches also have seq_len == query_len, but are not
            # prefills.
            if m.is_prefilling is not None:
                no_prior_state &= m.is_prefilling
            else:
                no_prior_state = torch.zeros_like(no_prior_state)
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    m.replace(is_prefilling=no_prior_state),
                    decode_threshold=1,
                    treat_short_extends_as_decodes=False,
                )
            )
            # Exclude trailing padding from both prefill counts.
            if num_prefills:
                num_prefills -= int((query_lens_cpu[num_decodes:] == 0).sum())
                num_prefill_tokens = (
                    int(query_start_loc_cpu[num_decodes + num_prefills])
                    - num_decode_tokens
                )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            non_spec_sequence_masks_cpu = ~spec_sequence_masks_cpu
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            spec_query_lens_cpu = query_lens_cpu[spec_sequence_masks_cpu]
            if spec_query_lens_cpu.numel() > 0:
                first_spec_sequence_length = int(spec_query_lens_cpu[0])
                if first_spec_sequence_length > 0 and bool(
                    torch.all(spec_query_lens_cpu == first_spec_sequence_length)
                ):
                    uniform_spec_sequence_length = first_spec_sequence_length

            # Use CPU tensors to avoid CPU-GPU sync
            non_spec_query_lens_cpu = query_lens_cpu[non_spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            # Exclude zero-length padded sequences from prefill count.
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            # num_decodes and num_spec_decodes are mutually exclusive.
            # Reclassify non-spec decodes as prefills when spec decodes
            # exist — the prefill kernel handles 1-token sequences with
            # initial state correctly, producing identical results.
            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.spec_state_slots
                ]
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = torch.argsort(spec_token_masks, stable=True)
                if self.use_recoverssm:
                    recoverssm_request_indices = async_tensor_h2d(
                        spec_sequence_masks_cpu.nonzero(as_tuple=True)[0],
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.spec_state_slots
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    non_spec_sequence_masks_cpu, 0
                ]

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks_cpu],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            if self.use_recoverssm:
                # The commit already folded the accepted tokens into the
                # checkpoint and conv window, so each step starts fresh.
                num_accepted_tokens = torch.ones_like(
                    num_accepted_tokens[:num_spec_decodes]
                )
            else:
                num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        if num_prefills > 0:
            # In a mixed non-spec batch, decodes are peeled off to the recurrent
            # kernel (decode-first front slice), so build chunk metadata from the
            # rebased prefill-only cu_seqlens; otherwise use the full non-spec one.
            # _forward_core keys off the same condition, so they agree.
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                assert non_spec_state_indices_tensor is not None
                prefill_query_start_loc = (
                    non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                )
                prefill_query_start_loc_cpu = (
                    non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
                prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
                prefill_state_indices = non_spec_state_indices_tensor

            chunk_indices, chunk_offsets = self._build_chunk_metadata(
                prefill_query_start_loc,
                prefill_query_start_loc_cpu,
                query_start_loc.device,
            )

        if num_prefills > 0:
            context_lens_tensor = m.compute_num_computed_tokens()
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
        else:
            has_initial_state = None

        checkpoint: MambaPrefillCheckpointMetadata | None = None
        if num_prefills > 0:
            request_rows = list(range(m.num_reqs))
            if spec_sequence_masks_cpu is not None:
                request_rows = (~spec_sequence_masks_cpu).nonzero().flatten().tolist()
            checkpoint = self.checkpoint_builder.build(m, request_rows)

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
        # token-padded for FULL graph replay, but the GDN state/query/accepted
        # metadata below is indexed by request.
        batch_size = m.num_reqs

        if self._stage_spec_decode(
            num_prefills, num_decodes, num_spec_decodes, num_spec_decode_tokens
        ):
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if self._stage_decode(num_prefills, num_decodes, num_spec_decodes):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            checkpoint=checkpoint,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_sequence_masks_cpu=spec_sequence_masks_cpu,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            uniform_spec_sequence_length=uniform_spec_sequence_length,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            recoverssm_context=(
                self._get_recoverssm_context()
                if self.use_recoverssm and num_spec_decodes > 0
                else None
            ),
            recoverssm_request_indices=recoverssm_request_indices,
        )
        return attn_metadata

    def _stage_spec_decode(
        self,
        num_prefills: int,
        num_decodes: int,
        num_spec_decodes: int,
        num_spec_decode_tokens: int,
    ) -> bool:
        """Whether spec-decode metadata goes into the FULL cudagraph buffers."""
        return (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        )

    def _stage_decode(
        self, num_prefills: int, num_decodes: int, num_spec_decodes: int
    ) -> bool:
        """Whether decode metadata goes into the FULL cudagraph buffers."""
        return (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        )

    def update_block_table(
        self,
        metadata: GDNAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> GDNAttentionMetadata:
        """Re-gather this group's state indices. The other fields are
        batch-level and stay shared with ``metadata``."""
        m = metadata
        if self.vllm_config.cache_config.mamba_cache_mode == "align":
            assert self.mamba_aligned_state_indices is not None
            blk_table = self.mamba_aligned_state_indices
        masks = m.spec_sequence_masks_cpu
        spec_indices = non_spec_indices = prefill_indices = None
        if masks is None:
            non_spec_indices = blk_table[:, 0]
            if m.num_prefills > 0:
                prefill_indices = non_spec_indices[m.num_decodes :]
        else:
            spec_indices = blk_table[masks, : self.spec_state_slots]
            if m.num_prefills > 0:
                non_spec_indices = prefill_indices = blk_table[~masks, 0]

        if self._stage_spec_decode(
            m.num_prefills, m.num_decodes, m.num_spec_decodes, m.num_spec_decode_tokens
        ):
            assert m.spec_state_indices_tensor is not None
            batch_size = m.spec_state_indices_tensor.shape[0]
            self.spec_state_indices_tensor[: m.num_spec_decodes].copy_(
                spec_indices, non_blocking=True
            )
            spec_indices = self.spec_state_indices_tensor[:batch_size]
            spec_indices[m.num_spec_decodes :].fill_(NULL_BLOCK_ID)

        if self._stage_decode(m.num_prefills, m.num_decodes, m.num_spec_decodes):
            assert m.non_spec_state_indices_tensor is not None
            batch_size = m.non_spec_state_indices_tensor.shape[0]
            self.non_spec_state_indices_tensor[: m.num_decodes].copy_(
                non_spec_indices, non_blocking=True
            )
            non_spec_indices = self.non_spec_state_indices_tensor[:batch_size]
            non_spec_indices[m.num_decodes :].fill_(NULL_BLOCK_ID)

        return replace(
            m,
            spec_state_indices_tensor=spec_indices,
            non_spec_state_indices_tensor=non_spec_indices,
            prefill_state_indices=prefill_indices,
            recoverssm_context=(
                self._get_recoverssm_context()
                if m.recoverssm_context is not None
                else None
            ),
        )

    # (dst ptrs, src ptrs, src strides) -> device pointer/stride tensors.
    _spec_staging_ptrs: dict[tuple, tuple[torch.Tensor, ...]] = {}

    @classmethod
    def update_block_tables(  # type: ignore[override]
        cls,
        metadata: GDNAttentionMetadata,
        builders: list["GDNAttentionMetadataBuilder"],  # type: ignore[override]
        blk_tables: list[torch.Tensor],
        slot_mappings: list[torch.Tensor],
    ) -> list[GDNAttentionMetadata]:
        """Spec-decode steps staged into the FULL graph buffers copy every
        group's state indices with one kernel; the per-group path costs a mask
        gather, a copy and a fill per group on the host. Other batches fall
        back to ``update_block_table``."""
        m = metadata
        b0 = builders[0]
        masks = m.spec_sequence_masks_cpu
        if (
            len(builders) == 1
            or masks is None
            or b0.vllm_config.cache_config.mamba_cache_mode == "align"
            or m.spec_state_indices_tensor is None
            or not b0._stage_spec_decode(
                m.num_prefills,
                m.num_decodes,
                m.num_spec_decodes,
                m.num_spec_decode_tokens,
            )
            # Spec rows must be the leading rows (padding at the back).
            or not bool(masks[: m.num_spec_decodes].all())
        ):
            metadatas = super().update_block_tables(
                m, builders, blk_tables, slot_mappings
            )
            if m.recoverssm_context is None or len(builders) == 1:
                return metadatas
            # RecoverSSM (e.g. mixed prefill/decode steps): still one commit
            # for all groups, reading this step's per-group index tensors.
            addrs = async_tensor_h2d(
                [
                    md.spec_state_indices_tensor.data_ptr()  # type: ignore[union-attr]
                    for b, md in zip(builders, metadatas)
                    for _ in b.layer_names
                ],
                dtype=torch.int64,
                device=blk_tables[0].device,
            )
            context = cls._get_group_recoverssm_context(builders)
            return [
                replace(
                    md,
                    recoverssm_context=context if i == 0 else None,
                    recoverssm_state_indices_addrs=addrs if i == 0 else None,
                )
                for i, md in enumerate(metadatas)
            ]

        key = (
            tuple(b.spec_state_indices_tensor.data_ptr() for b in builders),
            tuple(t.data_ptr() for t in blk_tables),
            tuple(t.stride(0) for t in blk_tables),
        )
        ptrs = cls._spec_staging_ptrs.get(key)
        if ptrs is None:
            assert all(t.dtype == torch.int32 for t in blk_tables)
            device = blk_tables[0].device
            ptrs = tuple(
                torch.tensor(v, dtype=torch.int64, device=device) for v in key
            )
            cls._spec_staging_ptrs[key] = ptrs
        dst_ptrs, src_ptrs, src_strides = ptrs

        batch_size = m.spec_state_indices_tensor.shape[0]
        width = b0.spec_state_slots
        _stage_spec_state_indices_kernel[(len(builders), batch_size)](
            src_ptrs,
            src_strides,
            dst_ptrs,
            m.num_spec_decodes,
            WIDTH=width,
            BLOCK=triton.next_power_of_2(width),
            NULL_ID=NULL_BLOCK_ID,
        )
        # RecoverSSM: one commit for all these groups (the first carries it),
        # reading each group's state indices from its staging buffer.
        recoverssm_context = (
            cls._get_group_recoverssm_context(builders)
            if m.recoverssm_context is not None
            else None
        )
        return [
            replace(
                m,
                spec_state_indices_tensor=b.spec_state_indices_tensor[:batch_size],
                non_spec_state_indices_tensor=None,
                prefill_state_indices=None,
                recoverssm_context=recoverssm_context if i == 0 else None,
            )
            for i, b in enumerate(builders)
        ]

    # builder ids -> RecoverSSM commit context over all their layers.
    _group_recoverssm_contexts: dict[tuple, "KDARecoverSSMCommitContext"] = {}

    @classmethod
    def _get_group_recoverssm_context(
        cls, builders: list["GDNAttentionMetadataBuilder"]
    ) -> "KDARecoverSSMCommitContext":
        key = tuple(id(b) for b in builders)
        context = cls._group_recoverssm_contexts.get(key)
        if context is None:
            from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
                KDARecoverSSMCommitContext,
            )

            b0 = builders[0]
            forward_context = b0.vllm_config.compilation_config.static_forward_context
            context = KDARecoverSSMCommitContext.create(
                [forward_context[name] for b in builders for name in b.layer_names],
                spec_query_len=1 + b0.num_spec,
                max_num_reqs=b0.vllm_config.scheduler_config.max_num_seqs,
                state_indices=[
                    b.spec_state_indices_tensor for b in builders for _ in b.layer_names
                ],
            )
            cls._group_recoverssm_contexts[key] = context
        return context

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = torch.diff(m.query_start_loc_cpu).sub_(1)
        assert num_decode_draft_tokens_cpu.shape == num_accepted_tokens.shape

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)


@triton.jit(do_not_specialize=["num_spec_decodes"])
def _stage_spec_state_indices_kernel(
    src_ptrs,  # [num_groups] int32 block table base pointers
    src_strides,  # [num_groups]
    dst_ptrs,  # [num_groups] spec_state_indices_tensor base pointers
    num_spec_decodes,
    WIDTH: tl.constexpr,  # spec_state_slots
    BLOCK: tl.constexpr,
    NULL_ID: tl.constexpr,
):
    # dst[g][row, :WIDTH] = src[g][row, :WIDTH], NULL_ID for padded rows.
    group = tl.program_id(0)
    row = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    mask = offs < WIDTH
    dst = tl.load(dst_ptrs + group).to(tl.pointer_type(tl.int32)) + row * WIDTH
    if row < num_spec_decodes:
        src = tl.load(src_ptrs + group).to(tl.pointer_type(tl.int32))
        src += row * tl.load(src_strides + group)
        vals = tl.load(src + offs, mask=mask)
    else:
        vals = tl.full([BLOCK], NULL_ID, tl.int32)
    tl.store(dst + offs, vals, mask=mask)
