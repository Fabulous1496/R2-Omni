from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .qwen_sspa_runtime import (
    _QwenSSPARuntime,
    _RuntimeLayerPlan,
    _tensor_bytes,
)
from .sspa_config import SSPAConfig
from ..kernels.triton_fourq_attention import triton_mixed_reference
from ..kernels.triton_qwen_mrope import qwen_qk_mrope, triton_fwht_128
from ..kernels.triton_fourq_splitk import (
    append_recent_fp_triton,
    allocated_workspace_bytes,
    allocate_splitk_workspace,
    prepare_splitk_executor,
    required_splitk_blocks,
    stack_homogeneous_splitk_executors,
)
from ..policy.dual_score import (
    compute_av_alignment_score,
    compute_query_attention_probabilities,
    compute_k_conditioned_value_residual_actions,
    compute_output_action_sensitivity,
    prepare_k_conditioned_value_context,
    fuse_semantic_value_priority,
    infer_temporal_chunk_ids,
    normalize_scores_by_modality,
    resolve_cross_modal_alignment,
    reduce_query_attention_probabilities,
)
from ..policy.four_quadrant import PageClass
from ..policy.precision_schedule import (
    assign_residual_pages_to_precision,
    build_sigmoid_precision_targets,
    evaluate_residual_page_assignment_components,
)
from ..policy.page_assignment import (
    PageAssignment,
    aggregate_page_scores,
    build_page_table,
    project_token_bits_to_pages,
)
from ..runtime.cache_plan import build_layer_compression_plan
from ..runtime.dynamic_fourq_cache import DynamicCacheStats, append_generated_token
from ..runtime.precision_pool import (
    CacheVariant,
    ThreePrecisionCache,
    build_three_precision_cache,
    precision_cache_active_storage_breakdown,
    precision_cache_storage_breakdown,
)
from ..runtime.int2_reference import normalized_fwht


class QwenSSPAAttentionManager(_QwenSSPARuntime):
    """Qwen integration for Semantic-Sensitivity Precision Allocation."""

    def __init__(
        self,
        model: Any,
        layer_strategy: dict[int, dict[str, Any]],
        assignment_config: SSPAConfig | dict[str, Any],
        *,
        kernel_backend: str = "triton_mixed_fused_splitk",
        cache_variant: CacheVariant = "v_fourq_k_int4_compact",
        share_workspace: bool = True,
        partial_output_dtype: str = "float16",
        fallback_on_error: bool = False,
        verbose: bool = False,
        collect_timings: bool = False,
        validate_kernel_output: bool = False,
        prefill_executor_warmup: bool = False,
        reuse_workspace_across_requests: bool = True,
        collect_policy_records: bool = False,
        validate_policy_state: bool = False,
        matched_fp_until_layer: int | None = None,
        key_quantization_superblock: int = 32,
        key_hadamard: bool = True,
        use_tensor_cores_qk: bool = True,
        use_tensor_cores_pv: bool = False,
        value_hadamard: bool = True,
        splitk_block_n: int = 0,
        splitk_num_warps: int = 0,
        splitk_num_stages: int = 0,
        splitk_reduce_num_warps: int = 0,
        splitk_hierarchical_reduce: bool = True,
        splitk_hierarchical_group_blocks: int = 64,
        throughput_use_tensor_cores_pv: bool = False,
        throughput_direct_reduce_max_splits: int = 0,
    ) -> None:
        if kernel_backend not in {"triton_mixed_fused_splitk", "triton_mixed_reference"}:
            raise ValueError("SSPA backend must be fused_splitk or mixed_reference")
        resolved_config = (
            SSPAConfig(**assignment_config)
            if isinstance(assignment_config, dict)
            else assignment_config
        )
        super().__init__(
            model, layer_strategy, resolved_config,
            kernel_backend="triton_mixed_v2", fallback_on_error=fallback_on_error,
            verbose=verbose, collect_timings=collect_timings,
            validate_kernel_output=validate_kernel_output,
            allow_layer0_compression=True,
        )
        self.kernel_backend = kernel_backend
        if cache_variant not in {
            "v_fourq_k_int2_compact",
            "v_fourq_k_int4_compact",
            "v_fourq_k_bf16",
        }:
            raise ValueError(f"unsupported cache_variant: {cache_variant}")
        self.cache_variant: CacheVariant = cache_variant
        if partial_output_dtype not in {"float16", "float32"}:
            raise ValueError("partial_output_dtype must be float16 or float32")
        self.share_workspace = bool(share_workspace)
        self.partial_output_dtype = partial_output_dtype
        self.prefill_executor_warmup = bool(prefill_executor_warmup)
        self.reuse_workspace_across_requests = bool(
            reuse_workspace_across_requests
        )
        self.collect_policy_records = bool(collect_policy_records)
        self.validate_policy_state = bool(validate_policy_state)
        self.matched_fp_until_layer = (
            None
            if matched_fp_until_layer is None or int(matched_fp_until_layer) < 0
            else int(matched_fp_until_layer)
        )
        if int(key_quantization_superblock) not in {32, 64, 128}:
            raise ValueError("key_quantization_superblock must be 32, 64, or 128")
        self.key_quantization_superblock = int(key_quantization_superblock)
        self.key_hadamard = bool(key_hadamard)
        self.use_tensor_cores_qk = bool(use_tensor_cores_qk)
        self.use_tensor_cores_pv = bool(use_tensor_cores_pv)
        self.throughput_use_tensor_cores_pv = bool(
            throughput_use_tensor_cores_pv
        )
        if int(throughput_direct_reduce_max_splits) not in {0, 16, 32, 64, 128}:
            raise ValueError(
                "throughput_direct_reduce_max_splits must be 0/16/32/64/128"
            )
        self.throughput_direct_reduce_max_splits = int(
            throughput_direct_reduce_max_splits
        )
        if int(splitk_block_n) not in {0, 64, 128}:
            raise ValueError("splitk_block_n must be 0 (auto), 64, or 128")
        self.splitk_block_n = int(splitk_block_n)
        if int(splitk_num_warps) not in {0, 4, 8}:
            raise ValueError("splitk_num_warps must be 0 (auto), 4, or 8")
        if int(splitk_num_stages) not in {0, 1, 2}:
            raise ValueError("splitk_num_stages must be 0 (auto), 1, or 2")
        if int(splitk_reduce_num_warps) not in {0, 4, 8}:
            raise ValueError(
                "splitk_reduce_num_warps must be 0 (auto), 4, or 8"
            )
        self.splitk_num_warps = int(splitk_num_warps)
        self.splitk_num_stages = int(splitk_num_stages)
        self.splitk_reduce_num_warps = int(splitk_reduce_num_warps)
        if int(splitk_hierarchical_group_blocks) not in {16, 32, 64}:
            raise ValueError(
                "splitk_hierarchical_group_blocks must be 16, 32, or 64"
            )
        self.splitk_hierarchical_reduce = bool(splitk_hierarchical_reduce)
        self.splitk_hierarchical_group_blocks = int(
            splitk_hierarchical_group_blocks
        )
        self.value_hadamard = bool(value_hadamard)

        self._shared_workspaces: dict[tuple[torch.device, int, int], Any] = {}
        self._group_page_assignments: dict[int, Any] = {}
        self._precision_targets = build_sigmoid_precision_targets(
            len(self._scoring_layers),
            shallow_v_payload_bit_ceiling=(
                self.assignment_config.shallow_v_payload_bit_ceiling
            ),
            deep_v_payload_bit_ceiling=(
                self.assignment_config.deep_v_payload_bit_ceiling
            ),
            steepness=self.assignment_config.precision_sigmoid_steepness,
        )
        self._page_table = None
        self._token_page_ids: torch.Tensor | None = None
        self._qk_rope_buffer: torch.Tensor | None = None
        self._qk_rope_batch_buffer: tuple[torch.Tensor, torch.Tensor] | None = None
        self._throughput_batch_size = 1
        self._native_prefill_batch_size = 1
        self._native_batched_prefill = False
        self._native_batched_workspaces: dict[tuple[Any, ...], Any] = {}
        self._layer0_length_buffers: tuple[torch.Tensor, torch.Tensor] | None = None

    def _average_key_bits(self) -> float:
        bit_tokens = 0.0
        token_count = 0
        for state in self._states.values():
            cache: ThreePrecisionCache | None = getattr(state, "fourq_cache", None)
            if cache is None:
                continue
            recent_tokens = cache.recent_fp_pool.token_count
            historical_tokens = (
                cache.fp_pool.token_count
                + cache.int4_pool.token_count
                + cache.int2_pool.token_count
            )
            if cache.matched_fp or cache.cache_variant == "v_fourq_k_bf16":
                historical_bits = 16.0
            else:
                historical_bits = 2.0 if cache.uniform_int2_key else 4.0
            bit_tokens += historical_bits * historical_tokens + 16.0 * recent_tokens
            token_count += historical_tokens + recent_tokens
        return bit_tokens / max(token_count, 1)

    def _kernel_key(self, key: torch.Tensor) -> torch.Tensor:
        if not self.key_hadamard:
            return key
        if key.is_cuda:
            return triton_fwht_128(key)
        return normalized_fwht(key).to(key.dtype)

    def _kernel_query(self, query: torch.Tensor) -> torch.Tensor:
        if not self.key_hadamard:
            return query
        return normalized_fwht(query).to(query.dtype)

    def _kernel_value(self, value: torch.Tensor) -> torch.Tensor:
        if not self.value_hadamard:
            return value
        if value.is_cuda:
            return triton_fwht_128(value)
        return normalized_fwht(value).to(value.dtype)

    def _matched_fp_for_layer(self, layer_id: int) -> bool:
        """Keep early compressed groups BF16 while allowing late low-bit groups."""
        return (
            self.matched_fp_until_layer is not None
            and int(layer_id) <= self.matched_fp_until_layer
        )

    def prepare(self, modality: torch.Tensor, *, modality_metadata: dict[str, Any] | None = None, sample_id: Any = None) -> None:
        super().prepare(modality, modality_metadata=modality_metadata, sample_id=sample_id)
        self._throughput_batch_size = 1
        self._native_prefill_batch_size = 1
        self._native_batched_prefill = False
        self._qk_rope_batch_buffer = None
        self._native_batched_workspaces.clear()
        self._group_page_assignments.clear()
        self._page_table = None
        self._token_page_ids = None
        self._layer0_length_buffers = None
        if not self.reuse_workspace_across_requests:
            self._shared_workspaces.clear()

    def enable_native_prefill_batch(self, batch_size: int) -> None:
        """Enable an opt-in independent-request prefill/decode batch.

        The production B=1 path never enters this mode.  A native batch runs
        one real batched prefill, packs each request's own K/V payload, and
        combines those independent payloads into the request-major decode ABI.
        Requests must currently have the same effective sequence length and
        modality layout, matching OScaR's fixed-context throughput protocol.
        """
        batch_size = int(batch_size)
        if batch_size < 2:
            raise ValueError("native prefill batching requires batch_size >= 2")
        if any(state.migrated for state in self._states.values()):
            raise RuntimeError("native prefill batching must be enabled before prefill")
        if self._throughput_batch_size != 1:
            raise RuntimeError("a decode batch is already active")
        self._native_prefill_batch_size = batch_size
        self._throughput_batch_size = batch_size
        self._native_batched_prefill = True

    def enable_homogeneous_decode_batch(self, batch_size: int) -> None:
        """Clone finalized B=1 caches into an opt-in decode throughput batch."""
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("decode throughput batch size must be positive")
        if batch_size == 1:
            self._throughput_batch_size = 1
            return
        if self._throughput_batch_size != 1:
            raise RuntimeError("decode throughput batch was already activated")
        shared_workspaces: dict[tuple[Any, ...], Any] = {}
        for layer_id, state in sorted(self._states.items()):
            executor = getattr(state, "fourq_executor", None)
            if not state.migrated or executor is None:
                raise RuntimeError(
                    f"layer {layer_id} must finish B=1 prefill before batching"
                )
            workspace_key = (
                executor.workspace.output.device,
                executor.workspace.block_n,
                executor.workspace.total_blocks,
                executor.workspace.hierarchical_group_blocks,
                executor.workspace.hierarchical_group_capacity,
                executor.num_query_heads,
                executor.head_dim,
            )
            batched = stack_homogeneous_splitk_executors(
                (executor,) * batch_size,
                workspace=shared_workspaces.get(workspace_key),
                throughput_use_tensor_cores_pv=(
                    self.throughput_use_tensor_cores_pv
                ),
                throughput_direct_reduce_max_splits=(
                    self.throughput_direct_reduce_max_splits
                ),
            )
            shared_workspaces[workspace_key] = batched.workspace
            state.fourq_executor = batched
            state.throughput_batch_size = batch_size
        self._throughput_batch_size = batch_size

    def _set_layer0_length_proxy(
        self,
        cache: Any,
        length: int,
        device: torch.device,
    ) -> None:
        """Expose global DynamicCache length without retaining layer-0 BF16 KV.

        HuggingFace derives the next ``cache_position`` from
        ``key_cache[0].shape[-2]``.  When layer 0 itself is owned by the packed
        runtime, a pair of one-byte-per-position tensors supplies only that
        shape contract; attention never reads their payload.  Capacity is
        reserved once for the requested generation horizon, avoiding a device
        allocation on every decode token.
        """
        length = int(length)
        if length <= 0:
            raise ValueError("layer-0 length proxy must be positive")
        capacity = length + int(self.assignment_config.fp_append_capacity)
        proxy_batch = (
            self._native_prefill_batch_size
            if self._native_batched_prefill
            else self._throughput_batch_size
        )
        buffers = self._layer0_length_buffers
        if (
            buffers is None
            or int(buffers[0].shape[0]) != proxy_batch
            or int(buffers[0].shape[2]) < length
            or buffers[0].device != device
        ):
            buffers = (
                torch.empty(
                    (proxy_batch, 1, capacity, 1),
                    device=device,
                    dtype=torch.uint8,
                ),
                torch.empty(
                    (proxy_batch, 1, capacity, 1),
                    device=device,
                    dtype=torch.uint8,
                ),
            )
            self._layer0_length_buffers = buffers
        self._set_cache_layer_tensors(
            cache,
            0,
            buffers[0][:, :, :length, :],
            buffers[1][:, :, :length, :],
        )
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = length

    def _release_dynamic_layer_cache(
        self,
        cache: Any,
        layer_id: int,
        *,
        logical_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if layer_id == 0:
            self._set_layer0_length_proxy(cache, logical_length, device)
        else:
            self._clear_layer_cache(cache, layer_id, device, dtype)

    def _decode_cache_position(
        self,
        layer_id: int,
        cache: Any,
        fallback: int,
        cache_position: Any,
    ) -> int:
        if layer_id != 0:
            return self._cache_position(cache, fallback)
        # Before layer 0 runs, its length proxy describes the previous decode
        # step. Use the explicit HF position to distinguish a new token from a
        # repeated forward at the same position. This is one scalar sync per
        # decode step, not per compressed layer.
        if torch.is_tensor(cache_position) and cache_position.numel():
            return int(cache_position.reshape(-1)[-1].item())
        return int(fallback)

    def _dynamic_cache_payload_bytes(self) -> tuple[int, int]:
        """Return physical DynamicCache bytes and the layer-0 proxy subset."""
        if self._last_cache is None:
            return 0, 0
        if hasattr(self._last_cache, "key_cache"):
            layer_tensors = list(zip(self._last_cache.key_cache, self._last_cache.value_cache))
        else:
            layer_tensors = [
                (layer.keys, layer.values)
                for layer in getattr(self._last_cache, "layers", [])
                if hasattr(layer, "keys") and hasattr(layer, "values")
            ]
        full_bytes = sum(
            _tensor_bytes(item)
            for key, value in layer_tensors
            for item in (key, value)
        )
        proxy_bytes = 0
        if self._layer0_length_buffers is not None:
            logical_proxy = 0
            if layer_tensors:
                logical_proxy += _tensor_bytes(layer_tensors[0][0])
                logical_proxy += _tensor_bytes(layer_tensors[0][1])
            proxy_bytes = sum(_tensor_bytes(item) for item in self._layer0_length_buffers)
            full_bytes += proxy_bytes - logical_proxy
        return full_bytes, proxy_bytes

    def _qk_rope_output(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(query.shape[0])
        if batch_size > 1:
            tokens = int(query.shape[2])
            query_shape = (batch_size, int(query.shape[1]), tokens, 128)
            key_shape = (batch_size, int(key.shape[1]), tokens, 128)
            buffers = self._qk_rope_batch_buffer
            if (
                buffers is None
                or tuple(buffers[0].shape) != query_shape
                or tuple(buffers[1].shape) != key_shape
                or buffers[0].device != query.device
                or buffers[1].device != query.device
                or buffers[0].dtype != query.dtype
                or buffers[1].dtype != query.dtype
            ):
                buffers = (
                    torch.empty(query_shape, device=query.device, dtype=query.dtype),
                    torch.empty(key_shape, device=query.device, dtype=query.dtype),
                )
                self._qk_rope_batch_buffer = buffers
            return buffers
        heads = int(query.shape[1]) + int(key.shape[1])
        buffer = self._qk_rope_buffer
        if (
            buffer is None
            or buffer.shape != (heads, int(query.shape[2]), 128)
            or buffer.device != query.device
            or buffer.dtype != query.dtype
        ):
            buffer = torch.empty(
                (heads, int(query.shape[2]), 128),
                device=query.device,
                dtype=query.dtype,
            )
            self._qk_rope_buffer = buffer
        query_heads = int(query.shape[1])
        query_out = buffer[:query_heads].view_as(query)
        key_out = buffer[query_heads:].view_as(key)
        return query_out, key_out

    def _tokens_for_pages(self, table: Any, page_ids: torch.Tensor) -> torch.Tensor:
        if self._token_page_ids is None:
            self._token_page_ids = torch.repeat_interleave(
                torch.arange(table.num_pages, device=page_ids.device), table.lengths
            )
        selected = torch.zeros(table.num_pages, device=page_ids.device, dtype=torch.bool)
        selected[page_ids] = True
        return selected[self._token_page_ids].nonzero(as_tuple=False).flatten()

    def _physical_key_order(
        self, table: Any, page_classes: torch.Tensor
    ) -> torch.Tensor:
        """Mirror the deployed FP16-V/INT4-V/INT2-V compact K order."""
        fp = (
            (page_classes == int(PageClass.FP_PAGE))
            | (page_classes == int(PageClass.PROTECTED_FP_PAGE))
            | (page_classes == int(PageClass.RECENT_FP_PAGE))
        ).nonzero(as_tuple=False).flatten()
        int4 = (page_classes == int(PageClass.INT4_PAGE)).nonzero(
            as_tuple=False
        ).flatten()
        int2 = (page_classes == int(PageClass.INT2_PAGE)).nonzero(
            as_tuple=False
        ).flatten()
        evicted = (page_classes == int(PageClass.EVICTED_PAGE)).nonzero(
            as_tuple=False
        ).flatten()
        order = torch.cat(
            tuple(
                self._tokens_for_pages(table, pages)
                for pages in (fp, int4, int2, evicted)
            )
        )
        if int(order.numel()) != int(table.token_indices.numel()):
            raise RuntimeError("physical K order does not cover every page-table token")
        return order

    @staticmethod
    def _key_calibration_partition_signature(
        table: Any,
        class_values: tuple[int, ...],
        superblock_size: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Canonical token sets sharing one affine K calibration block.

        Per-channel affine min/max calibration is invariant to both token order
        inside a block and block order.  Two V-tier layouts with this same
        signature therefore produce exactly the same logical INT4 K tensor,
        even though their packed physical order differs.
        """
        if superblock_size <= 0 or len(class_values) != table.num_pages:
            raise ValueError("invalid K partition geometry")
        offsets = [int(value) for value in table.offsets.tolist()]
        lengths = [int(value) for value in table.lengths.tolist()]
        fp_classes = {
            int(PageClass.FP_PAGE),
            int(PageClass.PROTECTED_FP_PAGE),
            int(PageClass.RECENT_FP_PAGE),
        }
        tier_sets = (
            fp_classes,
            {int(PageClass.INT4_PAGE)},
            {int(PageClass.INT2_PAGE)},
            {int(PageClass.EVICTED_PAGE)},
        )
        order: list[int] = []
        covered_pages = 0
        for tier in tier_sets:
            for page, value in enumerate(class_values):
                if value in tier:
                    start = offsets[page]
                    order.extend(range(start, start + lengths[page]))
                    covered_pages += 1
        if covered_pages != len(class_values):
            raise ValueError("unsupported page class in K partition signature")
        blocks = [
            tuple(sorted(order[start : start + superblock_size]))
            for start in range(0, len(order), superblock_size)
        ]
        return tuple(sorted(blocks))

    def _assign_k_conditioned_residual(
        self,
        state: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        table: Any,
        target: Any,
        attention_probabilities: torch.Tensor,
        page_modality: torch.Tensor,
        semantic_priority: torch.Tensor,
    ) -> tuple[Any, dict[str, Any]]:
        """Generate proposals, then self-evaluate each under its own K order."""
        config = self.assignment_config
        current_classes = torch.full(
            (table.num_pages,),
            int(PageClass.INT2_PAGE),
            device=q.device,
            dtype=torch.int8,
        )
        current_assignment = None
        candidates: dict[
            tuple[int, ...], tuple[Any, float, float, float]
        ] = {}
        proposal_path: list[tuple[int, ...]] = []
        fixed_point = False
        action_cache: dict[tuple[tuple[int, ...], ...], Any] = {}
        sensitivity_cache: dict[tuple[tuple[int, ...], ...], torch.Tensor] = {}
        action_cache_hits = 0
        if semantic_priority.shape != page_modality.shape:
            raise ValueError("semantic priority must match page modality shape")
        value_context = self._measure(
            state,
            "sensitivity_ms",
            q.device,
            lambda: prepare_k_conditioned_value_context(
                q,
                k,
                v,
                table.offsets,
                table.lengths,
                int2_group_size=config.value_int2_group_size,
                reference_attention_probabilities=attention_probabilities,
            ),
        )
        def class_key(classes: torch.Tensor) -> tuple[int, ...]:
            return tuple(int(value) for value in classes.tolist())

        def build_actions(
            classes: torch.Tensor,
            signature: tuple[tuple[int, ...], ...],
        ):
            nonlocal action_cache_hits
            cached = action_cache.get(signature)
            if cached is not None:
                action_cache_hits += 1
                return cached
            physical_order = self._physical_key_order(table, classes)
            actions = self._measure(
                state,
                "sensitivity_ms",
                q.device,
                lambda: compute_k_conditioned_value_residual_actions(
                    q,
                    k,
                    v,
                    table.offsets,
                    table.lengths,
                    physical_order,
                    key_bits=(
                        2
                        if self.cache_variant == "v_fourq_k_int2_compact"
                        else 4
                    ),
                    key_superblock_size=self.key_quantization_superblock,
                    int2_group_size=config.value_int2_group_size,
                    reference_attention_probabilities=attention_probabilities,
                    value_context=value_context,
                ),
            )
            action_cache[signature] = actions
            return actions

        def self_evaluate(
            actions: Any, classes: torch.Tensor
        ) -> tuple[float, float, float, float]:
            before, after, transition_before, transition_after = self._measure(
                state,
                "sensitivity_ms",
                q.device,
                lambda: evaluate_residual_page_assignment_components(
                    actions.residual,
                    actions.int2_to_int4,
                    actions.int4_to_fp16,
                    classes,
                    query_reduction=config.residual_query_objective,
                    query_ema_decay=config.query_anchor_ema_decay,
                    query_cvar_fraction=config.residual_query_cvar_fraction,
                    transition_weight=config.residual_transition_weight,
                ),
            )
            return (
                float(before),
                float(after),
                float(transition_before),
                float(transition_after),
            )

        def build_action_priority(
            signature: tuple[tuple[int, ...], ...], actions: Any
        ) -> torch.Tensor:
            cached = sensitivity_cache.get(signature)
            if cached is not None:
                return cached
            output_sensitivity = self._measure(
                state,
                "sensitivity_ms",
                q.device,
                lambda: compute_output_action_sensitivity(
                    actions,
                    page_modality,
                    query_tail_fraction=config.residual_query_cvar_fraction,
                    transition_weight=config.residual_transition_weight,
                    normalization=config.score_normalization,
                ),
            )
            priority = fuse_semantic_value_priority(
                semantic_priority.unsqueeze(0).expand_as(output_sensitivity),
                output_sensitivity,
                config.semantic_priority_weight,
            )
            sensitivity_cache[signature] = priority
            return priority

        for _ in range(config.k_conditioned_iterations):
            key = class_key(current_classes)
            proposal_path.append(key)
            signature = self._key_calibration_partition_signature(
                table, key, self.key_quantization_superblock
            )
            actions = build_actions(current_classes, signature)
            action_priority = build_action_priority(signature, actions)
            before, after, transition_before, transition_after = self_evaluate(
                actions, current_classes
            )
            updated_assignment = self._measure(
                state,
                "sensitivity_ms",
                q.device,
                lambda: assign_residual_pages_to_precision(
                    actions.residual,
                    actions.int2_to_int4,
                    actions.int4_to_fp16,
                    table.lengths,
                    target,
                    kv_heads=int(v.shape[-3]),
                    head_dim=int(v.shape[-1]),
                    int2_group_size=config.value_int2_group_size,
                    allocation_rounds=config.residual_allocation_rounds,
                    query_reduction=config.residual_query_objective,
                    query_ema_decay=config.query_anchor_ema_decay,
                    query_cvar_fraction=config.residual_query_cvar_fraction,
                    transition_weight=config.residual_transition_weight,
                    page_modalities=page_modality,
                    modality_budget_weights=(
                        config.video_precision_weight,
                        config.audio_precision_weight,
                    ),
                    action_priority_scores=action_priority,
                    action_priority_weight=config.output_priority_weight,
                ),
            )
            if current_assignment is None:
                total_tokens = int(table.lengths.sum())
                current_assignment = replace(
                    updated_assignment,
                    page_classes=current_classes.clone(),
                    fp16_threshold=math.inf,
                    int4_threshold=math.inf,
                    fp16_tokens=0,
                    int4_tokens=0,
                    int2_tokens=total_tokens,
                    average_v_bits=2.0,
                    selected_positive_gain=0.0,
                    allocation_rounds=0,
                )
            current_assignment = replace(
                current_assignment,
                predicted_distortion_before=before,
                predicted_distortion_after=after,
                predicted_transition_distortion_before=transition_before,
                predicted_transition_distortion_after=transition_after,
                selected_positive_gain=before - after,
            )
            candidates[key] = (
                current_assignment,
                before,
                after,
                float(actions.key_reconstruction_mse),
            )
            updated_key = class_key(updated_assignment.page_classes)
            if updated_key == key:
                fixed_point = True
                break
            updated_signature = self._key_calibration_partition_signature(
                table, updated_key, self.key_quantization_superblock
            )
            if updated_signature == signature:
                # The proposal sees the exact same logical K realization.
                # Re-running the deterministic allocator would return the same
                # assignment, so record its already-computed exact objective
                # and terminate without a redundant quantize/readout pass.
                proposal_path.append(updated_key)
                updated_before = float(updated_assignment.predicted_distortion_before)
                updated_after = float(updated_assignment.predicted_distortion_after)
                updated_assignment = replace(
                    updated_assignment,
                    selected_positive_gain=updated_before - updated_after,
                )
                candidates[updated_key] = (
                    updated_assignment,
                    updated_before,
                    updated_after,
                    float(actions.key_reconstruction_mse),
                )
                current_classes = updated_assignment.page_classes
                current_assignment = updated_assignment
                fixed_point = True
                action_cache_hits += 1
                break
            current_classes = updated_assignment.page_classes
            current_assignment = updated_assignment

        # The final generated proposal has not yet been scored under the K
        # blocks induced by its own tier order. That self-evaluation is the
        # step that makes comparison valid even when proposals oscillate.
        final_key = class_key(current_classes)
        if final_key not in candidates:
            final_signature = self._key_calibration_partition_signature(
                table, final_key, self.key_quantization_superblock
            )
            actions = build_actions(current_classes, final_signature)
            build_action_priority(final_signature, actions)
            before, after, transition_before, transition_after = self_evaluate(
                actions, current_classes
            )
            assert current_assignment is not None
            current_assignment = replace(
                current_assignment,
                predicted_distortion_before=before,
                predicted_distortion_after=after,
                predicted_transition_distortion_before=transition_before,
                predicted_transition_distortion_after=transition_after,
                selected_positive_gain=before - after,
            )
            candidates[final_key] = (
                current_assignment,
                before,
                after,
                float(actions.key_reconstruction_mse),
            )

        candidate_items = list(candidates.items())
        selected_index, (selected_key, selected) = min(
            enumerate(candidate_items),
            key=lambda item: (
                item[1][1][2],
                item[1][1][0].average_v_bits,
                item[0],
            ),
        )
        assignment, before, after, key_mse = selected
        selected_signature = self._key_calibration_partition_signature(
            table, selected_key, self.key_quantization_superblock
        )
        selected_priority = sensitivity_cache.get(selected_signature)
        cycle_detected = len(proposal_path) != len(set(proposal_path))
        stats = {
            "k_conditioned_candidates_evaluated": len(candidates),
            "k_conditioned_proposal_steps": len(proposal_path),
            "k_conditioned_fixed_point": fixed_point,
            "k_conditioned_cycle_detected": cycle_detected,
            "k_conditioned_action_cache_hits": action_cache_hits,
            "k_conditioned_unique_k_partitions": len(action_cache),
            "k_conditioned_selected_candidate": selected_index,
            "k_conditioned_key_mse": key_mse,
            "predicted_distortion_before": before,
            "predicted_distortion_after": after,
            "selected_positive_gain": before - after,
            "residual_allocation_rounds_used": assignment.allocation_rounds,
            "residual_query_objective": config.residual_query_objective,
            "residual_query_cvar_fraction": config.residual_query_cvar_fraction,
            "residual_transition_weight": config.residual_transition_weight,
            "output_priority_weight": config.output_priority_weight,
            "semantic_priority_weight": config.semantic_priority_weight,
            "video_precision_weight": config.video_precision_weight,
            "audio_precision_weight": config.audio_precision_weight,
            "semantic_priority_mean": semantic_priority.mean(),
            "combined_int2_to_int4_priority_mean": (
                selected_priority[0].mean()
                if selected_priority is not None
                else torch.zeros((), device=q.device)
            ),
            "combined_int4_to_fp16_priority_mean": (
                selected_priority[1].mean()
                if selected_priority is not None
                else torch.zeros((), device=q.device)
            ),
            "predicted_transition_distortion_before": (
                assignment.predicted_transition_distortion_before
            ),
            "predicted_transition_distortion_after": (
                assignment.predicted_transition_distortion_after
            ),
            "candidate_actual_distortions": [
                value[2] for value in candidates.values()
            ],
        }
        return assignment, stats

    def _score_assignment(
        self,
        layer_id: int,
        q: torch.Tensor,
        k: torch.Tensor,
        modality: torch.Tensor,
        v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        config = self.assignment_config
        state = self._states[layer_id]
        if self._page_table is None:
            # Historical V precision is score-only: text, boundary, and recent
            # prefill pages participate in the same three-tier budget.
            unprotected = torch.zeros_like(modality, dtype=torch.bool)
            self._page_table = build_page_table(
                modality, page_size=config.page_size,
                logical_positions=torch.arange(modality.numel(), device=modality.device),
                global_token_indices=torch.arange(modality.numel(), device=modality.device),
                protected_mask=unprotected, recent_mask=unprotected,
            )
        table = self._page_table
        unit_index = self._scoring_layers.index(layer_id)
        all_tokens = table.token_indices
        attention_probabilities = self._measure(
            state,
            "scoring_ms",
            q.device,
            lambda: compute_query_attention_probabilities(q, k, all_tokens),
        )
        attention = reduce_query_attention_probabilities(
            attention_probabilities,
            reduce_queries=config.query_anchor_reduction,
            query_ema_decay=config.query_anchor_ema_decay,
        )
        temporal = self._modality_metadata.get("temporal_chunk_ids")
        temporal_chunks = (
            temporal.to(device=k.device, dtype=torch.long)
            if torch.is_tensor(temporal) and int(temporal.numel()) == int(modality.numel())
            else infer_temporal_chunk_ids(modality, config.temporal_num_chunks)
        )
        alignment = self._measure(
            state, "alignment_ms", q.device,
            lambda: compute_av_alignment_score(
                k[0].float().mean(dim=0), modality, temporal_chunks,
                all_tokens,
                topk=config.av_alignment_topk,
                num_temporal_chunks=config.temporal_num_chunks,
            ),
        )
        page_attention = aggregate_page_scores(attention, table)
        page_alignment = aggregate_page_scores(alignment, table)
        page_modality = table.modality
        resolved_page_alignment, alignment_fallback = resolve_cross_modal_alignment(
            page_attention,
            page_alignment,
            page_modality,
        )
        normalized_attention = normalize_scores_by_modality(page_attention, page_modality, config.score_normalization)
        normalized_alignment = normalize_scores_by_modality(
            resolved_page_alignment,
            page_modality,
            config.score_normalization,
        )
        semantic_fused = (
            config.attention_weight * normalized_attention
            + (1.0 - config.attention_weight) * normalized_alignment
        )
        # Semantic scores are A/V-normalized. Text and other page types are
        # outside that comparison and receive a neutral modifier rather than
        # the zero initialization used by the legacy tier score.
        semantic_priority = torch.where(
            (page_modality == 1) | (page_modality == 2),
            semantic_fused,
            torch.full_like(semantic_fused, 0.5),
        )
        target = self._precision_targets[unit_index]
        if v is None:
            raise ValueError("SSPA requires the historical V cache")
        tier_assignment, residual_stats = self._assign_k_conditioned_residual(
            state,
            q,
            k,
            v,
            table,
            target,
            attention_probabilities,
            page_modality,
            semantic_priority,
        )
        classes = tier_assignment.page_classes
        def modality_tier_tokens(modality_value: int, page_class: PageClass) -> int:
            pages = (
                (page_modality == modality_value)
                & (classes == int(page_class))
            ).nonzero(as_tuple=False).flatten()
            return (
                int(table.lengths.index_select(0, pages).sum())
                if pages.numel()
                else 0
            )

        video_fp_tokens = modality_tier_tokens(1, PageClass.FP_PAGE)
        video_int4_tokens = modality_tier_tokens(1, PageClass.INT4_PAGE)
        video_int2_tokens = modality_tier_tokens(1, PageClass.INT2_PAGE)
        audio_fp_tokens = modality_tier_tokens(2, PageClass.FP_PAGE)
        audio_int4_tokens = modality_tier_tokens(2, PageClass.INT4_PAGE)
        audio_int2_tokens = modality_tier_tokens(2, PageClass.INT2_PAGE)

        def average_modality_bits(fp: int, int4: int, int2: int) -> float:
            total = fp + int4 + int2
            return (16.0 * fp + 4.0 * int4 + 2.0 * int2) / max(total, 1)

        assignment = PageAssignment(
            table, classes,
            normalized_attention, normalized_alignment, semantic_fused,
        )
        source = int(self.layer_strategy[layer_id]["assignment_source_layer"])
        self._group_page_assignments[source] = assignment
        state.page_assignment = assignment
        page_bits = torch.zeros(table.num_pages, device=q.device, dtype=torch.int8)
        page_bits[(classes == int(PageClass.INT2_PAGE))] = 2
        page_bits[(classes == int(PageClass.INT4_PAGE))] = 4
        fp_class = (
            (classes == int(PageClass.FP_PAGE))
            | (classes == int(PageClass.PROTECTED_FP_PAGE))
            | (classes == int(PageClass.RECENT_FP_PAGE))
        )
        page_bits[fp_class] = 16
        bits = torch.repeat_interleave(page_bits, table.lengths)
        if self.collect_policy_records:
            self._unit_records.append({
                "unit_index": unit_index,
                "layer_id": layer_id,
                "input_survivor_count": int(table.token_indices.numel()),
                "output_survivor_count": int(table.token_indices.numel()),
                "drop_count": 0,
                "actual_cumulative_retention": 1.0,
                "v_payload_bit_ceiling": target.v_payload_bit_ceiling,
                "actual_average_v_bits": tier_assignment.average_v_bits,
                "depth_fraction": target.depth_fraction,
                "fp16_score_threshold": tier_assignment.fp16_threshold,
                "int4_score_threshold": tier_assignment.int4_threshold,
                "alignment_fallback_to_attention": alignment_fallback,
                "query_anchor_count": int(q.shape[2]),
                "query_anchor_reduction": config.query_anchor_reduction,
                "value_distortion_metric": config.value_distortion_metric,
                "precision_allocator": config.precision_allocator,
                "fp_pages": int((fp_class).sum()),
                "int4_pages": int((classes == int(PageClass.INT4_PAGE)).sum()),
                "int2_pages": int((classes == int(PageClass.INT2_PAGE)).sum()),
                "fp_tokens": tier_assignment.fp16_tokens,
                "int4_tokens": tier_assignment.int4_tokens,
                "int2_tokens": tier_assignment.int2_tokens,
                "video_fp_tokens": video_fp_tokens,
                "video_int4_tokens": video_int4_tokens,
                "video_int2_tokens": video_int2_tokens,
                "video_average_value_bits": average_modality_bits(
                    video_fp_tokens, video_int4_tokens, video_int2_tokens
                ),
                "audio_fp_tokens": audio_fp_tokens,
                "audio_int4_tokens": audio_int4_tokens,
                "audio_int2_tokens": audio_int2_tokens,
                "audio_average_value_bits": average_modality_bits(
                    audio_fp_tokens, audio_int4_tokens, audio_int2_tokens
                ),
                "evicted_pages": 0,
                **residual_stats,
            })
        return bits

    def _page_plan(
        self,
        layer_id: int,
        bits: torch.Tensor,
        modality: torch.Tensor,
        *,
        assignment_source_layer: int,
        score_at_this_layer: bool,
    ):
        config = self.assignment_config
        unprotected = torch.zeros_like(modality, dtype=torch.bool)
        table = self._page_table
        if table is None or int(table.token_indices.numel()) != int(modality.numel()):
            table = build_page_table(
                modality,
                page_size=config.page_size,
                logical_positions=torch.arange(modality.numel(), device=modality.device),
                global_token_indices=torch.arange(modality.numel(), device=modality.device),
                protected_mask=unprotected,
                recent_mask=unprotected,
            )
            self._page_table = table
        state = self._states[layer_id]
        state_assignment = getattr(state, "page_assignment", None)
        if state_assignment is not None:
            assignment = state_assignment
        elif not score_at_this_layer and assignment_source_layer in self._group_page_assignments:
            assignment = self._group_page_assignments[assignment_source_layer]
        else:
            assignment = project_token_bits_to_pages(table, bits)
        if score_at_this_layer:
            self._group_page_assignments[assignment_source_layer] = assignment
        initial_recent = 0
        # ``append_fp_only`` is the conservative generated-cache policy: every
        # token in the requested decode horizon must remain in the Recent-FP
        # pool.  The old unconditional 128-token cap silently made this mode
        # behave like chunked recompression for longer generations.
        recent_headroom = (
            config.fp_append_capacity
            if config.generated_cache_policy == "append_fp_only"
            else min(
                config.fp_append_capacity,
                max(config.recent_fp_window, config.recompress_chunk_size),
            )
        )
        minimum_recent_capacity = (
            config.fp_append_capacity
            if config.generated_cache_policy == "append_fp_only"
            else config.recompress_chunk_size
        )
        recent_capacity = max(
            minimum_recent_capacity,
            ((initial_recent + recent_headroom + config.page_size - 1) // config.page_size)
            * config.page_size,
        )
        if recent_capacity <= initial_recent:
            recent_capacity += config.page_size
        plan = build_layer_compression_plan(
            layer_id, assignment, recent_fp_capacity=recent_capacity
        )
        if plan.evicted_pages.numel():
            raise RuntimeError("score-tier V policy forbids eviction")
        return plan

    def _prepare_executor(self, state: Any) -> None:
        # The correctness oracle directly dequantizes ``fourq_cache`` and does
        # not use a Triton executor/workspace.  Skipping executor construction
        # also permits the matched-BF16 diagnostic cache, whose named INT4/INT2
        # pools intentionally hold FP payloads.
        if self.kernel_backend == "triton_mixed_reference":
            state.fourq_workspace = None
            state.fourq_executor = None
            return
        current_workspace = getattr(state, "fourq_workspace", None)
        # Compact K4 is reduction-bound beyond roughly 6K retained tokens.
        # A 64-token split improves occupancy there; shorter contexts retain
        # the lower-overhead 128-token tile.  Keep the established K2 schedule
        # unchanged.
        block_n = self.splitk_block_n or (
            64
            if state.fourq_cache.compact_int4_key
            and state.fourq_cache.survivor_count > 6144
            else 128
        )

        def launch_config(
            cache: ThreePrecisionCache, selected_block_n: int
        ) -> dict[str, int]:
            survivors = cache.survivor_count
            return {
                "num_warps": self.splitk_num_warps or 8,
                # Long cache tiles are register-heavy; one pipeline stage
                # consistently improves the uncaptured decode fastpath.
                "num_stages": self.splitk_num_stages or (
                    2 if survivors <= 1024 else 1
                ),
                "reduce_num_warps": self.splitk_reduce_num_warps or (
                    8 if selected_block_n == 64 or survivors <= 1024 else 4
                ),
                "broadcast_k_metadata": cache.compact_key is not None,
                "use_tensor_cores_qk": self.use_tensor_cores_qk,
                "use_tensor_cores_pv": self.use_tensor_cores_pv,
                "output_hadamard": self.value_hadamard,
                "hierarchical_reduce": self.splitk_hierarchical_reduce,
            }

        recent = state.fourq_cache.recent_fp_pool
        recent_free = max(0, recent.capacity - recent.token_count)
        lowbit_growth = max(0, self.assignment_config.fp_append_capacity - recent_free)
        required = required_splitk_blocks(
            state.fourq_cache, block_n=block_n, reserve_tokens=lowbit_growth
        )
        partial_dtype = torch.float16 if self.partial_output_dtype == "float16" else torch.float32
        if self.share_workspace:
            device = recent.fp_k.device
            workspace_key = (device, block_n, state.num_heads)
            workspace = self._shared_workspaces.get(workspace_key)
            if workspace is None or workspace.total_blocks < required:
                old_workspace = workspace
                workspace = allocate_splitk_workspace(
                    state.fourq_cache, block_n=block_n, reserve_tokens=lowbit_growth,
                    num_query_heads=state.num_heads,
                    partial_output_dtype=partial_dtype,
                    hierarchical_group_blocks=(
                        self.splitk_hierarchical_group_blocks
                    ),
                )
                self._shared_workspaces[workspace_key] = workspace
                if old_workspace is not None:
                    for candidate in self._states.values():
                        if getattr(candidate, "fourq_workspace", None) is old_workspace:
                            candidate.fourq_workspace = workspace
                            candidate.fourq_executor = prepare_splitk_executor(
                                candidate.fourq_cache,
                                workspace,
                                block_n=block_n,
                                **launch_config(candidate.fourq_cache, block_n),
                            )
            state.fourq_workspace = workspace
        elif (
            current_workspace is None
            or current_workspace.block_n != block_n
            or current_workspace.total_blocks < required
        ):
            state.fourq_workspace = allocate_splitk_workspace(
                state.fourq_cache, block_n=block_n, reserve_tokens=lowbit_growth,
                num_query_heads=state.num_heads,
                partial_output_dtype=partial_dtype,
                hierarchical_group_blocks=self.splitk_hierarchical_group_blocks,
            )
        block_n = state.fourq_workspace.block_n
        state.fourq_executor = prepare_splitk_executor(
            state.fourq_cache,
            state.fourq_workspace,
            block_n=block_n,
            **launch_config(state.fourq_cache, block_n),
        )

    def _project_qkv(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        position_embeddings: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(hidden_states.shape[0])
        if int(hidden_states.shape[1]) != 1:
            raise ValueError("compact fused mRoPE decode requires one token per request")
        if batch_size != self._throughput_batch_size:
            raise ValueError(
                "decode batch does not match the active throughput cache arena: "
                f"hidden={batch_size}, cache={self._throughput_batch_size}"
            )
        if position_embeddings is None or len(position_embeddings) != 2:
            raise ValueError("Qwen position_embeddings are required")
        q = module.q_proj(hidden_states).view(
            batch_size, 1, -1, module.head_dim
        ).transpose(1, 2).contiguous()
        k = module.k_proj(hidden_states).view(
            batch_size, 1, -1, module.head_dim
        ).transpose(1, 2).contiguous()
        v = module.v_proj(hidden_states).view(
            batch_size, 1, -1, module.head_dim
        ).transpose(1, 2).contiguous()
        cos, sin = position_embeddings
        if type(module).__name__ == "Qwen3Attention":
            q = module.q_norm(q.transpose(1, 2)).transpose(1, 2)
            k = module.k_norm(k.transpose(1, 2)).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if self.key_hadamard:
                q = triton_fwht_128(q) if q.is_cuda else normalized_fwht(q).to(q.dtype)
                k = triton_fwht_128(k) if k.is_cuda else normalized_fwht(k).to(k.dtype)
            return q.contiguous(), k.contiguous(), v
        rope_scaling = getattr(
            module, "rope_scaling", getattr(module.config, "rope_scaling", None)
        )
        q, k = qwen_qk_mrope(
            q,
            k,
            cos,
            sin,
            rope_scaling["mrope_section"],
            output=self._qk_rope_output(q, k),
            apply_hadamard=self.key_hadamard,
        )
        return q, k, v

    def _project_prefill_query_anchors(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        position_embeddings: Any,
    ) -> torch.Tensor:
        batch_size = int(hidden_states.shape[0])
        if batch_size > 1 and (
            not self._native_batched_prefill
            or batch_size != self._native_prefill_batch_size
        ):
            raise ValueError(
                "batched prefill requires enable_native_prefill_batch(B)"
            )
        if position_embeddings is None or len(position_embeddings) != 2:
            raise ValueError("Qwen position_embeddings are required")
        anchor_count = min(
            int(hidden_states.shape[1]), self.assignment_config.query_anchor_count
        )
        anchor_hidden = hidden_states[:, -anchor_count:]
        q = module.q_proj(anchor_hidden).view(
            batch_size, anchor_count, -1, module.head_dim
        ).transpose(1, 2)
        # The HF helper rotates q and k together. Projecting only one K token is
        # still much cheaper than re-projecting the complete prefill sequence.
        k = module.k_proj(anchor_hidden).view(
            batch_size, anchor_count, -1, module.head_dim
        ).transpose(1, 2)
        cos, sin = position_embeddings
        cos = cos[..., -anchor_count:, :]
        sin = sin[..., -anchor_count:, :]
        if type(module).__name__ == "Qwen3Attention":
            q = module.q_norm(q.transpose(1, 2)).transpose(1, 2)
            k = module.k_norm(k.transpose(1, 2)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, k, cos, sin)
            if self.key_hadamard:
                q = triton_fwht_128(q) if q.is_cuda else normalized_fwht(q).to(q.dtype)
            return q.contiguous()
        rope_scaling = getattr(module, "rope_scaling", getattr(module.config, "rope_scaling", None))
        q, _ = qwen_qk_mrope(
            q.contiguous(),
            k.contiguous(),
            cos,
            sin,
            rope_scaling["mrope_section"],
            output=self._qk_rope_output(q, k),
            apply_hadamard=self.key_hadamard,
        )
        return q.contiguous()

    def _finalize_native_batched_prefill_layer(
        self,
        plan: _RuntimeLayerPlan,
        module: Any,
        hidden_states: torch.Tensor,
        past: Any,
        position_embeddings: Any,
        attention_mask: torch.Tensor | None,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
    ) -> None:
        """Pack independent prefill K/V into the batched decode ABI.

        Precision geometry is computed once from request 0 so every request
        belongs to one homogeneous kernel bucket.  Each request is nevertheless
        packed from its own K/V tensors; no request reuses or clones request 0's
        payload.  This is the fixed-length systems path used by the OScaR-style
        benchmark, while B=1 continues through the established implementation.
        """
        layer_id = plan.layer_id
        state = self._states[layer_id]
        batch_size = int(full_k.shape[0])
        if (
            not self._native_batched_prefill
            or batch_size != self._native_prefill_batch_size
            or batch_size != int(hidden_states.shape[0])
        ):
            raise ValueError(
                "prefill batch does not match the enabled native batch"
            )
        if full_k.shape != full_v.shape or full_k.ndim != 4:
            raise ValueError("native batched prefill K/V must be matching [B,H,S,D]")
        seq_len = int(full_k.shape[2])
        if not self._mask_validated:
            self._validate_mask(attention_mask, seq_len)
            self._mask_validated = True

        q = self._project_prefill_query_anchors(
            module, hidden_states, position_embeddings
        )
        modality = self._modality(seq_len, q.device)
        # The allocator remains the validated B=1 oracle.  Its selected tier
        # counts define a homogeneous batch geometry; independent request K/V
        # payloads are then quantized using that shared geometry.
        first_key = self._kernel_key(full_k[0])
        first_value = self._kernel_value(full_v[0])
        source = plan.assignment_source_layer
        if plan.score_at_this_layer:
            bits = self._score_assignment(
                layer_id,
                q[0:1],
                first_key.unsqueeze(0),
                modality,
                v=first_value,
            )
            self._group_assignments[source] = bits
        else:
            if source not in self._group_assignments:
                raise RuntimeError(
                    f"block assignment from layer {source} is unavailable"
                )
            bits = self._group_assignments[source]
            if int(bits.numel()) != seq_len:
                raise RuntimeError("shared block assignment length mismatch")
        page_plan = self._page_plan(
            layer_id,
            bits,
            modality,
            assignment_source_layer=source,
            score_at_this_layer=plan.score_at_this_layer,
        )

        state.num_heads = int(q.shape[1])
        state.head_dim = int(q.shape[-1])
        request_caches: list[ThreePrecisionCache] = []
        request_executors: list[Any] = []
        for request in range(batch_size):
            cache_key = (
                first_key if request == 0 else self._kernel_key(full_k[request])
            )
            cache_value = (
                first_value
                if request == 0
                else self._kernel_value(full_v[request])
            )
            cache = self._measure(
                state,
                "packing_ms",
                q.device,
                lambda cache_key=cache_key, cache_value=cache_value: (
                    build_three_precision_cache(
                        cache_key,
                        cache_value,
                        page_plan,
                        page_size=self.assignment_config.page_size,
                        matched_fp=self._matched_fp_for_layer(layer_id),
                        cache_variant=self.cache_variant,
                        key_quantization_superblock=(
                            self.key_quantization_superblock
                        ),
                        int2_group_size=(
                            self.assignment_config.value_int2_group_size
                        ),
                        generated_value_precision=(
                            self.assignment_config.generated_value_precision
                        ),
                        compact_key_append_capacity=max(
                            0, self.assignment_config.fp_append_capacity
                        ),
                        recent_flush_size=(
                            128
                            if self.assignment_config.generated_cache_policy
                            == "append_fp_only"
                            else self.assignment_config.recompress_chunk_size
                        ),
                    )
                ),
            )
            request_caches.append(cache)
            state.fourq_cache = cache
            state.fourq_workspace = None
            self._prepare_executor(state)
            if state.fourq_executor is None:
                raise RuntimeError(
                    "native batched prefill requires the fused split-K backend"
                )
            request_executors.append(state.fourq_executor)

        first_executor = request_executors[0]
        workspace_key = (
            first_executor.workspace.output.device,
            first_executor.workspace.block_n,
            first_executor.workspace.total_blocks,
            first_executor.workspace.hierarchical_group_blocks,
            first_executor.workspace.hierarchical_group_capacity,
            first_executor.num_query_heads,
            first_executor.head_dim,
            batch_size,
        )
        batched = stack_homogeneous_splitk_executors(
            request_executors,
            workspace=self._native_batched_workspaces.get(workspace_key),
            throughput_use_tensor_cores_pv=(
                self.throughput_use_tensor_cores_pv
            ),
            throughput_direct_reduce_max_splits=(
                self.throughput_direct_reduce_max_splits
            ),
        )
        self._native_batched_workspaces[workspace_key] = batched.workspace
        state.fourq_cache = request_caches[0]
        state.fourq_workspace = batched.workspace
        state.fourq_executor = batched
        state.throughput_batch_size = batch_size
        state.dynamic_stats = DynamicCacheStats()
        state.bits = bits
        state.page_plan = page_plan
        state.original_seq_len = seq_len
        state.kv_object_id = id(batched)
        state.original_full_bytes = _tensor_bytes(full_k) + _tensor_bytes(full_v)
        state.last_cache_position = seq_len - 1
        self._release_dynamic_layer_cache(
            past,
            layer_id,
            logical_length=seq_len,
            device=q.device,
            dtype=full_k.dtype,
        )
        if self.prefill_executor_warmup:
            self._measure(
                state,
                "prefill_warmup_ms",
                q.device,
                lambda: state.fourq_executor(q),
            )
            state.prefill_warmup_count += 1
        state.migrated = True
        state.migration_count += 1
        state.prefill_finalize_count += 1

    def _finalize_prefill_layer(
        self,
        plan: _RuntimeLayerPlan,
        module: Any,
        hidden_states: torch.Tensor,
        past: Any,
        position_embeddings: Any,
        attention_mask: torch.Tensor | None,
    ) -> None:
        layer_id = plan.layer_id
        state = self._states[layer_id]
        if state.migrated:
            raise RuntimeError(f"layer {layer_id} prefill was finalized more than once")
        if past is None:
            raise TypeError("four-quadrant prefill finalization requires DynamicCache")
        full_k, full_v = self._cache_layer_tensors(past, layer_id)
        if not torch.is_tensor(full_k) or not full_k.numel():
            raise RuntimeError(f"layer {layer_id} has no prefill KV to finalize")
        if int(full_k.shape[0]) > 1:
            self._finalize_native_batched_prefill_layer(
                plan,
                module,
                hidden_states,
                past,
                position_embeddings,
                attention_mask,
                full_k,
                full_v,
            )
            return
        seq_len = int(full_k.shape[2])
        if int(full_v.shape[2]) != seq_len:
            raise ValueError("prefill K/V sequence lengths differ")
        if not self._mask_validated:
            self._validate_mask(attention_mask, seq_len)
            self._mask_validated = True
        q = self._project_prefill_query_anchors(
            module, hidden_states, position_embeddings
        )
        cache_key = self._kernel_key(full_k[0])
        cache_value = self._kernel_value(full_v[0])
        modality = self._modality(seq_len, q.device)
        source = plan.assignment_source_layer
        if plan.score_at_this_layer:
            bits = self._score_assignment(
                layer_id, q, cache_key.unsqueeze(0), modality, v=cache_value
            )
            self._group_assignments[source] = bits
        else:
            if source not in self._group_assignments:
                raise RuntimeError(f"block assignment from layer {source} is unavailable")
            bits = self._group_assignments[source]
            if int(bits.numel()) != seq_len:
                raise RuntimeError("shared block assignment length mismatch")
        page_plan = self._page_plan(
            layer_id,
            bits,
            modality,
            assignment_source_layer=source,
            score_at_this_layer=plan.score_at_this_layer,
        )
        state.fourq_cache = self._measure(
            state,
            "packing_ms",
            q.device,
            lambda: build_three_precision_cache(
                cache_key,
                cache_value,
                page_plan,
                page_size=self.assignment_config.page_size,
                matched_fp=self._matched_fp_for_layer(layer_id),
                cache_variant=self.cache_variant,
                key_quantization_superblock=self.key_quantization_superblock,
                int2_group_size=self.assignment_config.value_int2_group_size,
                generated_value_precision=(
                    self.assignment_config.generated_value_precision
                ),
                # Reserve against the full requested decode horizon. The
                # Recent-FP ring flushes complete 128-token blocks; subtracting
                # its capacity can leave the compact K/V arenas one page short
                # when the initial low-bit pool is not superblock aligned.
                compact_key_append_capacity=max(
                    0, self.assignment_config.fp_append_capacity
                ),
                recent_flush_size=(
                    128
                    if self.assignment_config.generated_cache_policy == "append_fp_only"
                    else self.assignment_config.recompress_chunk_size
                ),
            ),
        )
        state.num_heads = int(q.shape[1])
        state.head_dim = int(q.shape[-1])
        state.dynamic_stats = DynamicCacheStats()
        self._prepare_executor(state)
        state.bits = bits
        state.page_plan = page_plan
        state.original_seq_len = seq_len
        state.kv_object_id = id(state.fourq_cache)
        state.original_full_bytes = _tensor_bytes(full_k) + _tensor_bytes(full_v)
        state.last_cache_position = seq_len - 1
        self._release_dynamic_layer_cache(
            past,
            layer_id,
            logical_length=seq_len,
            device=q.device,
            dtype=full_k.dtype,
        )
        if (
            self.kernel_backend == "triton_mixed_fused_splitk"
            and self.prefill_executor_warmup
        ):
            self._measure(
                state,
                "prefill_warmup_ms",
                q.device,
                lambda: state.fourq_executor(q),
            )
            state.prefill_warmup_count += 1
        state.migrated = True
        state.migration_count += 1
        state.prefill_finalize_count += 1

    def _dispatch(self, plan: _RuntimeLayerPlan, module: Any, *args: Any, **kwargs: Any) -> Any:
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        if (
            not torch.is_tensor(hidden_states)
            or int(hidden_states.shape[1]) == 1
            or self._prefill_modality is None
        ):
            return super()._dispatch(plan, module, *args, **kwargs)

        # Preserve the exact native prefill output. Compression is a post-layer
        # cache finalization step and is therefore complete before decode begins.
        result = self._call_original(plan.layer_id, *args, **kwargs)
        state = self._states[plan.layer_id]
        past = kwargs.get("past_key_values", kwargs.get("past_key_value"))
        self._last_cache = past
        self._measure(
            state,
            "prefill_finalize_ms",
            hidden_states.device,
            lambda: self._finalize_prefill_layer(
                plan,
                module,
                hidden_states,
                past,
                kwargs.get("position_embeddings"),
                kwargs.get("attention_mask"),
            ),
        )
        return result

    def _run_fourq(
        self,
        state: Any,
        q: torch.Tensor,
        append: tuple[torch.Tensor, torch.Tensor, int, int] | None = None,
    ) -> torch.Tensor:
        state.kernel_calls += 1
        if self.kernel_backend == "triton_mixed_reference":
            return self._measure(
                state, "kernel_ms", q.device,
                lambda: triton_mixed_reference(q, state.fourq_cache),
            )
        return self._measure(
            state, "kernel_ms", q.device, lambda: state.fourq_executor(q, append=append)
        )

    def _append_generated_token(
        self,
        state: Any,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        current_position: int,
    ) -> bool:
        recent = state.fourq_cache.recent_fp_pool
        if (
            self.kernel_backend == "triton_mixed_fused_splitk"
            and recent.active_length is not None
            and recent.active_length < recent.capacity
        ):
            append_recent_fp_triton(
                state.fourq_cache,
                new_k,
                new_v,
                global_token_index=current_position,
                logical_position=current_position,
            )
            state.dynamic_stats.appended_tokens += 1
            return False
        return append_generated_token(
            state.fourq_cache,
            new_k,
            new_v,
            global_token_index=current_position,
            logical_position=current_position,
            stats=state.dynamic_stats,
        )

    def _active_forward(self, plan: _RuntimeLayerPlan, module: Any, *args: Any, **kwargs: Any) -> Any:
        layer_id = plan.layer_id
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        attention_mask = kwargs.get("attention_mask")
        uses_plural_cache_api = "past_key_values" in kwargs
        past = kwargs.get("past_key_values", kwargs.get("past_key_value"))
        position_embeddings = kwargs.get("position_embeddings")
        if bool(kwargs.get("output_attentions", False)):
            raise ValueError("four-quadrant active attention does not return attention weights")
        if past is None:
            raise TypeError("four-quadrant active decode requires DynamicCache")
        self._last_cache = past
        state = self._states[layer_id]
        state.calls += 1
        q, new_k, new_v = self._project_qkv(module, hidden_states, position_embeddings)

        if not state.migrated:
            prefix_k, prefix_v = self._cache_layer_tensors(past, layer_id)
            if not torch.is_tensor(prefix_k) or not prefix_k.numel():
                raise RuntimeError(f"layer {layer_id} has no prefill KV to migrate")
            full_k = torch.cat((prefix_k, new_k), dim=2).contiguous()
            full_v = torch.cat((prefix_v, new_v), dim=2).contiguous()
            seq_len = int(full_k.shape[2])
            if not self._mask_validated:
                self._validate_mask(attention_mask, seq_len)
                self._mask_validated = True
            modality = self._modality(seq_len, q.device)
            source = plan.assignment_source_layer
            cache_key = (
                torch.cat((self._kernel_key(prefix_k[0]), new_k[0]), dim=1)
                if self.key_hadamard
                else full_k[0]
            )
            cache_value = self._kernel_value(full_v[0])
            score_key = cache_key.unsqueeze(0)
            if plan.score_at_this_layer:
                bits = self._score_assignment(
                    layer_id, q, score_key, modality, v=cache_value
                )
                self._group_assignments[source] = bits
            else:
                if source not in self._group_assignments:
                    raise RuntimeError(f"block assignment from layer {source} is unavailable")
                bits = self._group_assignments[source]
                if int(bits.numel()) != seq_len:
                    raise RuntimeError("shared block assignment length mismatch")
            page_plan = self._page_plan(
                layer_id, bits, modality,
                assignment_source_layer=source,
                score_at_this_layer=plan.score_at_this_layer,
            )
            state.fourq_cache = self._measure(
                state, "packing_ms", q.device,
                lambda: build_three_precision_cache(
                    cache_key, cache_value, page_plan,
                    page_size=self.assignment_config.page_size,
                    matched_fp=self._matched_fp_for_layer(layer_id),
                    cache_variant=self.cache_variant,
                    key_quantization_superblock=self.key_quantization_superblock,
                    int2_group_size=self.assignment_config.value_int2_group_size,
                    generated_value_precision=(
                        self.assignment_config.generated_value_precision
                    ),
                    compact_key_append_capacity=max(
                        0, self.assignment_config.fp_append_capacity
                    ),
                    recent_flush_size=(
                        128
                        if self.assignment_config.generated_cache_policy == "append_fp_only"
                        else self.assignment_config.recompress_chunk_size
                    ),
                ),
            )
            state.num_heads = int(q.shape[1])
            state.head_dim = int(q.shape[-1])
            state.dynamic_stats = DynamicCacheStats()
            self._prepare_executor(state)
            state.bits = bits
            state.page_plan = page_plan
            state.original_seq_len = seq_len
            state.kv_object_id = id(state.fourq_cache)
            mixed = self._run_fourq(state, q)
            state.migrated = True
            state.migration_count += 1
            state.last_cache_position = self._decode_cache_position(
                layer_id,
                past,
                seq_len - 1,
                kwargs.get("cache_position"),
            )
            state.original_full_bytes = _tensor_bytes(full_k) + _tensor_bytes(full_v)
            self._release_dynamic_layer_cache(
                past,
                layer_id,
                logical_length=seq_len,
                device=q.device,
                dtype=new_k.dtype,
            )
        else:
            kernel_new_k = new_k
            kernel_new_v = self._kernel_value(new_v)
            current_position = self._decode_cache_position(
                layer_id,
                past,
                state.original_seq_len,
                kwargs.get("cache_position"),
            )
            fused_append = None
            if current_position == state.last_cache_position:
                state.duplicate_append_avoided += 1
            else:
                if current_position != state.original_seq_len:
                    raise RuntimeError(
                        f"cache position {current_position} is not append position {state.original_seq_len}"
                    )
                recent = state.fourq_cache.recent_fp_pool
                if (
                    self.kernel_backend == "triton_mixed_fused_splitk"
                    and recent.active_length is not None
                    and recent.active_length < recent.capacity
                ):
                    fused_append = (
                        kernel_new_k,
                        kernel_new_v,
                        current_position,
                        current_position,
                    )
                    flushed = False
                else:
                    flushed = self._measure(
                        state, "cache_append_ms", q.device,
                        lambda: self._append_generated_token(
                            state, kernel_new_k, kernel_new_v, current_position
                        ),
                    )
                if flushed:
                    self._prepare_executor(state)
                    state.generated_recompress_count += 1
                    state.flush_decode_steps.append(state.append_count + 1)
                state.append_count += 1
                state.original_seq_len += 1
                state.last_cache_position = current_position
                state.original_full_bytes += _tensor_bytes(new_k) + _tensor_bytes(new_v)
            mixed = self._run_fourq(state, q, append=fused_append)
            if fused_append is not None:
                state.fourq_cache.recent_fp_pool.active_length += 1
                state.dynamic_stats.appended_tokens += 1
            if layer_id == 0:
                self._set_layer0_length_proxy(
                    past, state.original_seq_len, q.device
                )
        if self.validate_kernel_output:
            torch._assert_async(torch.isfinite(mixed).all(), "four-quadrant attention produced NaN/Inf")
        if (
            self.value_hadamard
            and self.kernel_backend != "triton_mixed_fused_splitk"
        ):
            mixed = (
                triton_fwht_128(mixed)
                if mixed.is_cuda
                else normalized_fwht(mixed).to(mixed.dtype)
            )
        output = module.o_proj(
            mixed.reshape(int(hidden_states.shape[0]), 1, -1).to(hidden_states.dtype)
        )
        if uses_plural_cache_api or type(module).__name__ == "Qwen3Attention":
            return output, None
        return output, None, past

    def get_stats(self) -> dict[str, Any]:
        self._resolve_pending_timings()
        layer_records = []
        persistent_payload = 0
        persistent_metadata = 0
        active_payload = 0
        active_metadata = 0
        persistent_payload_allocations = 0
        persistent_metadata_allocations = 0
        working = 0
        fp_pool_bytes = 0
        int4_pool_bytes = 0
        int2_pool_bytes = 0
        recent_fp_pool_bytes = 0
        compact_key_bytes = 0
        compact_int4_key_bytes = 0
        compact_int4_value_bytes = 0
        workspace_ids: set[int] = set()
        for layer_id, state in sorted(self._states.items()):
            cache: ThreePrecisionCache | None = getattr(state, "fourq_cache", None)
            if cache is not None:
                storage = precision_cache_storage_breakdown(cache)
                active_storage = precision_cache_active_storage_breakdown(cache)
                cache_batch = int(getattr(state, "throughput_batch_size", 1))
                persistent_payload += cache_batch * storage["kv_payload_bytes"]
                persistent_metadata += cache_batch * storage["metadata_bytes"]
                active_payload += cache_batch * active_storage["kv_payload_bytes"]
                active_metadata += cache_batch * active_storage["metadata_bytes"]
                persistent_payload_allocations += (
                    cache_batch * storage["kv_payload_allocation_count"]
                )
                persistent_metadata_allocations += (
                    cache_batch * storage["metadata_allocation_count"]
                )
                workspace = getattr(state, "fourq_workspace", None)
                workspace_id = id(workspace)
                if workspace is not None and workspace_id not in workspace_ids:
                    working += allocated_workspace_bytes(workspace)
                    workspace_ids.add(workspace_id)
                fp_pool_bytes += cache_batch * cache.fp_pool.payload_bytes
                int4_pool_bytes += cache_batch * cache.int4_pool.payload_bytes
                int2_pool_bytes += cache_batch * cache.int2_pool.payload_bytes
                recent_fp_pool_bytes += (
                    cache_batch * cache.recent_fp_pool.payload_bytes
                )
                if cache.compact_key is not None:
                    compact_key_bytes += cache_batch * cache.compact_key.nbytes
                if cache.compact_int4_key:
                    compact_int4_key_bytes += cache_batch * cache.compact_key.nbytes
                if cache.compact_int4_value is not None:
                    compact_int4_value_bytes += (
                        cache_batch * cache.compact_int4_value.nbytes
                    )
            dynamic = getattr(state, "dynamic_stats", DynamicCacheStats())
            layer_records.append({
                "layer_id": layer_id,
                "mode": self.layer_strategy[layer_id]["mode"],
                "score_at_this_layer": bool(self.layer_strategy[layer_id]["score_at_this_layer"]),
                "assignment_source_layer": self.layer_strategy[layer_id]["assignment_source_layer"],
                "group_start_layer": self.layer_strategy[layer_id]["group_start_layer"],
                "group_end_layer": self.layer_strategy[layer_id]["group_end_layer"],
                "calls": state.calls,
                "kernel_calls": state.kernel_calls,
                "kernel_ms": state.kernel_ms,
                "packing_ms": state.packing_ms,
                "prefill_finalize_ms": state.prefill_finalize_ms,
                "prefill_warmup_ms": state.prefill_warmup_ms,
                "prefill_finalize_count": state.prefill_finalize_count,
                "prefill_warmup_count": state.prefill_warmup_count,
                "migration_count": state.migration_count,
                "cache_append_ms": state.cache_append_ms,
                "scoring_ms": state.scoring_ms,
                "alignment_ms": state.alignment_ms,
                "sensitivity_ms": state.sensitivity_ms,
                "quadrant_assignment_ms": state.quadrant_assignment_ms,
                "sorting_ms": state.sorting_ms,
                "request_count": int(getattr(state, "throughput_batch_size", 1)),
                "fp_tokens": 0 if cache is None else cache.fp_pool.token_count + cache.recent_fp_pool.token_count,
                "recent_fp_tokens": 0 if cache is None else cache.recent_fp_pool.token_count,
                "recent_flush_size": 0 if cache is None else cache.recent_flush_size,
                "generated_value_precision": (
                    "none" if cache is None else cache.generated_value_precision
                ),
                "static_int4_tokens": 0 if cache is None else cache.static_int4_tokens,
                "generated_int4_tokens": (
                    0
                    if cache is None
                    else max(0, cache.int4_pool.token_count - cache.static_int4_tokens)
                ),
                "int4_tokens": 0 if cache is None else cache.int4_pool.token_count,
                "int2_tokens": 0 if cache is None else cache.int2_pool.token_count,
                "evicted_tokens": 0 if cache is None else int(cache.evicted_global_indices.numel()),
                "actual_kv_payload_bytes": (
                    0
                    if cache is None
                    else int(getattr(state, "throughput_batch_size", 1))
                    * active_storage["kv_payload_bytes"]
                ),
                "cache_metadata_bytes": (
                    0
                    if cache is None
                    else int(getattr(state, "throughput_batch_size", 1))
                    * active_storage["metadata_bytes"]
                ),
                "kv_payload_allocation_count": (
                    0
                    if cache is None
                    else int(getattr(state, "throughput_batch_size", 1))
                    * storage["kv_payload_allocation_count"]
                ),
                "metadata_allocation_count": (
                    0
                    if cache is None
                    else int(getattr(state, "throughput_batch_size", 1))
                    * storage["metadata_allocation_count"]
                ),
                "actual_kv_bytes": (
                    0
                    if cache is None
                    else int(getattr(state, "throughput_batch_size", 1))
                    * active_storage["persistent_cache_bytes"]
                ),
                "allocated_kv_bytes": (
                    0
                    if cache is None
                    else int(getattr(state, "throughput_batch_size", 1))
                    * storage["persistent_cache_bytes"]
                ),
                "workspace_bytes": (
                    0
                    if cache is None or getattr(state, "fourq_workspace", None) is None
                    else allocated_workspace_bytes(state.fourq_workspace)
                ),
                "workspace_shared": bool(cache is not None and self.share_workspace),
                "partial_output_dtype": self.partial_output_dtype,
                "flush_count": dynamic.flushed_superblocks,
                "flushed_int4_value_tokens": dynamic.flushed_int4_value_tokens,
                "flush_decode_steps": list(state.flush_decode_steps),
                "full_dequant_tensor_count": dynamic.full_dequant_tensor_count,
                "fallback_count": state.fallback_count,
                "errors": list(state.errors),
            })
        full_bytes, layer0_proxy_bytes = self._dynamic_cache_payload_bytes()
        persistent_payload += full_bytes
        active_payload += full_bytes
        allocated_persistent = persistent_payload + persistent_metadata
        active_persistent = active_payload + active_metadata
        fp_tokens = sum(item["fp_tokens"] for item in layer_records)
        recent_fp_tokens = sum(item["recent_fp_tokens"] for item in layer_records)
        int4_tokens = sum(item["int4_tokens"] for item in layer_records)
        int2_tokens = sum(item["int2_tokens"] for item in layer_records)
        evicted_tokens = sum(item["evicted_tokens"] for item in layer_records)
        original_tokens = fp_tokens + int4_tokens + int2_tokens + evicted_tokens
        quantized_tokens = fp_tokens + int4_tokens + int2_tokens
        average_value_bits = (
            16.0 * fp_tokens + 4.0 * int4_tokens + 2.0 * int2_tokens
        ) / max(quantized_tokens, 1)
        average_key_bits = self._average_key_bits()
        average_kv_bits = (average_key_bits + average_value_bits) / 2.0
        kernel_calls = sum(item["kernel_calls"] for item in layer_records)
        kernel_ms = sum(item["kernel_ms"] for item in layer_records)
        groups = []
        for source in sorted({
            int(item["assignment_source_layer"])
            for item in layer_records
            if item["group_start_layer"] is not None
        }):
            members = [item for item in layer_records if item["assignment_source_layer"] == source]
            groups.append({
                "assignment_source_layer": source,
                "group_start_layer": min(int(item["layer_id"]) for item in members),
                "group_end_layer": max(int(item["layer_id"]) for item in members),
                "group_scoring_ms": sum(item["scoring_ms"] for item in members),
                "group_total_packing_ms": sum(item["packing_ms"] for item in members),
                "group_total_kernel_ms": sum(item["kernel_ms"] for item in members),
            })
        return {
            "backend": self.kernel_backend,
            "precision_mode": "sspa",
            "cache_variant": self.cache_variant,
            "layer0_compressed": bool(self.layer_strategy.get(0, {}).get("enabled")),
            "matched_fp_until_layer": self.matched_fp_until_layer,
            "splitk_num_warps": self.splitk_num_warps,
            "splitk_num_stages": self.splitk_num_stages,
            "splitk_reduce_num_warps": self.splitk_reduce_num_warps,
            "key_quantization_superblock": self.key_quantization_superblock,
            "generated_cache_policy": self.assignment_config.generated_cache_policy,
            "generated_value_precision": self.assignment_config.generated_value_precision,
            "recent_fp_window": self.assignment_config.recent_fp_window,
            "recompress_chunk_size": self.assignment_config.recompress_chunk_size,
            "query_anchor_count": self.assignment_config.query_anchor_count,
            "query_anchor_reduction": self.assignment_config.query_anchor_reduction,
            "query_anchor_ema_decay": self.assignment_config.query_anchor_ema_decay,
            "value_distortion_metric": self.assignment_config.value_distortion_metric,
            "precision_allocator": self.assignment_config.precision_allocator,
            "residual_allocation_rounds": self.assignment_config.residual_allocation_rounds,
            "k_conditioned_iterations": self.assignment_config.k_conditioned_iterations,
            "residual_query_objective": self.assignment_config.residual_query_objective,
            "residual_query_cvar_fraction": self.assignment_config.residual_query_cvar_fraction,
            "residual_transition_weight": self.assignment_config.residual_transition_weight,
            "output_priority_weight": self.assignment_config.output_priority_weight,
            "semantic_priority_weight": self.assignment_config.semantic_priority_weight,
            "precision_schedule": "sigmoid",
            "precision_sigmoid_steepness": self.assignment_config.precision_sigmoid_steepness,
            "shallow_v_payload_bit_ceiling": (
                self.assignment_config.shallow_v_payload_bit_ceiling
            ),
            "deep_v_payload_bit_ceiling": (
                self.assignment_config.deep_v_payload_bit_ceiling
            ),
            "value_int2_group_size": self.assignment_config.value_int2_group_size,
            "share_workspace": self.share_workspace,
            "partial_output_dtype": self.partial_output_dtype,
            "collect_timings": self.collect_timings,
            "validate_kernel_output": self.validate_kernel_output,
            "prefill_executor_warmup": self.prefill_executor_warmup,
            "reuse_workspace_across_requests": self.reuse_workspace_across_requests,
            "collect_policy_records": self.collect_policy_records,
            "validate_policy_state": self.validate_policy_state,
            "layers": layer_records,
            "groups": groups,
            "pruning_units": list(self._unit_records),
            "precision_units": list(self._unit_records),
            "actual_cache_storage_bytes": active_persistent,
            "cache_storage": {
                "persistent_cache_bytes": active_persistent,
                "kv_payload_bytes": active_payload,
                "metadata_bytes": active_metadata,
                "active_persistent_cache_bytes": active_persistent,
                "active_kv_payload_bytes": active_payload,
                "active_metadata_bytes": active_metadata,
                "allocated_persistent_cache_bytes": allocated_persistent,
                "allocated_kv_payload_bytes": persistent_payload,
                "allocated_metadata_bytes": persistent_metadata,
                "reserved_unused_bytes": max(
                    0, allocated_persistent - active_persistent
                ),
                "kv_payload_allocation_count": persistent_payload_allocations,
                "metadata_allocation_count": persistent_metadata_allocations,
                "unmodified_layer_bf16_bytes": full_bytes,
                "layer0_length_proxy_bytes": layer0_proxy_bytes,
                "fp_pool_bytes": fp_pool_bytes,
                "int4_pool_bytes": int4_pool_bytes,
                "int2_pool_bytes": int2_pool_bytes,
                "recent_fp_pool_bytes": recent_fp_pool_bytes,
                "compact_int4_key_bytes": compact_int4_key_bytes,
                "compact_key_bytes": compact_key_bytes,
                "compact_int4_value_bytes": compact_int4_value_bytes,
                "evicted_payload_bytes": 0,
                "partial_buffer_bytes": working,
                "total_cache_and_working_bytes": active_persistent + working,
                "total_allocated_cache_and_working_bytes": (
                    allocated_persistent + working
                ),
            },
            "scoring_latency_ms": sum(item["scoring_ms"] for item in layer_records),
            "alignment_latency_ms": sum(item["alignment_ms"] for item in layer_records),
            "sensitivity_latency_ms": sum(
                item["sensitivity_ms"] for item in layer_records
            ),
            "quadrant_assignment_latency_ms": sum(item["quadrant_assignment_ms"] for item in layer_records),
            "sorting_latency_ms": sum(item["sorting_ms"] for item in layer_records),
            "packing_latency_ms": sum(item["packing_ms"] for item in layer_records),
            "prefill_finalize_latency_ms": sum(item["prefill_finalize_ms"] for item in layer_records),
            "prefill_warmup_latency_ms": sum(item["prefill_warmup_ms"] for item in layer_records),
            "prefill_finalize_count": sum(item["prefill_finalize_count"] for item in layer_records),
            "prefill_warmup_count": sum(item["prefill_warmup_count"] for item in layer_records),
            "mixed_kernel_latency_ms": sum(item["kernel_ms"] for item in layer_records),
            "mixed_kernel_total_ms": kernel_ms,
            "mixed_kernel_avg_ms": kernel_ms / max(kernel_calls, 1),
            "cache_append_latency_ms": sum(item["cache_append_ms"] for item in layer_records),
            "generated_recompress_latency_ms": 0.0,
            "flush_decode_steps": sorted({
                step for item in layer_records for step in item["flush_decode_steps"]
            }),
            "mixed_kernel_calls": kernel_calls,
            "fallback_count": sum(item["fallback_count"] for item in layer_records),
            "full_dequant_tensor_count": sum(item["full_dequant_tensor_count"] for item in layer_records),
            "fp_tokens": fp_tokens,
            "recent_fp_tokens": recent_fp_tokens,
            "int4_tokens": int4_tokens,
            "int2_tokens": int2_tokens,
            "evicted_tokens": evicted_tokens,
            "average_value_bits": average_value_bits,
            "average_key_bits": average_key_bits,
            "average_kv_bits": average_kv_bits,
            "final_actual_retention": (fp_tokens + int4_tokens + int2_tokens) / max(original_tokens, 1),
            "compression_ratio": active_persistent / max(
                full_bytes + sum(state.original_full_bytes for state in self._states.values()), 1
            ),
        }

    def get_summary_stats(self) -> dict[str, Any]:
        """Return aggregate runtime and cache metrics."""
        self._resolve_pending_timings()
        persistent_payload = 0
        persistent_metadata = 0
        active_payload = 0
        active_metadata = 0
        payload_allocations = 0
        metadata_allocations = 0
        working = 0
        fp_pool_bytes = 0
        int4_pool_bytes = 0
        int2_pool_bytes = 0
        recent_fp_pool_bytes = 0
        compact_key_bytes = 0
        compact_int4_key_bytes = 0
        compact_int4_value_bytes = 0
        fp_tokens = 0
        recent_fp_tokens = 0
        int4_tokens = 0
        int2_tokens = 0
        evicted_tokens = 0
        original_active_bytes = 0
        workspace_ids: set[int] = set()

        states = list(self._states.values())
        for state in states:
            cache: ThreePrecisionCache | None = getattr(state, "fourq_cache", None)
            original_active_bytes += int(getattr(state, "original_full_bytes", 0))
            if cache is None:
                continue
            storage = precision_cache_storage_breakdown(cache)
            active_storage = precision_cache_active_storage_breakdown(cache)
            persistent_payload += storage["kv_payload_bytes"]
            persistent_metadata += storage["metadata_bytes"]
            active_payload += active_storage["kv_payload_bytes"]
            active_metadata += active_storage["metadata_bytes"]
            payload_allocations += storage["kv_payload_allocation_count"]
            metadata_allocations += storage["metadata_allocation_count"]
            workspace = getattr(state, "fourq_workspace", None)
            if workspace is not None and id(workspace) not in workspace_ids:
                working += allocated_workspace_bytes(workspace)
                workspace_ids.add(id(workspace))
            fp_pool_bytes += cache.fp_pool.payload_bytes
            int4_pool_bytes += cache.int4_pool.payload_bytes
            int2_pool_bytes += cache.int2_pool.payload_bytes
            recent_fp_pool_bytes += cache.recent_fp_pool.payload_bytes
            if cache.compact_key is not None:
                compact_key_bytes += cache.compact_key.nbytes
            if cache.compact_int4_key:
                compact_int4_key_bytes += cache.compact_key.nbytes
            if cache.compact_int4_value is not None:
                compact_int4_value_bytes += cache.compact_int4_value.nbytes
            fp_tokens += cache.fp_pool.token_count + cache.recent_fp_pool.token_count
            recent_fp_tokens += cache.recent_fp_pool.token_count
            int4_tokens += cache.int4_pool.token_count
            int2_tokens += cache.int2_pool.token_count
            evicted_tokens += int(cache.evicted_global_indices.numel())

        full_bytes, layer0_proxy_bytes = self._dynamic_cache_payload_bytes()
        persistent_payload += full_bytes
        active_payload += full_bytes
        allocated_persistent = persistent_payload + persistent_metadata
        active_persistent = active_payload + active_metadata
        kernel_calls = sum(int(state.kernel_calls) for state in states)
        kernel_ms = sum(float(state.kernel_ms) for state in states)
        original_tokens = fp_tokens + int4_tokens + int2_tokens + evicted_tokens
        quantized_tokens = fp_tokens + int4_tokens + int2_tokens
        average_value_bits = (
            16.0 * fp_tokens + 4.0 * int4_tokens + 2.0 * int2_tokens
        ) / max(quantized_tokens, 1)
        average_key_bits = self._average_key_bits()
        average_kv_bits = (average_key_bits + average_value_bits) / 2.0
        fallback_layers = sorted(
            layer_id
            for layer_id, state in self._states.items()
            if int(getattr(state, "fallback_count", 0)) > 0
        )
        splitk_executors = [
            getattr(state, "fourq_executor", None) for state in states
        ]
        splitk_executors = [
            executor for executor in splitk_executors if executor is not None
        ]
        active_splitk_block_ns = sorted({
            int(executor.block_n) for executor in splitk_executors
        })
        active_splitk_totals = [
            int(executor.total) for executor in splitk_executors
        ]
        active_fused_tensor_core_dots = bool(
            splitk_executors
            and all(executor.fuse_tensor_core_dots for executor in splitk_executors)
        )
        active_native_full_head_layout = bool(
            splitk_executors
            and all(
                executor.native_full_head_layout for executor in splitk_executors
            )
        )

        return {
            "backend": self.kernel_backend,
            "precision_mode": "sspa",
            "cache_variant": self.cache_variant,
            "layer0_compressed": bool(self.layer_strategy.get(0, {}).get("enabled")),
            "matched_fp_until_layer": self.matched_fp_until_layer,
            "splitk_num_warps": self.splitk_num_warps,
            "splitk_num_stages": self.splitk_num_stages,
            "splitk_reduce_num_warps": self.splitk_reduce_num_warps,
            "active_splitk_block_ns": active_splitk_block_ns,
            "active_fused_tensor_core_dots": active_fused_tensor_core_dots,
            "active_native_full_head_layout": active_native_full_head_layout,
            "active_splitk_total_range": (
                [min(active_splitk_totals), max(active_splitk_totals)]
                if active_splitk_totals
                else []
            ),
            "key_quantization_superblock": self.key_quantization_superblock,
            "generated_cache_policy": self.assignment_config.generated_cache_policy,
            "generated_value_precision": self.assignment_config.generated_value_precision,
            "recent_fp_window": self.assignment_config.recent_fp_window,
            "recompress_chunk_size": self.assignment_config.recompress_chunk_size,
            "query_anchor_count": self.assignment_config.query_anchor_count,
            "query_anchor_reduction": self.assignment_config.query_anchor_reduction,
            "query_anchor_ema_decay": self.assignment_config.query_anchor_ema_decay,
            "value_distortion_metric": self.assignment_config.value_distortion_metric,
            "precision_allocator": self.assignment_config.precision_allocator,
            "residual_allocation_rounds": self.assignment_config.residual_allocation_rounds,
            "k_conditioned_iterations": self.assignment_config.k_conditioned_iterations,
            "residual_query_objective": self.assignment_config.residual_query_objective,
            "residual_query_cvar_fraction": self.assignment_config.residual_query_cvar_fraction,
            "residual_transition_weight": self.assignment_config.residual_transition_weight,
            "output_priority_weight": self.assignment_config.output_priority_weight,
            "semantic_priority_weight": self.assignment_config.semantic_priority_weight,
            "precision_schedule": "sigmoid",
            "precision_sigmoid_steepness": self.assignment_config.precision_sigmoid_steepness,
            "shallow_v_payload_bit_ceiling": (
                self.assignment_config.shallow_v_payload_bit_ceiling
            ),
            "deep_v_payload_bit_ceiling": (
                self.assignment_config.deep_v_payload_bit_ceiling
            ),
            "value_int2_group_size": self.assignment_config.value_int2_group_size,
            "share_workspace": self.share_workspace,
            "partial_output_dtype": self.partial_output_dtype,
            "collect_timings": self.collect_timings,
            "validate_kernel_output": self.validate_kernel_output,
            "prefill_executor_warmup": self.prefill_executor_warmup,
            "reuse_workspace_across_requests": self.reuse_workspace_across_requests,
            "collect_policy_records": self.collect_policy_records,
            "validate_policy_state": self.validate_policy_state,
            "stats_detail": "summary",
            "layers": [],
            "groups": [],
            "actual_cache_storage_bytes": active_persistent,
            "cache_storage": {
                "persistent_cache_bytes": active_persistent,
                "kv_payload_bytes": active_payload,
                "metadata_bytes": active_metadata,
                "active_persistent_cache_bytes": active_persistent,
                "active_kv_payload_bytes": active_payload,
                "active_metadata_bytes": active_metadata,
                "allocated_persistent_cache_bytes": allocated_persistent,
                "allocated_kv_payload_bytes": persistent_payload,
                "allocated_metadata_bytes": persistent_metadata,
                "reserved_unused_bytes": max(
                    0, allocated_persistent - active_persistent
                ),
                "kv_payload_allocation_count": payload_allocations,
                "metadata_allocation_count": metadata_allocations,
                "unmodified_layer_bf16_bytes": full_bytes,
                "layer0_length_proxy_bytes": layer0_proxy_bytes,
                "fp_pool_bytes": fp_pool_bytes,
                "int4_pool_bytes": int4_pool_bytes,
                "int2_pool_bytes": int2_pool_bytes,
                "recent_fp_pool_bytes": recent_fp_pool_bytes,
                "compact_int4_key_bytes": compact_int4_key_bytes,
                "compact_key_bytes": compact_key_bytes,
                "compact_int4_value_bytes": compact_int4_value_bytes,
                "evicted_payload_bytes": 0,
                "partial_buffer_bytes": working,
                "total_cache_and_working_bytes": active_persistent + working,
                "total_allocated_cache_and_working_bytes": (
                    allocated_persistent + working
                ),
            },
            "scoring_latency_ms": sum(float(state.scoring_ms) for state in states),
            "alignment_latency_ms": sum(float(state.alignment_ms) for state in states),
            "sensitivity_latency_ms": sum(float(state.sensitivity_ms) for state in states),
            "quadrant_assignment_latency_ms": sum(
                float(state.quadrant_assignment_ms) for state in states
            ),
            "sorting_latency_ms": sum(float(state.sorting_ms) for state in states),
            "packing_latency_ms": sum(float(state.packing_ms) for state in states),
            "prefill_finalize_latency_ms": sum(
                float(state.prefill_finalize_ms) for state in states
            ),
            "prefill_warmup_latency_ms": sum(
                float(state.prefill_warmup_ms) for state in states
            ),
            "prefill_finalize_count": sum(
                int(state.prefill_finalize_count) for state in states
            ),
            "prefill_warmup_count": sum(
                int(state.prefill_warmup_count) for state in states
            ),
            "mixed_kernel_latency_ms": kernel_ms,
            "mixed_kernel_total_ms": kernel_ms,
            "mixed_kernel_avg_ms": kernel_ms / max(kernel_calls, 1),
            "cache_append_latency_ms": sum(
                float(state.cache_append_ms) for state in states
            ),
            "fallback_latency_ms": sum(
                float(state.fallback_latency_ms) for state in states
            ),
            "mixed_kernel_calls": kernel_calls,
            "fallback_count": sum(int(state.fallback_count) for state in states),
            "fallback_layers": fallback_layers,
            "full_dequant_tensor_count": sum(
                int(getattr(state, "dynamic_stats", DynamicCacheStats()).full_dequant_tensor_count)
                for state in states
            ),
            "flushed_int4_value_tokens": sum(
                int(
                    getattr(
                        state, "dynamic_stats", DynamicCacheStats()
                    ).flushed_int4_value_tokens
                )
                for state in states
            ),
            "fp_tokens": fp_tokens,
            "recent_fp_tokens": recent_fp_tokens,
            "int4_tokens": int4_tokens,
            "int2_tokens": int2_tokens,
            "evicted_tokens": evicted_tokens,
            "average_value_bits": average_value_bits,
            "average_key_bits": average_key_bits,
            "average_kv_bits": average_kv_bits,
            "final_actual_retention": (
                fp_tokens + int4_tokens + int2_tokens
            ) / max(original_tokens, 1),
            "compression_ratio": active_persistent
            / max(full_bytes + original_active_bytes, 1),
        }

    def get_quality_stats(self) -> dict[str, Any]:
        """Return correctness diagnostics without timing or storage traversal."""
        states = list(self._states.values())
        fallback_layers = sorted(
            layer_id
            for layer_id, state in self._states.items()
            if int(getattr(state, "fallback_count", 0)) > 0
        )
        return {
            "fallback_count": sum(
                int(getattr(state, "fallback_count", 0)) for state in states
            ),
            "fallback_layers": fallback_layers,
            "full_dequant_tensor_count": sum(
                int(
                    getattr(
                        state, "dynamic_stats", DynamicCacheStats()
                    ).full_dequant_tensor_count
                )
                for state in states
            ),
        }
