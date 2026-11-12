"""Pure SSPA cache-compression policies without CUDA/runtime ownership."""

from .layer_strategy import LayerStrategyConfig, build_layer_strategy

__all__ = [
    "LayerStrategyConfig",
    "build_layer_strategy",
]
