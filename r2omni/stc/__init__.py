"""Stage I: Structure-Aware Token Consolidation (STC)."""

from .qwen_stc import STCState, install_stc
from .pre_llm_units import apply_stc, load_config

__all__ = ["STCState", "apply_stc", "install_stc", "load_config"]
