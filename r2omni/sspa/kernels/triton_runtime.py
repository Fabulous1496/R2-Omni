from __future__ import annotations

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None and tl is not None
