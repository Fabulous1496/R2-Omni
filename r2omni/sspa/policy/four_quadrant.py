from __future__ import annotations

from enum import IntEnum


class PageClass(IntEnum):
    EVICTED_PAGE = 0
    FP_PAGE = 1
    INT4_PAGE = 2
    RECENT_FP_PAGE = 3
    PROTECTED_FP_PAGE = 4
    INT2_PAGE = 5
