from enum import Flag, auto


class LoopIndicator(Flag):
    """LoopIndicator is a flag class that contains loog begin and end flags."""

    NONE = 0
    BEGIN = auto()
    END = auto()
