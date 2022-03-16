"""Message class."""

from enum import Enum

class MessageType(Enum):
    """Define Message types."""

    WEIGHTS = 1  # model weights
    EOT = 2  # end of training