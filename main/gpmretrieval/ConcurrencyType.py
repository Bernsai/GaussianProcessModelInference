from enum import Enum


class ConcurrencyType(Enum):
    NONE = 0
    MULTI_THREADING = 1
    MULTI_PROCESSING = 2
