from enum import IntEnum


class MessageLimit(IntEnum):
    NEWLY_CREATED = 250
    SCALING_PATH = 2000
    AUTOMATIC_SCALLING_1 = 10_000
    AUTOMATIC_SCALLING_2 = 100_000
    UNLIMITED = 1_000_000
