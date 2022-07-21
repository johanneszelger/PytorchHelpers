from enum import Enum


class BalancingMode(Enum):
    UNDERSAMPLE = 1

    def __str__(self):
        return self.name.lower()