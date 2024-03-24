from abc import ABC, abstractmethod
from marketsim.fourheap.order import Order
from typing import List

class Agent(ABC):
    @abstractmethod
    def get_id(self) -> int:
        pass

    @abstractmethod
    def take_action(self, side: bool) -> List[Order]:
        pass

    def get_pos_value(self) -> float:
        pass
