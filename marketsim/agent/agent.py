from abc import ABC, abstractmethod
from marketsim.fourheap.order import Order


class Agent(ABC):
    @abstractmethod
    def get_id(self) -> int:
        pass

    @abstractmethod
    def take_action(self, side: bool) -> Order:
        pass

    def get_pos_value(self) -> float:
        pass
