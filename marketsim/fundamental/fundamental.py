from abc import ABC, abstractmethod
from typing import List, Tuple, Protocol


class Fundamental(ABC):
    @abstractmethod
    def get_value_at(self, time: int) -> float:
        pass

    @abstractmethod
    def get_fundamental_values(self) -> List[Tuple[float, float]]:
        pass

    # @abstractmethod
    # def get_view(self, sim: 'Sim') -> 'FundamentalView':
    #     pass
