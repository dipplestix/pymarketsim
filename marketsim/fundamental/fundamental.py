from abc import ABC, abstractmethod
from typing import List, Tuple, Protocol
import torch


class Fundamental(ABC):
    @abstractmethod
    def get_value_at(self, time: int) -> float:
        pass

    @abstractmethod
    def get_fundamental_values(self) -> torch.Tensor:
        pass

    # @abstractmethod
    # def get_view(self, sim: 'Sim') -> 'FundamentalView':
    #     pass
