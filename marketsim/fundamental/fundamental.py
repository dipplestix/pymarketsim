from abc import ABC, abstractmethod
import torch


class Fundamental(ABC):
    @abstractmethod
    def get_value_at(self, time: int) -> float:
        pass

    @abstractmethod
    def get_fundamental_values(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_final_time(self) -> int:
        pass
    # @abstractmethod
    # def get_view(self, sim: 'Sim') -> 'FundamentalView':
    #     pass
