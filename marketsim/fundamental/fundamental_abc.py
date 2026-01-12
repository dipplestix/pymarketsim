from abc import ABC, abstractmethod


class Fundamental(ABC):
    @abstractmethod
    def get_value_at(self, time: int) -> float:
        pass

    @abstractmethod
    def get_fundamental_values(self):
        pass

    @abstractmethod
    def get_info(self):
        pass

    @abstractmethod
    def get_final_fundamental(self) -> float:
        pass

