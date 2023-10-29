from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def get_id(self) -> int:
        pass

