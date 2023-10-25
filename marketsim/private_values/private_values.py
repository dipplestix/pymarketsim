import torch
from ....fourheap.constants import BUY, SELL


class PrivateValues:
    def __init__(self, q_max: int, val_var=1):
        self.values = torch.randn(2*q_max)*val_var
        self.values, _ = self.values.sort(descending=True)

        self.offset = q_max // 2

        self.extra_buy = min(self.values[-1].item(), 0)
        self.extra_sell = max(self.values[0].item(), 0)

    def value_for_exchange(self, position: int, order_type: int) -> float:
        index = position + self.offset - (1 if order_type == SELL else 0)
        if index >= len(self.values):
            return self.extra_buy*order_type
        elif index < 0:
            return self.extra_sell*order_type
        else:
            return self.values[index]*order_type

    def value_at_position(self, position: int) -> float:
        value = 0
        position += self.offset
        if position > self.offset:
            index = min(position, len(self.values))
            value += (position - index)*self.extra_buy
            value += torch.sum(self.values[self.offset:index])
        else:
            index = max(0, position)
            value += (position - index)*self.extra_sell
            value -= torch.sum(self.values[index:self.offset])
        return value
