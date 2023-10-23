import torch


class PrivateValues:
    def __init__(self, q_max: int):
        self.values = torch.randn(2*q_max)
        self.values, _ = self.values.sort(descending=True)

        self.offset = q_max

        self.extra_buy = min(self.values[-1].item(), 0)
        self.extra_sell = max(self.values[0].item(), 0)

    def value_at_position(self, pos):
        pos += self.offset
