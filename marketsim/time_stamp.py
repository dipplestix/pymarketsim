class TimeStamp:
    def __init__(self, ticks: int):
        self.ticks = ticks

    @staticmethod
    def of(ticks: int):
        return TimeStamp(ticks)

    def get(self):
        return self.ticks

    def __lt__(self, other):
        return self.ticks < other.ticks

    def __eq__(self, other):
        if other is None or not isinstance(other, TimeStamp):
            return False
        return self.ticks == other.ticks

    def __hash__(self):
        return hash(self.ticks)

    def __str__(self):
        return f"{self.ticks}t"
