from typing import Callable, Tuple, Dict, List
from collections import defaultdict
import heapq
import random
from ..time_stamp import TimeStamp


class EventQueue:
    def __init__(self, rand_seed: int = None):
        self.rand = random.Random(rand_seed)
        self.scheduled_activities = []
        self.pending_scheduled_activities = defaultdict(list)
        self.current_time = TimeStamp(0)

    def more_scheduled_activities(self, time: TimeStamp) -> bool:
        if self.scheduled_activities and self.scheduled_activities[0][0] <= time:
            return True
        for scheduled_time in self.pending_scheduled_activities:
            if scheduled_time <= time:
                return True
        return False

    def execute_until(self, time: TimeStamp):
        while self.more_scheduled_activities(time):
            act_time, act = self.pop()
            self.current_time = act_time
            act()
        if time > self.current_time:
            self.current_time = time

    def pop(self) -> Tuple[TimeStamp, Callable]:
        for time, activities in self.pending_scheduled_activities.items():
            for act in activities:
                heapq.heappush(self.scheduled_activities, (time, act))
        self.pending_scheduled_activities.clear()

        _, act = heapq.heappop(self.scheduled_activities)
        assert act[0] >= self.current_time, "Activities aren't in proper order"
        return act

    def schedule_activity_in(self, delay: TimeStamp, act: Callable):
        self.pending_scheduled_activities[].append(act)

    @property
    def get_current_time(self):
        return self.current_time
