import json

class Regulator:
    bid_correct = 7777
    ask_correct = 8888

    def __init__(self):
        self.bids_nbbo = []
        self.asks_nbbo = []
        self.fix_bogus_info = []
        self.overwrite_sip = []
        self.time_needed_to_fix_bogus_info = []
        self.has_started_fixing_bogus_info = []

    def initialize(self, final_time, market_num):
        self.overwrite_sip = [False] * final_time.get()
        self.fix_bogus_info = [[False] * final_time.get() for _ in range(market_num)]
        self.has_started_fixing_bogus_info = [False] * market_num

    def set_overwrite_sip(self, val, curr_time):
        self.overwrite_sip[curr_time.get() - 1] = val

    def try_fix_bogus_info(self, is_bogus_info_now, curr_time, sip, market_state):
        if is_bogus_info_now:
            for market_id, has_started in enumerate(self.has_started_fixing_bogus_info):
                if not market_state[market_id] and not has_started:
                    time_offset = self.time_needed_to_fix_bogus_info[market_id]
                    self.fix_bogus_info[market_id][curr_time.get() + time_offset - 1] = True
                    self.has_started_fixing_bogus_info[market_id] = True
        else:
            self.has_started_fixing_bogus_info = [False] * len(self.has_started_fixing_bogus_info)

        for market_id, market in enumerate(self.fix_bogus_info):
            if market[curr_time.get() - 1]:
                sip.set_bogus_info_end_time(curr_time, market_id, True)

    def set_time_needed_to_fix_bogus_info(self, time_needed_to_fix_bogus_info_string):
        self.time_needed_to_fix_bogus_info = list(map(int, time_needed_to_fix_bogus_info_string.split('/')))

    def get_newest_nbbo_bid_and_ask(self, time, sip_bid, sip_ask):
        need_regulation = self.overwrite_sip[time.get() - 1]
        nbbo_bid = self.bid_correct if need_regulation else sip_bid
        nbbo_ask = self.ask_correct if need_regulation else sip_ask
        self.update_nbbo_bid_and_ask(nbbo_bid, nbbo_ask, time)
        return (nbbo_bid, nbbo_ask)

    def update_nbbo_bid_and_ask(self, nbbo_bid, nbbo_ask, time):
        self.update_series(self.bids_nbbo, nbbo_bid, time)
        self.update_series(self.asks_nbbo, nbbo_ask, time)

    @staticmethod
    def update_series(series, value, time):
        if not series or series[-1][0].get() < time.get() - 1:
            series.append((0, 0))
        while time.get() > series[-1][0].get() + 1:
            last_time, last_value = series[-1]
            series.append((last_time.get() + 1), last_value)
        if series[-1][0] == time:
            series[-1] = (time, value)
        else:
            series.append((time, value))

    def get_features(self):
        features = {}
        bids = [[obs[0].get(), obs[1].get()] for obs in self.bids_nbbo]
        asks = [[obs[0].get(), obs[1].get()] for obs in self.asks_nbbo]
        features["Event timestamp, NBBO bid:"] = bids
        features["Event timestamp, NBBO ask:"] = asks
        return json.dumps(features, indent=4)