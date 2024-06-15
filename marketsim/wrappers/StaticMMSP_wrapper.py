import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
from fourheap.constants import BUY, SELL
from market.market import Market
from fundamental.mean_reverting import GaussianMeanReverting
from agent.zero_intelligence_agent import ZIAgent
from agent.hbl_agent import HBLAgent
from agent.spoofer import SpoofingAgent
from agent.market_maker import MMAgent
from wrappers.metrics import volume_imbalance, queue_imbalance, realized_volatility, relative_strength_index, midprice_move
import torch.distributions as dist
import torch
from collections import defaultdict
import matplotlib.pyplot as plt

COUNT = 0
DATA_SAVE_PATH = "spoofer_mm_exps/rl/low_liq_PPO_low_shock/a"

def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 10000 sampled time steps

class MMSPEnv(gym.Env):
    def __init__(self,
                 num_background_agents: int,
                 sim_time: int,
                 num_assets: int = 1,
                 lam: float = 5e-3,
                 lamSP: float = 5e-2, # Tune
                 lamMM: float = 8e-2,
                 mean: float = 1e5,
                 r: float = 0.05,
                 shock_var: float = 5e6,
                 q_max: int = 10,
                 pv_var: float = 5e6,
                 shade=None,
                 xi: float = 100, # rung size
                 omega: float = 256, #spread
                 K: int = 8,
                 normalizers=None,
                 fundamental = None,
                 order_size=1, # the size of regular order: NEED TUNING
                 spoofing_size=200, # the size of spoofing order: NEED TUNING
                 pvalues = None,
                 sampled_arr = None,
                 spoofer_arrivals = None,
                 MM_arrivals = None,
                 learning = False,
                 learnedActions = False,
                 analytics = False,
                 random_seed = None,
                 ):

        # MarketSim Setup
        if shade is None:
            shade = [10, 30]
        self.num_agents = num_background_agents
        self.num_assets = num_assets
        self.sim_time = sim_time
        self.lam = lam
        self.time = 0

        self.sampled_arr = sampled_arr
        self.spoofer_arrivals = spoofer_arrivals
        self.MM_arrivals = MM_arrivals

        self.learning = learning
        self.learnedActions = learnedActions
        self.final_fundamental = 1e5

        self.pvalues = pvalues

        if learning == True:
            self.sampled_arr = sampled_arr = sample_arrivals(lam,sim_time)
            self.spoofer_arrivals = sample_arrivals(lamSP,sim_time)
            self.MM_arrivals = sample_arrivals(lamMM, sim_time)
            random_seed = [random.randint(0,100000) for _ in range(sim_time)]
            fundamental = GaussianMeanReverting(mean=mean, final_time=sim_time + 1, r=r, shock_var=shock_var)
            self.pvalues = [-1] * (num_background_agents - 1)
            self.spoof_position = []
            self.spoof_profits = []
            self.spoofer_orders = [[], []]

        self.analytics = analytics
        if analytics:
            self.most_recent_trade = {key: np.nan for key in range(0, sim_time + 1)}
            self.spoof_orders = {key: (np.nan,0) for key in range(0, sim_time + 1)}
            self.sell_orders = {key: np.nan for key in range(0, sim_time + 1)}
            self.best_buys = {key: np.nan for key in range(0, sim_time + 1)}
            self.best_asks = {key: np.nan for key in range(0, sim_time + 1)}
            self.sell_above_best = []
            self.spoofer_quantity = {key: np.nan for key in range(0, sim_time + 1)}
            self.spoofer_value = {key: np.nan for key in range(0, sim_time + 1)}
            self.trade_volume = {key: 0 for key in range(0, self.sim_time + 1)}
            self.mid_prices = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.spoof_activity = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.aggregate_behavior = []
            self.aggregate_above_ask = []
            self.aggregate_below_buy = []

        # Regular traders
        self.arrivals = defaultdict(list)
        self.arrivals_sampled = 10000
        self.arrival_times = sample_arrivals(lam, self.arrivals_sampled)
        self.arrival_index = 0

        self.normalizers = normalizers

        # Spoofer
        self.lamSP = lamSP
        self.arrivals_SP = defaultdict(list)
        if spoofer_arrivals != None:
            self.arrival_times_SP = spoofer_arrivals
        else:
            self.arrival_times_SP = sample_arrivals(lamSP, sim_time)
        self.arrival_index_SP = 0

        # MM
        self.lamMM = lamMM
        self.arrivals_MM = defaultdict(list)
        self.arrival_times_MM  = sample_arrivals(lamMM, self.arrivals_sampled)
        self.arrival_index_MM = 0

        self.mean = mean
        self.shock_var = shock_var
        self.r = r
        self.markets = []
        if num_assets > 1:
            raise NotImplemented("Only support single market currently")

        self.marketConfig = {"mean": mean, "r": r, "shock_var": shock_var, "num_assets": num_assets}
        for _ in range(num_assets):
            fundamental = fundamental
            self.markets.append(Market(fundamental=fundamental, time_steps=sim_time))

        self.agents = {}
        self.backgroundAgentConfig = {"q_max":q_max, "pv_var": pv_var, "shade": shade, "L": 4, "spoof_size": spoofing_size, "reg_size": order_size}
        for agent_id in range(12):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
            self.arrival_index += 1
            self.agents[agent_id] = (
                ZIAgent(
                    agent_id=agent_id,
                    market=self.markets[0],
                    q_max=q_max,
                    shade=shade,
                    pv_var=pv_var,
                    pv=self.pvalues[agent_id]
                ))

        for agent_id in range(12, self.num_agents - 1):
                self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
                self.arrival_index += 1
                self.agents[agent_id] = (HBLAgent(
                    agent_id = agent_id,
                    market = self.markets[0],
                    pv_var = pv_var,
                    q_max= q_max,
                    shade = shade,
                    L = 4,
                    arrival_rate = self.lam,
                    pv=self.pvalues[agent_id]
                ))

        # Set up for market makers.
        self.arrivals_MM[self.arrival_times_MM[self.arrival_index_MM].item()].append(self.num_agents - 1)
        self.arrival_index_MM += 1
        self.MM_id = self.num_agents - 1
        self.MM = MMAgent(
                agent_id=self.MM_id,
                market=self.markets[0],
                xi=xi,
                K=K,
                omega=omega,
            )

        # Set up for spoofer.
        self.arrivals_SP[self.arrival_times_SP[self.arrival_index_SP].item() + 1000].append(self.num_agents)
        self.arrival_index_SP += 1

        self.spoofer = SpoofingAgent(
            agent_id=self.num_agents,
            market=self.markets[0],
            q_max=q_max,
            order_size=order_size,
            spoofing_size=spoofing_size,
            normalizers=normalizers,
            learning=learnedActions,
        )

        self.random_seed = random_seed

        # Gym Setup
        """
        Given a market state s when the self agent arrives at time t, 
        1.the agent receives an observation O(s) that includes the number of time steps left T − t, 
        2.the current fundamental value rt, 
        3.the current best BID price in the limit order book (if any), 
        4.the current best ASK price in the limit order book (if any), 
        5.the self agent’s inventory I, 
        ------
        Extra from Rahul's paper:
        6.Mid-price move,
        7.Volume imbalance
        8.Queue imbalance,
        9.Volatility,
        10.Relative strength index,
        11.Est-fundamental,
        12.Midprice
        """
        obs_space_low = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        obs_space_high = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])
        # obs_space_low = np.concatenate([obs_space_low, pv_low])
        # obs_space_high = np.concatenate([obs_space_high, pv_high])

        # self.spoofer_norm_pvs = np.array(self.spoofer.pv.values) / pv_var

        self.observation_space = spaces.Box(low=(obs_space_low),
                                    high=(obs_space_high),
                                    shape=(len(obs_space_low),),
                                    dtype=np.float64) # Need rescale the obs.
        self.action_space = spaces.Box(low=np.array([0.01, 0.1]), high=np.array([1.0, 1.0]), dtype=np.float32) # price for regular order and price for spoofing

    def get_obs(self):
        return self.observation

    def update_obs(self):
        time_left = self.sim_time - self.time
        fundamental_value = self.markets[0].fundamental.get_value_at(self.time)
        est_fund = self.spoofer.estimate_fundamental()

        best_ask = self.markets[0].order_book.get_best_ask()
        best_bid = self.markets[0].order_book.get_best_bid()
        SPinvt = self.spoofer.position

        midprice_delta = midprice_move(self.markets[0])
        vol_imbalance = volume_imbalance(self.markets[0])
        que_imbalance = queue_imbalance(self.markets[0])
        vr = realized_volatility(self.markets[0])
        rsi = relative_strength_index(self.markets[0])
        
        if not math.isinf(best_ask) and not math.isinf(best_bid):
            midprice = (best_bid + best_ask) / 2
        else:
            midprice = 0
        
        self.observation = self.normalization(
            time_left=time_left,
            fundamental_value=fundamental_value,
            best_ask=best_ask,
            best_bid=best_bid,
            SPinvt=SPinvt,
            midprice_delta=midprice_delta,
            vol_imbalance=vol_imbalance,
            que_imbalance=que_imbalance,
            vr=vr,
            rsi=rsi,
            estimated_fundamental=est_fund,
            midprice=midprice
            )

    def normalization(self,
                      time_left: int,
                      fundamental_value: float,
                      best_ask: float,
                      best_bid: float,
                      SPinvt: float,
                      midprice_delta: float,
                      vol_imbalance: float,
                      que_imbalance: float,
                      vr: float,
                      rsi: float,
                      estimated_fundamental: float,
                      midprice: float,
                      ):
        '''
            We need to define min/max bounds for the price or else the range is TOO BIG.
            Can't have a fundamental of 1e5 and a spoofing order of price = 7 for example. 
        '''
        if self.normalizers is None:
            print("No normalizer warning!")
            return np.array([time_left, fundamental_value, best_ask, best_bid, SPinvt])

        time_left /= self.sim_time
        #TODO
        fundamental_value /= self.normalizers["fundamental"]
        estimated_fundamental /= self.normalizers["fundamental"]
        # print("BEST ASK")
        # print(best_ask, best_bid)
        
        if math.isinf(abs(best_ask)):
            best_ask = 1.01
        else:
            #TODO: FIX
            best_ask /= self.normalizers["fundamental"] 

        if math.isinf(abs(best_bid)):
            best_bid = 0.98
        else:
            best_bid /= self.normalizers["fundamental"]

        SPinvt /= self.normalizers["invt"]

        midprice_delta /= 2e2  # TODO: need to tune
        rsi /= 100

        midprice /= self.normalizers["fundamental"]

        obs = np.array([time_left,
                         fundamental_value,
                         best_ask,
                         best_bid,
                         SPinvt,
                         midprice_delta,
                         vol_imbalance,
                         que_imbalance,
                         vr,
                         rsi,
                         estimated_fundamental,
                         midprice             
                         ])
        return obs

    def reset(self, seed=None, options=None):
        self.time = 0

        self.observation = None
        if self.learning:
            #When learning, want to change fundamental and PVs so RL learns from the distribution
            # and not a specific instance.
            for market in self.markets:
                market.fundamental._generate()
                market.reset()
            
            self.final_fundamental = self.markets[0].get_final_fundamental()
            self.random_seed = [random.randint(0,100000) for _ in range(10000)]
            
            for agent_id in range(self.num_agents - 1):
                agent = self.agents[agent_id]
                agent.generate_pv()
                agent.reset()

            # self.spoof_order_history = np.zeros(self.action_history_length)
            # self.reg_order_history = np.zeros(self.action_history_length)
            # self.current_order_count = 0
        else:
            # Reset the markets
            for market in self.markets:
                market.reset()

            self.final_fundamental = self.markets[0].get_final_fundamental()

            # Reset the agents
            for agent_id in range(self.num_agents - 1):
                agent = self.agents[agent_id]
                agent.reset()

        # Reset MM and spoofer
        self.MM.reset()
        self.spoofer.reset()

        self.spoof_position = []
        self.spoof_profits = []
        self.spoofer_orders = [[], []]
        # Reset Arrivals
        self.reset_arrivals()

        if self.analytics:
            self.most_recent_trade = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.spoof_orders = {key: (np.nan,0) for key in range(0, self.sim_time + 1)}
            self.sell_orders = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.best_buys = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.best_asks = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.sell_above_best = []
            self.buy_below_best = []
            self.spoofer_quantity = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.spoofer_value = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.trade_volume = {key: 0 for key in range(0, self.sim_time + 1)}
            self.mid_prices = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.spoof_activity = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.aggregate_behavior = []
            self.aggregate_above_ask = []
            self.aggregate_below_buy = []

        end = self.run_until_next_SP_arrival()
        if end:
            input()
        #     raise ValueError("An episode without spoofer. Length of an episode should be set large.")

        return self.get_obs(), {}

    def reset_arrivals(self):
        # Regular Trader
        self.arrivals = defaultdict(list)
        self.arrivals_SP = defaultdict(list)
        self.arrivals_MM = defaultdict(list)

        self.arrivals_sampled = self.sim_time
        # self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)

        if self.learning:
            self.arrival_times = sample_arrivals(self.lam, self.sim_time)
            self.arrival_times_SP = sample_arrivals(self.lamSP, self.sim_time)
            self.arrival_times_MM = sample_arrivals(self.lamMM, self.sim_time)
        else:
            self.arrival_times = self.sampled_arr
            self.arrival_times_SP = self.spoofer_arrivals
            self.arrival_times_MM = self.MM_arrivals

        self.arrival_index = 0
        self.arrival_index_SP = 0
        self.arrival_index_MM = 0

        for agent_id in range(self.num_agents - 1):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
            self.arrival_index += 1

        #Xintong paper - Spoofer arrives after timestep 1000
        self.arrivals_SP[self.arrival_times_SP[self.arrival_index_SP].item() + 1000].append(self.num_agents)
        self.arrival_index_SP += 1

        self.arrivals_MM[self.arrival_times_MM[self.arrival_index_MM].item()].append(self.MM_id)
        self.arrival_index_MM += 1


    def step(self, action):
        # print("----midprices：", self.MM.market.get_midprices())
        # print("----Best ask：", self.MM.market.order_book.get_best_ask())
        # print("----Best bid：", self.MM.market.order_book.get_best_bid())
        if self.time < self.sim_time:
            if len(self.arrivals_SP[self.time]) != 0:
                self.SP_step(action)
            self.agents_step()
            reward = self.market_step(agent_only=False)
            self.time += 1
            end = self.run_until_next_SP_arrival()
            if end:
                return self.end_sim()
            return self.get_obs(), reward, False, False, {}
        else:
            return self.end_sim()


    def agents_step(self):
        agents = self.arrivals[self.time]
        agents.extend(self.arrivals_MM[self.time])
        if len(agents) != 0:
            for market in self.markets:
                market.event_queue.set_time(self.time)
                for agent_id in agents:
                    if agent_id != self.MM_id:
                        agent = self.agents[agent_id]
                    else:
                        agent = self.MM
                    market.withdraw_all(agent_id)
                    if agent_id != self.MM_id:
                        random.seed(self.time + self.random_seed[self.time])
                        side = random.choice([BUY, SELL])
                        orders = agent.take_action(side, seed = self.random_seed[self.time])
                    else:
                        orders = agent.take_action()
                    market.add_orders(orders)

                    if self.arrival_index == self.arrivals_sampled:
                        self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)
                        self.arrival_index = 0
                    
                    if self.arrival_index_MM == self.arrivals_sampled:
                        self.arrival_times_MM = sample_arrivals(self.lamMM, self.arrivals_sampled)
                        self.arrival_index_MM = 0

                    if agent_id != self.MM_id:
                        self.arrivals[self.arrival_times[self.arrival_index].item() + 1 + self.time].append(agent_id)
                        self.arrival_index += 1
                    else:
                        self.arrivals_MM[self.arrival_times_MM[self.arrival_index_MM].item() + 1 + self.time].append(agent_id)
                        self.arrival_index_MM += 1
                    

    def SP_step(self, action):
        for market in self.markets:
            market.event_queue.set_time(self.time)
            market.withdraw_all(self.num_agents)
            orders, status = self.spoofer.take_action(action, seed=self.random_seed[self.time])
            market.add_orders(orders)
            #Regular FIRST Spoof SECOND
            self.spoofer_orders[0].append((action[0], status))
            self.spoofer_orders[1].append(action[1])
            if self.analytics:
                self.spoof_orders[self.time] = (orders[1].price, status)
                self.sell_orders[self.time] = orders[0].price
                self.sell_above_best.append(orders[0].price - market.order_book.sell_unmatched.peek())
                self.buy_below_best.append(market.order_book.buy_unmatched.peek() - orders[1].price)


            if self.arrival_index_SP == self.arrivals_sampled:
                self.arrival_times_SP = self.spoofer_arrivals
                self.arrival_index_SP = 0
            self.arrivals_SP[self.arrival_times_SP[self.arrival_index_SP].item() + 1 + self.time].append(self.num_agents)
            self.arrival_index_SP += 1

    def MM_step(self):
        for market in self.markets:
            market.event_queue.set_time(self.time)
            market.withdraw_all(self.num_agents)
            orders = self.MM.take_action()
            market.add_orders(orders)

            if self.arrival_index_MM == self.arrivals_sampled:
                self.arrival_times_MM = sample_arrivals(self.lamMM, self.arrivals_sampled)
                self.arrival_index_MM = 0
            self.arrivals_MM[self.arrival_times_MM[self.arrival_index_MM].item() + 1 + self.time].append(self.num_agents)
            self.arrival_index_MM += 1

    def market_step(self, agent_only=True, verbose=False):
        # if verbose:
        #     print("----Last Best ask：", self.MM.market.order_book.get_best_ask())
        #     print("----Last Best bid：", self.MM.market.order_book.get_best_bid())
        for market in self.markets:
            new_orders = market.step()
            for matched_order in new_orders:
                agent_id = matched_order.order.agent_id
                quantity = matched_order.order.order_type * matched_order.order.quantity
                cash = -matched_order.price * matched_order.order.quantity * matched_order.order.order_type
                if agent_id == self.num_agents:
                    self.spoofer.update_position(quantity, cash)
                    self.spoof_position.append(self.spoofer.position)
                    a = (self.spoofer.position*self.final_fundamental + self.spoofer.cash)
                    self.spoof_profits.append((self.spoofer.position*self.final_fundamental + self.spoofer.cash))
                elif agent_id == self.MM_id:
                    self.MM.update_position(quantity, cash)
                else:
                    self.agents[agent_id].update_position(quantity, cash)

            # ANALYTICAL DATA
            if self.analytics:
                self.trade_volume[self.time] = len(new_orders) // 2
                if len(self.markets[0].matched_orders) > 0:
                    self.most_recent_trade[self.time] = self.markets[0].matched_orders[-1].price
                if not math.isinf(self.markets[0].order_book.sell_unmatched.peek()):
                    self.best_asks[self.time] = self.markets[0].order_book.sell_unmatched.peek()
                if not math.isinf(self.markets[0].order_book.buy_unmatched.peek()):
                    self.best_buys[self.time] = self.markets[0].order_book.buy_unmatched.peek() 
                if not math.isinf(self.markets[0].order_book.sell_unmatched.peek()) and not math.isinf(self.markets[0].order_book.buy_unmatched.peek()):
                    self.mid_prices[self.time] = (self.best_asks[self.time] + self.best_buys[self.time]) / 2
                if len(self.markets[0].matched_orders) > 0:
                    self.most_recent_trade[self.time] = self.markets[0].matched_orders[-1].price
            
                # SPOOFER ANALYTICS
                self.spoofer_quantity[self.time] = self.spoofer.position
                self.spoof_activity[self.time] = (self.spoofer.position*self.final_fundamental + self.spoofer.cash)

            if not agent_only:
                current_value = (self.spoofer.position*self.final_fundamental + self.spoofer.cash)
                reward = current_value - self.spoofer.last_value
                self.spoofer.last_value = current_value  # TODO: Check if we need to normalize the reward
                
                # if verbose:
                #     estimated_fundamental = self.spoofer.estimate_fundamental()
                #     print("----matched orders:", new_orders)
                #     print("----estimated_fundamental:", estimated_fundamental)
                #     print("----Best ask: ", self.MM.market.order_book.get_best_ask())
                #     print("----Best bid: ", self.MM.market.order_book.get_best_bid())
                #     print("----Bids: ", self.MM.market.order_book.buy_unmatched)
                #     print("----Asks: ", self.MM.market.order_book.sell_unmatched)

                return reward / self.normalizers["reward"]  # TODO: Check if this normalizer works.
                
    def end_sim_summarize(self):
        fundamental_val = self.markets[0].get_final_fundamental()
        values = {}
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            values[agent_id] = agent.position * fundamental_val + agent.cash

        values[self.num_agents] = self.MM.position * fundamental_val + self.MM.cash
        # print(f'At the end of the simulation we get {values}')

    def end_sim(self):
        # estimated_fundamental = self.spoofer.estimate_fundamental()
        current_value = (self.spoofer.position*self.final_fundamental + self.spoofer.cash)
        reward = current_value - self.spoofer.last_value
        global COUNT
        print(COUNT)
        if COUNT % 50 == 0 and self.learning:
            self.aggregate_behavior.append(list(self.most_recent_trade.values()))
            above_ask_pad = np.full(400, np.nan)
            above_ask_pad[:len(self.sell_above_best)] = self.sell_above_best
            self.aggregate_above_ask.append(above_ask_pad)
            below_buy_pad = np.full(400, np.nan)
            below_buy_pad[:len(self.buy_below_best)] = self.buy_below_best
            self.aggregate_below_buy.append(below_buy_pad)

            plt.figure()
            plt.plot(list(self.most_recent_trade.keys()), np.nanmean(self.aggregate_behavior, axis=0))
            plt.xlabel('Timestep')
            plt.ylabel('Price Level')
            plt.savefig(DATA_SAVE_PATH + "/{}_diff_sim.jpg".format(COUNT))
            plt.close()
            
            plt.figure()
            plt.scatter(np.arange(len(self.sell_above_best)), self.sell_above_best, s=10)
            plt.xlabel('Order entry')
            plt.savefig(DATA_SAVE_PATH + "/{}_sell_above_best.jpg".format(COUNT))
            plt.ylabel('Sell price - best_ask')
            plt.close()

            plt.figure()
            plt.scatter(np.arange(len(self.buy_below_best)), -np.array(self.buy_below_best), s=10)
            plt.xlabel('Order entry')
            plt.ylabel('Spoof - best buy')
            plt.savefig(DATA_SAVE_PATH + "/{}_buy_below_best.jpg".format(COUNT))
            plt.close()

            plt.figure()
            colors = ['red' if value == 1 else 'blue' for value in np.array(self.spoofer_orders[0])[:,1]]
            plt.scatter([i for i in range(len(self.spoofer_orders[0]))], np.array(self.spoofer_orders[0])[:,0], c=colors, label="reg order: red(midpirce) blue(fund)", s=15)
            plt.title('Reg Orders')
            plt.xlabel('Order Entry')
            plt.ylabel('Normalized action')
            plt.legend()
            plt.savefig(DATA_SAVE_PATH + "/{}_reg_order.jpg".format(COUNT))
            plt.close()

            plt.figure()
            plt.scatter([i for i in range(len(self.spoofer_orders[1]))], self.spoofer_orders[1], label="reg orders", s=15)
            plt.title('Spoof Orders')
            plt.xlabel('Order Entry')
            plt.ylabel('Normalized action')
            plt.savefig(DATA_SAVE_PATH + "/{}_spoof_order.jpg".format(COUNT))
            plt.close()


            plt.figure()
            plt.title('Average Sell Above Ask')
            plt.scatter(np.arange(400), np.nanmean(self.aggregate_above_ask, axis=0), s=15)
            plt.xlabel('Order Entry')
            plt.ylabel('Sell above ask')
            plt.savefig(DATA_SAVE_PATH + "/{}_AVG_above_ask.jpg".format(COUNT))
            plt.close()

            plt.figure()
            plt.scatter(np.arange(400), -np.nanmean(self.aggregate_below_buy, axis=0), s=10)
            plt.xlabel('Order entry')
            plt.ylabel('Spoof - best buy')
            plt.savefig(DATA_SAVE_PATH + "/{}_AVG_buy_below.jpg".format(COUNT))
            plt.close()

            f = open(DATA_SAVE_PATH + "/{}_position_profit.txt".format(COUNT), "a")
            f.write(str(self.spoof_position))
            f.write(str(self.spoof_profits))
            f.close()

        COUNT += 1
        current_value = (self.spoofer.position*self.final_fundamental + self.spoofer.cash)
        reward = current_value - self.spoofer.last_value

        return self.get_obs(), reward / self.normalizers["reward"], True, False, {}

    def run_until_next_SP_arrival(self):
        while len(self.arrivals_SP[self.time]) == 0 and self.time < self.sim_time:
            self.agents_step()
            self.market_step(agent_only=True)
            # print(self.markets[0].order_book.observe())
            self.time += 1

        if self.time >= self.sim_time:
            return True
        else:
            self.update_obs()
            return False

    def run_agents_only(self):
        for t in range(int(0.01 * self.sim_time)):
            if self.arrivals[t]:
                self.agents_step()
                self.market_step(agent_only=True)
            self.time += 1


