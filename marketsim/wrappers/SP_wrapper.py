import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from fourheap.constants import BUY, SELL
from market.market import Market
from fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from wrappers.metrics import volume_imbalance, queue_imbalance, realized_volatility, relative_strength_index, midprice_move
from fundamental.mean_reverting import GaussianMeanReverting
from agent.zero_intelligence_agent import ZIAgent
from agent.spoofer import SpoofingAgent
from agent.hbl_agent import HBLAgent
import torch.distributions as dist
import torch
from collections import defaultdict
def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 1000 sampled time steps

COUNT = 0
DATA_SAVE_PATH = "spoofer_exps/PPO_options/16_RPPO/a"
'''
Helpful notes:

Ep_length: Number of times the Spoofer enters because step will run intermediate steps
lamSP: 5e-3, 5e-2 (~50-55 entries per 1000), 5e-1 (~500 entries)
'''
class SPEnv(gym.Env):
    def __init__(self,
                 num_background_agents: int,
                 sim_time: int,
                 num_assets: int = 1,
                 lam: float = 5e-3,
                 lamSP: float = 5e-2, # Tune
                 mean: float = 1e5,
                 r: float = 0.05,
                 shock_var: float = 1e5,
                 q_max: int = 10,
                 pv_var: float = 5e6,
                 shade=None,
                 order_size=1, # the size of regular order: NEED TUNING
                 spoofing_size=200, # the size of spoofing order: NEED TUNING
                 normalizers = None, # normalizer for obs: NEED TUNING
                 pvalues = None,
                 sampled_arr = None,
                 spoofer_arrivals = None,
                 fundamental = None,
                 learning = False,
                 learnedActions = False,
                 analytics = False,
                 random_seed = None,
                 action_history_length = 500,
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
        self.current_action = None

        self.learning = learning
        self.learnedActions = learnedActions
        self.final_fundamental = 1e5
        
        # self.current_order_count = 0
        # self.spoof_order_history = np.zeros(action_history_length)
        # self.reg_order_history = np.zeros(action_history_length)
        # self.action_history_length = action_history_length
        
        self.pvalues = pvalues

        if learning == True:
            self.sampled_arr = sampled_arr = sample_arrivals(lam,sim_time)
            self.spoofer_arrivals = sample_arrivals(lamSP,sim_time)
            random_seed = [random.randint(0,100000) for _ in range(sim_time)]
            fundamental = GaussianMeanReverting(mean=mean, final_time=sim_time + 1, r=r, shock_var=shock_var)
            self.pvalues = [-1] * (num_background_agents + 1)
            self.spoof_position = []
            self.spoof_profits = []
            self.spoofer_orders = [[], []]
            
        self.analytics = analytics
        
        if analytics:
            self.most_recent_trade = {key: np.nan for key in range(0, sim_time + 1)}
            self.spoof_orders = {key: np.nan for key in range(0, sim_time + 1)}
            self.sell_orders = {key: np.nan for key in range(0, sim_time + 1)}
            self.best_buys = {key: np.nan for key in range(0, sim_time + 1)}
            self.best_asks = {key: np.nan for key in range(0, sim_time + 1)}
            self.sell_above_best = []
            self.buy_below_best = []
            self.spoofer_quantity = {key: np.nan for key in range(0, sim_time + 1)}
            self.spoofer_value = {key: np.nan for key in range(0, sim_time + 1)}
            self.trade_volume = {key: 0 for key in range(0, self.sim_time + 1)}
            self.mid_prices = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.spoof_activity = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.aggregate_behavior = []
            self.aggregate_above_ask = []
            self.aggregate_below_buy = []

        # Regular Trader
        self.arrivals = defaultdict(list)
        self.arrivals_sampled = self.sim_time
        # self.arrival_times = sample_arrivals(lam, self.arrivals_sampled)
        if sampled_arr != None:
            self.arrival_times = sampled_arr
        else:
            self.arrival_times = sample_arrivals(lam, sim_time)

        self.arrival_index = 0

        # Spoofer
        self.lamSP = lamSP
        self.arrivals_SP = defaultdict(list)
        if spoofer_arrivals != None:
            self.arrival_times_SP = spoofer_arrivals
        else:
            self.arrival_times_SP = sample_arrivals(lamSP, sim_time)
        self.arrival_index_SP = 0
        self.normalizers = normalizers

        # Set up markets
        self.marketConfig = {"mean": mean, "r": r, "shock_var": shock_var, "num_assets": num_assets}
        self.markets = []
        for _ in range(num_assets):
            fundamental = fundamental
            self.markets.append(Market(fundamental=fundamental, time_steps=sim_time))

        self.final_fundamental = self.markets[0].get_final_fundamental()
        # Set up for regular traders.
        self.agents = {}
        self.backgroundAgentConfig = {"q_max":q_max, "pv_var": pv_var, "shade": shade, "L": 4, "spoof_size": spoofing_size, "reg_size": order_size}
        for agent_id in range(23):
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

        for agent_id in range(23, self.num_agents):
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

        # Set up for spoofer.
        self.arrivals_SP[self.arrival_times_SP[self.arrival_index_SP].item() + 1000].append(self.num_agents)
        self.arrival_index_SP += 1

        self.spoofer = SpoofingAgent(
            agent_id=self.num_agents,
            market=self.markets[0],
            q_max=q_max,
            # pv_var=pv_var,
            order_size=order_size,
            spoofing_size=spoofing_size,
            normalizers=normalizers,
            learning=learnedActions,
            # pv=self.pvalues[self.num_agents]
        )
                
        self.random_seed = random_seed
        
        self.means = {key: [] for key in range(0, 10001)}
        self.bestSells = {key: [] for key in range(0, 10001)}
        self.bestBids = {key: [] for key in range(0, 10001)}

        # Gym Setup
        """
        Given a market state s when the self agent arrives at time t, 
        1.the agent receives an observation O(s) that includes the number of time steps left T - t, 
        2.the current fundamental value rt, 
        3.the current best BID
        4.the current best ASK price in the limit order book (if any), 
        5.the self agent's inventory I, 
        ------
        Extra from Rahul's paper:
        6.Mid-price move,
        7.Volume imbalance
        8.Queue imbalance,
        9.Volatility,
        10.Relative strength index,
        11.Estimated Fundamental,
        12-31. All PVs
        """
        #TODO: NOT SURE ABOUT MID-PRICE MOVE. HAVE SEEN 1.07
        # self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, -1.0, -10.0, -1.0, -1.0]),
        #                             high=np.array([1.0, 1.0, 2.0, 1.0, 1.0, 10.0]),
        #                             shape=(6,),
        #                             dtype=np.float64) # Need rescale the obs.

        # orig_obs_low = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -10.0, -1.0, -1.0])
        # orig_obs_high = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 10.0, 1.0, 1.0])
        obs_space_low = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0])
        obs_space_high = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
        # pv_low = -2 * np.ones(len(self.spoofer.pv.values))
        # pv_high = 2 * np.ones(len(self.spoofer.pv.values))
        # obs_space_low = np.concatenate([obs_space_low, pv_low])
        # obs_space_high = np.concatenate([obs_space_high, pv_high])

        self.observation_space= spaces.Box(low=(obs_space_low),
                                    high=(obs_space_high),
                                    shape=(len(obs_space_low),),
                                    dtype=np.float64) # Need rescale the obs.
        
        self.action_space = spaces.Box(low=np.array([0.0, 0.1]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32) # price for regular order and price for spoofing

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
        
        
        # prev_actions = np.full(n_history * action_space.shape[0])

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
                         estimated_fundamental                 
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
            
            for agent_id in range(self.num_agents):
                agent = self.agents[agent_id]
                agent.generate_pv()
                agent.reset()

            self.spoofer.reset()
            
            # self.spoof_order_history = np.zeros(self.action_history_length)
            # self.reg_order_history = np.zeros(self.action_history_length)
            # self.current_order_count = 0
        else:
            # Reset the markets
            for market in self.markets:
                market.reset()

            # Reset the agents
            for agent_id in self.agents:
                agent = self.agents[agent_id]
                agent.reset()

            # Reset spoofer
            self.spoofer.reset()

        self.spoof_position = []
        self.spoof_profits = []
        self.spoofer_orders = [[], []]
        # Reset Arrivals
        self.reset_arrivals()

        if self.analytics:
            self.most_recent_trade = {key: np.nan for key in range(0, self.sim_time + 1)}
            self.spoof_orders = {key: np.nan for key in range(0, self.sim_time + 1)}
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

        end = self.run_until_next_SP_arrival()
        if end:
            input()
        #     raise ValueError("An episode without spoofer. Length of an episode should be set large.")

        return self.get_obs(), {}

    def reset_arrivals(self):
        # Regular Trader
        self.arrivals = defaultdict(list)
        self.arrivals_SP = defaultdict(list)

        self.arrivals_sampled = self.sim_time
        # self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)

        if self.learning:
            self.arrival_times = sample_arrivals(self.lam, self.sim_time)
        else:
            self.arrival_times = self.sampled_arr

        self.arrival_index = 0

        if self.learning:
            self.arrival_times_SP = sample_arrivals(self.lamSP, self.sim_time)
        else:
            self.arrival_times_SP = self.spoofer_arrivals

        self.arrival_index_SP = 0

        for agent_id in range(self.num_agents):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
            self.arrival_index += 1

        #Xintong paper - Spoofer arrives after timestep 1000
        self.arrivals_SP[self.arrival_times_SP[self.arrival_index_SP].item() + 1000].append(self.num_agents)
        self.arrival_index_SP += 1

    def step(self, action):
        if self.time < self.sim_time:
            # Only matters for first iteration through.
            # random.seed(self.time + self.random_seed[self.time])
            # side = random.choice([BUY, SELL])
            self.SP_step(action)
            # if self.current_order_count >= self.action_history_length:
            #     self.spoof_order_history = np.roll(self.spoof_order_history, -1)
            #     self.reg_order_history = np.roll(self.reg_order_history, -1)
            # self.spoof_order_history[min(self.current_order_count, self.action_history_length - 1)] = action[0]
            # self.reg_order_history[min(self.current_order_count, self.action_history_length - 1)] = action[1]
            # self.current_order_count += 1
            
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
        if self.time < self.sim_time:
            for market in self.markets:
                market.event_queue.set_time(self.time)
                for agent_id in agents:
                    agent = self.agents[agent_id]
                    market.withdraw_all(agent_id)
                    random.seed(self.time + self.random_seed[self.time])
                    side = random.choice([BUY, SELL])
                    orders = agent.take_action(side, seed=self.random_seed[self.time])
                    market.add_orders(orders)
                   
                    if self.arrival_index == self.arrivals_sampled:
                        self.arrival_times = self.sampled_arr
                        self.arrival_index = 0
                    self.arrivals[self.arrival_times[self.arrival_index].item() + 1 + self.time].append(agent_id)
                    self.arrival_index += 1
                
        else:
            self.end_sim()

    def SP_step(self, action):
        for market in self.markets:
            market.event_queue.set_time(self.time)
            market.withdraw_all(self.num_agents)
            # input(self.time)
            # input(random.random())
            orders = self.spoofer.take_action(action, seed=self.random_seed[self.time])
            market.add_orders(orders)
            #Regular FIRST Spoof SECOND
            self.spoofer_orders[0].append(action[0])
            self.spoofer_orders[1].append(action[1])
            if self.analytics:
                self.spoof_orders[self.time] = orders[1].price
                self.sell_orders[self.time] = orders[0].price
                self.sell_above_best.append(orders[0].price - market.order_book.sell_unmatched.peek())
                self.buy_below_best.append(market.order_book.buy_unmatched.peek() - orders[1].price)

            if self.arrival_index_SP == self.arrivals_sampled:
                self.arrival_times_SP = self.spoofer_arrivals
                self.arrival_index_SP = 0
            self.arrivals_SP[self.arrival_times_SP[self.arrival_index_SP].item() + 1 + self.time].append(self.num_agents)
            self.arrival_index_SP += 1
    
    def market_step(self, agent_only=True):
        for market in self.markets:
            new_orders = market.step()
            for matched_order in new_orders:
                agent_id = matched_order.order.agent_id
                quantity = matched_order.order.order_type * matched_order.order.quantity
                cash = -matched_order.price * matched_order.order.quantity * matched_order.order.order_type
                if agent_id == self.num_agents:
                    self.spoofer.update_position(quantity, cash)
                    self.spoof_position.append(self.spoofer.position)
                    self.spoof_profits.append((self.spoofer.position*self.final_fundamental + self.spoofer.cash, matched_order.time))
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
                # Try using actual fundamental for learning
                # Change to use final fundamental always
                # Get rid of PV
                # Plot profit made on orders
                # Lower spoofing arrival rate
                current_value = (self.spoofer.position*self.final_fundamental + self.spoofer.cash)
                reward = current_value - self.spoofer.last_value
                # reward = current_value 
                self.spoofer.last_value = current_value  # TODO: Check if we need to normalize the reward
                
                # return 0  # TODO: Check if this normalizer works.
                return reward / self.normalizers["reward"]  # TODO: Check if this normalizer works.


    def end_sim_summarize(self):
        fundamental_val = self.markets[0].get_final_fundamental()
        values = {}
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            values[agent_id] = agent.position * fundamental_val + agent.cash

        values[self.num_agents] = self.spoofer.position * fundamental_val + self.spoofer.cash
        # print(f'At the end of the simulation we get {values}')

    def end_sim(self):       
        # estimated_fundamental = self.spoofer.estimate_fundamental()
        current_value = (self.spoofer.position*self.final_fundamental + self.spoofer.cash)
        reward = current_value - self.spoofer.last_value
        # reward = current_value
        global COUNT
        if COUNT % 20 == 0 and self.learning:
            print(COUNT)
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
            plt.scatter([i for i in range(len(self.spoofer_orders[0]))], self.spoofer_orders[0], label="reg orders", s=15)
            plt.title('Reg Orders')
            plt.xlabel('Order Entry')
            plt.ylabel('Normalized action')
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
        for t in range(int(0.1 * self.sim_time)):
            if self.arrivals[t]:
                self.agents_step()
                self.market_step(agent_only=True)
            self.time += 1
