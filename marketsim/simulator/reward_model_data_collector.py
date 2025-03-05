import numpy as np
import pandas as pd
from collections import defaultdict


class RewardModelDataCollector:
    def __init__(self):
        # Dictionary to store data
        self.data = defaultdict(list)
        self.order_map = {}  # Maps (agent_id, order_id) to index in self.data
        self.index = 0
        
        # Lists for each feature
        self.fundamentals = []
        self.buy_prices = []
        self.sell_prices = []
        self.offer_prices = []
        self.ts = []
        self.Ts = []
        self.sides = []
        self.executed = []
        self.agent_ids = []
        self.order_ids = []
        
    def record_order(self, fundamental, buy_price, sell_price, offer_price,
                    t, T, side, executed, agent_id, order_id):
        """
        Record a new order.
        """
        self.fundamentals.append(fundamental)
        self.buy_prices.append(buy_price)
        self.sell_prices.append(sell_price)
        self.offer_prices.append(offer_price)
        self.ts.append(t)
        self.Ts.append(T)
        self.sides.append(side)
        self.executed.append(executed)
        self.agent_ids.append(agent_id)
        self.order_ids.append(order_id)
        
        # Map the agent_id and order_id to the index
        self.order_map[(agent_id, order_id)] = self.index
        self.index += 1
        
    def update_execution(self, agent_id, order_id, executed):
        """
        Update the execution status of an order.
        """
        idx = self.order_map.get((agent_id, order_id))
        if idx is not None:
            self.executed[idx] = executed
    
    def get_dataframe(self):
        """
        Convert the collected data to a pandas DataFrame.
        """
        # Create a dictionary of data
        data_dict = {
            'fundamental': self.fundamentals,
            'buy_price': self.buy_prices,
            'sell_price': self.sell_prices,
            'offer_price': self.offer_prices,
            't': self.ts,
            'T': self.Ts,
            'side': self.sides,
            'executed': self.executed,
            'agent_id': self.agent_ids,
            'order_id': self.order_ids
        }
        
        # Create a DataFrame
        df = pd.DataFrame(data_dict)
        
        # Calculate z-scores
        df = self.calculate_z_scores(df)
        
        # Calculate time features
        df['t_over_T'] = df['t'] / df['T']
        df['T_minus_t_over_T'] = (df['T'] - df['t']) / df['T']
        
        # Create final feature columns
        features = [
            'fundamental_zscore',
            'buy_price_zscore',
            'sell_price_zscore',
            'offer_price_zscore',
            't_over_T',
            'T_minus_t_over_T',
            'side',
            'executed'
        ]
        
        return df[features]
    
    def calculate_z_scores(self, df):
        """
        Calculate z-scores for the numerical features.
        """
        # Handle NaN values in buy_price and sell_price
        df['buy_price'].fillna(df['buy_price'].mean(), inplace=True)
        df['sell_price'].fillna(df['sell_price'].mean(), inplace=True)
        
        # Calculate z-scores
        df['fundamental_zscore'] = (df['fundamental'] - df['fundamental'].mean()) / df['fundamental'].std()
        df['buy_price_zscore'] = (df['buy_price'] - df['buy_price'].mean()) / df['buy_price'].std()
        df['sell_price_zscore'] = (df['sell_price'] - df['sell_price'].mean()) / df['sell_price'].std()
        df['offer_price_zscore'] = (df['offer_price'] - df['offer_price'].mean()) / df['offer_price'].std()
        
        return df 