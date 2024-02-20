# %%
from marketsim.event.event_queue import EventQueue
from marketsim.fourheap.order import Order
from marketsim.fourheap.constants import BUY, SELL
from marketsim.market.market import Market
from marketsim.fundamental.mean_reverting import GaussianMeanReverting


# %%
e = EventQueue()

# %%
order1 = Order(price=1, quantity=1, time=1, agent_id=1, order_id=1, order_type=BUY)
order2 = Order(price=1, quantity=1, time=1, agent_id=1, order_id=2, order_type=BUY)
order3 = Order(price=1, quantity=1, time=3, agent_id=1, order_id=3, order_type=SELL)


# %%
e.schedule_activity(order1)
e.schedule_activity(order2)
e.schedule_activity(order3)


# %%
e.step()

# %%
e.step()

# %%
e.step()

# %%
f = GaussianMeanReverting(mean=100, r=0.2, final_time=100, shock_var=10)

# %%
m = Market(f, time_steps=60000)

# %%
m.get_fundamental_value()

# %%
m.add_order(order1)
m.add_order(order2)
m.add_order(order3)


# %%
m.step()

# %%
m.event_queue.scheduled_activities

# %%
m.get_time()

# %%



