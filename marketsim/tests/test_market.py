# import sys
# print(sys.path)
from fourheap.fourheap import FourHeap
from fourheap.order import Order
from fourheap.constants import BUY, SELL
from event.event_queue import EventQueue
from market.market import Market
from fundamental.mean_reverting import GaussianMeanReverting


# %%
e = EventQueue()

# %%
order1 = Order(price=1, quantity=1, time=1, agent_id=1, order_id=1, order_type=BUY)
order2 = Order(price=3, quantity=1, time=1, agent_id=2, order_id=2, order_type=BUY)
order3 = Order(price=4, quantity=1, time=3, agent_id=3, order_id=3, order_type=SELL)


# %%
e.schedule_activity(order1)
e.schedule_activity(order2)
e.schedule_activity(order3)


# %%
print(e.step())

# %%
print(e.step())
# %%
print(e.step())
# %%
f = GaussianMeanReverting(mean=100, r=0.2, final_time=100, shock_var=10)

m = Market(f, time_steps=60000)
m.get_fundamental_value()

m.add_order(order1)
m.add_order(order2)
m.add_order(order3)

m.step()
m.step()
m.step()
m.step()

print("___________________")
print(m.event_queue.scheduled_activities)
print(m.matched_orders)

print("________________")
print(m.order_book.buy_unmatched.peek_order())