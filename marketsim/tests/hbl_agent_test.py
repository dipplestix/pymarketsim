import matplotlib.pyplot as plt
import numpy as np
from fourheap.fourheap import FourHeap
from fourheap.order import Order
from fourheap.constants import BUY, SELL
from event.event_queue import EventQueue
from market.market import Market
from fundamental.mean_reverting import GaussianMeanReverting
from agent.hbl_agent import HBLAgent

e = EventQueue()

order1 = Order(price=10.2, quantity=1, time=1, agent_id=2, order_id=1, order_type=BUY)
order2 = Order(price=9, quantity=1, time=2, agent_id=2, order_id=2, order_type=BUY)
order3 = Order(price=11, quantity=1, time=3, agent_id=3, order_id=3, order_type=BUY)
order4 = Order(price=0.1, quantity=1, time=4, agent_id=4, order_id=4, order_type=BUY)
order5 = Order(price=0.3, quantity=1, time=5, agent_id=5, order_id=5, order_type=BUY)
order6 = Order(price=6, quantity=1, time=6, agent_id=6, order_id=6, order_type=BUY)

order10 = Order(price=9, quantity=1, time=7, agent_id=7, order_id=7, order_type=SELL) 
order20 = Order(price=1, quantity=1, time=8, agent_id=8, order_id=8, order_type=SELL)
order30 = Order(price=2, quantity=1, time=9, agent_id=9, order_id=9, order_type=SELL)
order40 = Order(price=10, quantity=1, time=10, agent_id=10, order_id=10, order_type=SELL)
order50 = Order(price=7, quantity=1, time=11, agent_id=11, order_id=11, order_type=SELL)
order80 = Order(price=8, quantity=1, time=12, agent_id=12, order_id=14, order_type=SELL)
order90 = Order(price=3.2, quantity=1, time=13, agent_id=13, order_id=15, order_type=SELL)
order100 = Order(price=3.84, quantity=1, time=14, agent_id=14, order_id=16, order_type=SELL)
order70 = Order(price=10.2, quantity=1, time=15, agent_id=15, order_id=13, order_type=SELL)

e.schedule_activity(order1)
e.schedule_activity(order2)
e.schedule_activity(order3)
e.schedule_activity(order4)
e.schedule_activity(order5)
e.schedule_activity(order6)
e.schedule_activity(order10)
e.schedule_activity(order20)
e.schedule_activity(order30)
e.schedule_activity(order40)
e.schedule_activity(order50)
e.schedule_activity(order70)
e.schedule_activity(order80)
e.schedule_activity(order90)
e.schedule_activity(order100)

e.step()
e.step()
e.step()
e.step()
e.step()
e.step()
e.step()
e.step()
e.step()
e.step()
e.step()
e.step()
e.step()
e.step()
e.step()

f = GaussianMeanReverting(mean=100, r=0.2, final_time=100, shock_var=10)

m = Market(f, time_steps=60000)
m.get_fundamental_value()

agent = HBLAgent(agent_id=1, market=m, q_max=100, offset=1, shade=[10,30], L=3)

m.add_order(order1)
m.add_order(order2)
m.add_order(order3)
m.add_order(order4)
m.add_order(order5)
m.add_order(order6)
m.add_order(order10)
m.add_order(order20)
m.add_order(order30)
m.add_order(order40)
m.add_order(order50)
m.add_order(order70)
m.add_order(order80)
m.add_order(order90)
m.add_order(order100)

m.step()
m.step()
m.step()
m.step()
m.step()
m.step()
m.step()
m.step()
m.step()
m.step()
m.step()
m.step()
m.step()
m.step()
m.step()
m.step()
m.step()
m.step()


prices, belief = agent.belief_interpolation(SELL)
# print(prices)
# print(belief)
# print(m.matched_orders)


#TODO: REMOVE SELF.ESTIMATE.FUNDAMENTAL() when testing this because test prices are too low for fundamental
print("OPT PRICE", agent.determine_optimal_price(SELL))

# print("___________________")
# print(m.event_queue.scheduled_activities)
# print(m.matched_orders)

# print("________________")
# print(m.order_book.buy_unmatched)
# print("________________")
# print(m.order_book.sell_unmatched)

fig, ax = plt.subplots()

ax.plot(prices, belief(prices))
plt.show()