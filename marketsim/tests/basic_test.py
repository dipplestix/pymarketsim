# import sys
# print(sys.path)
from fourheap.fourheap import FourHeap
from fourheap.order import Order
from fourheap.constants import BUY, SELL

# %%
# Let's start with order

order1 = Order(price=12, order_type=BUY, quantity=12, time=1, agent_id=1, order_id=1)
order2 = Order(price=32, order_type=SELL, quantity=22, time=1, agent_id=1, order_id=2)
order3 = Order(price=7, order_type=SELL, quantity=7, time=1, agent_id=1, order_id=3)
order4 = Order(price=9, order_type=SELL, quantity=8, time=1, agent_id=1, order_id=4)

# %%
# Now let's see initialize the fourheap

fh = FourHeap() 

# %%
# Adding an order

fh.insert(order1)
fh.buy_unmatched.order_dict

# %%
# Let's add in a sell order

fh.insert(order2)

# Because it's price is higher than the buy price this won't cause a match

fh.sell_unmatched.order_dict

# %%
# Let's add in another sell order

fh.insert(order3)

# Because it's price is lower this will match

print(fh.sell_matched.order_dict)

# Since order 1 is larger some will be matched and some will be unmatched
print(fh.buy_unmatched.order_dict)
print(fh.buy_matched.order_dict)


# %%
# We can remove an order too

fh.remove(3)

# This will unmatch the buy order entirely

print(fh.sell_matched.order_dict)
print(fh.buy_unmatched.order_dict)
print(fh.buy_matched.order_dict)


# %%
# We'll add in 3 and 4

fh.insert(order3)
fh.insert(order4)

print(fh.sell_matched.order_dict)
print(fh.sell_unmatched.order_dict)
print(fh.buy_matched.order_dict)


# %%
# Now we'll remove order 3

fh.remove(3)

print(fh.sell_matched.order_dict)
print(fh.sell_unmatched.order_dict)
print(fh.buy_matched.order_dict)
print(fh.buy_unmatched.order_dict)


# %%
# We'll do it in reverse just to see how the sell side works
fh = FourHeap()

order1 = Order(price=12, order_type=SELL, quantity=12, time=1, agent_id=1, order_id=1)
order2 = Order(price=8, order_type=BUY, quantity=22, time=1, agent_id=1, order_id=2)
order3 = Order(price=15, order_type=BUY, quantity=7, time=1, agent_id=1, order_id=3)
order4 = Order(price=24, order_type=BUY, quantity=8, time=1, agent_id=1, order_id=4)

fh.insert(order1)
fh.insert(order2)
fh.insert(order3)
fh.insert(order4)

print(fh.sell_matched.order_dict)
print(fh.sell_unmatched.order_dict)
print(fh.buy_matched.order_dict)
print(fh.buy_unmatched.order_dict)


# %%
fh.remove(4)

print(fh.sell_matched.order_dict)
print(fh.sell_unmatched.order_dict)
print(fh.buy_matched.order_dict)
print(fh.buy_unmatched.order_dict)


# %%
from fundamental.lazy_mean_reverting import LazyGaussianMeanReverting

# Let's look at the fundamental next

f = LazyGaussianMeanReverting(final_time=100, mean=12, r=.2, shock_var=.01)

# %%
# The fundamental starts at the mean

f.fundamental_values

# %%
# It's only evaluated at times it's called

f.get_value_at(12)

print(f.fundamental_values)

# %%
# When the simulation ends you find the value at the final time step

f.get_final_fundamental()

print(f.fundamental_values)

# %%
