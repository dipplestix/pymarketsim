import constants
from fourheap import FourHeap
from order import Order

def get_vwap(matched_orders):
    num = 0
    den = 0
    for order in matched_orders:
        if order.order.order_type == constants.BUY:
            price = order.price
            quantity = order.order.quantity
            num += price*quantity
            den += quantity
            
    return num/den

fh = FourHeap()

oid = 0
order =  Order(price=102, quantity=1, order_id=oid, agent_id=1, order_type=constants.SELL, time=1)
fh.insert(order)
oid += 1
order =  Order(price=101, quantity=1, order_id=oid, agent_id=1, order_type=constants.SELL, time=1)
fh.insert(order)
oid += 1
order =  Order(price=97, quantity=1, order_id=oid, agent_id=1, order_type=constants.BUY, time=1)
fh.insert(order)
oid += 1
order =  Order(price=98, quantity=1, order_id=oid, agent_id=1, order_type=constants.BUY, time=1)
fh.insert(order)
oid += 1
order =  Order(price=99, quantity=1, order_id=oid, agent_id=1, order_type=constants.BUY, time=1)
fh.insert(order)
out = fh.market_clear()
oid += 1
order =  Order(price=101, quantity=1, order_id=oid, agent_id=1, order_type=constants.BUY, time=2)
fh.insert(order)
out += fh.market_clear()
oid += 1
order =  Order(price=99, quantity=1, order_id=oid, agent_id=1, order_type=constants.SELL, time=4)
fh.insert(order)
out += fh.market_clear()
oid += 1
order =  Order(price=101, quantity=1, order_id=oid, agent_id=1, order_type=constants.SELL, time=5)
fh.insert(order)
out += fh.market_clear()
oid += 1
order =  Order(price=101, quantity=1, order_id=oid, agent_id=1, order_type=constants.BUY, time=6)
fh.insert(order)
out += fh.market_clear()
oid += 1



get_vwap(out)

order =  Order(price=99, quantity=1, order_id=oid, agent_id=1, order_type=constants.SELL, time=7)
fh.insert(order)
out += fh.market_clear()
oid += 1
order =  Order(price=98, quantity=1, order_id=oid, agent_id=1, order_type=constants.SELL, time=8)
fh.insert(order)
out += fh.market_clear()
oid += 1
order =  Order(price=97, quantity=1, order_id=oid, agent_id=1, order_type=constants.SELL, time=9)
fh.insert(order)
out += fh.market_clear()
oid += 1

get_vwap(out)

a = [1, 2, 3, 4]

import random

random.shuffle(a)
a

random.Random(None)



