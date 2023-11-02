from ..fourheap.constants import BUY


def vwap(matched_orders):
    num = 0
    den = 0
    for order in matched_orders:
        if order.order.order_type == BUY:
            price = order.price
            quantity = order.order.quantity
            num += price * quantity
            den += quantity

    return num/den
