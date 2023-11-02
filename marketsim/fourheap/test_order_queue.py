import pytest

from order import Order
from order_queue import OrderQueue


def test_order_queue_operations():
    # Create an OrderQueue instance
    order_queue = OrderQueue()

    # Create some Order instances
    order1 = Order(price=100.0, order_type=1, quantity=10.0, agent_id=1, time=1, order_id=1)
    order2 = Order(price=200.0, order_type=-1, quantity=5.0, agent_id=2, time=2, order_id=2)

    # Test add_order method
    order_queue.add_order(order1)
    assert order_queue.count() == 1
    assert order_queue.contains(1)

    order_queue.add_order(order2)
    assert order_queue.count() == 2
    assert order_queue.contains(2)

    # Test peek and peek_key methods
    assert order_queue.peek() == 100.0
    assert order_queue.peek_order() == order1

    # Test remove method
    order_queue.remove(1)
    assert not order_queue.contains(1)
    assert order_queue.count() == 1

    # Test push_to method
    popped_order = order_queue.push_to()
    assert popped_order == order2
    assert order_queue.count() == 0
    assert not order_queue.contains(2)

    # Test is_empty method
    assert order_queue.is_empty()

    # Test clear method
    order_queue.add_order(order1)
    order_queue.clear()
    assert order_queue.is_empty()
    assert order_queue.count() == 0


def test_order_comparisons():
    # Create some Order instances
    buy_order1 = Order(price=200.0, order_type=1, quantity=20.0, agent_id=3, time=1, order_id=3)
    buy_order2 = Order(price=200.0, order_type=1, quantity=15.0, agent_id=4, time=2, order_id=4)
    sell_order1 = Order(price=100.0, order_type=-1, quantity=10.0, agent_id=1, time=1, order_id=1)
    sell_order2 = Order(price=100.0, order_type=-1, quantity=5.0, agent_id=2, time=2, order_id=2)

    # Test __gt__ method for buy orders
    assert buy_order1 > buy_order2
    assert not (buy_order2 > buy_order1)

    # Test __gt__ method for sell orders
    assert sell_order1 > sell_order2
    assert not (sell_order2 > sell_order1)


# if you want to run this file directly, you can use the following command
if __name__ == "__main__":
    pytest.main()
