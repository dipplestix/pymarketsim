import pytest

from order import Order  # Assuming order.py is the name of the file containing your Order class


def test_creation():
    order = Order(price=100.0, order_type=1, quantity=10.0, agent_id=1, time=1234567890, order_id=1)
    assert order.price == 100.0
    assert order.order_type == 1
    assert order.quantity == 10.0
    assert order.agent_id == 1
    assert order.time == 1234567890
    assert order.order_id == 1


def test_update_quantity_filled():
    order = Order(price=100.0, order_type=1, quantity=10.0, agent_id=1, time=1234567890, order_id=1)
    order.update_quantity_filled(3.0)
    assert order.quantity == 7.0


def test_equality():
    order1 = Order(price=100.0, order_type=1, quantity=10.0, agent_id=1, time=1234567890, order_id=1)
    order2 = Order(price=200.0, order_type=-1, quantity=5.0, agent_id=2, time=1234567999, order_id=2)
    order3 = Order(price=150.0, order_type=-1, quantity=8.0, agent_id=3, time=1234568000, order_id=1)
    assert not (order1 == order2)
    assert order1 == order3


def test_comparators():
    sell_order1 = Order(price=100.0, order_type=-1, quantity=10.0, agent_id=1, time=1, order_id=1)
    sell_order2 = Order(price=100.0, order_type=-1, quantity=5.0, agent_id=2, time=2, order_id=2)
    sell_order3 = Order(price=100.0, order_type=-1, quantity=10.0, agent_id=1, time=1, order_id=1)
    sell_order4 = Order(price=200.0, order_type=-1, quantity=5.0, agent_id=2, time=1, order_id=2)

    buy_order1 = Order(price=200.0, order_type=1, quantity=20.0, agent_id=3, time=1, order_id=3)
    buy_order2 = Order(price=200.0, order_type=1, quantity=15.0, agent_id=4, time=2, order_id=4)
    buy_order3 = Order(price=200.0, order_type=1, quantity=20.0, agent_id=3, time=1, order_id=3)
    buy_order4 = Order(price=300.0, order_type=1, quantity=15.0, agent_id=4, time=1, order_id=4)

    assert sell_order1 > sell_order2
    assert sell_order3 > sell_order4

    assert buy_order1 > buy_order2
    assert buy_order4 > buy_order3

    assert buy_order4 > sell_order4


# if you want to run this file directly, you can use the following command
if __name__ == "__main__":
    pytest.main()
