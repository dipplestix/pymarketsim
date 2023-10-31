from dataclasses import dataclass


@dataclass
class Order:
    price: float
    order_type: int  # -1 for a sell order, +1 for a buy order
    quantity: float
    agent_id: int
    time: int
    order_id: int
    asset_id: int = 1

    def update_quantity_filled(self, transact_quantity: float) -> None:
        self.quantity -= transact_quantity

    def copy_and_decrease(self, transact_quantity: float) -> 'Order':
        new_order = Order(self.price,
                          self.order_type,
                          self.quantity - transact_quantity,
                          self.agent_id,
                          self.time,
                          self.order_id + 1,
                          self.asset_id
                          )
        self.update_quantity_filled(self.quantity - transact_quantity)
        return new_order

    def __eq__(self, other: 'Order') -> bool:
        return self.order_id == other.order_id

    def __gt__(self, other: 'Order') -> bool:
        if self.order_type == -1 and other.order_type == -1:
            return (self.price, self.time) < (other.price, other.time)
        elif self.order_type == 1 and other.order_type == 1:
            return (self.price, -self.time) > (other.price, -other.time)
        # I don't think these are needed
        elif self.order_type == -1:
            return self.price < other.price
        elif self.order_type == 1:
            return self.price > other.price


@dataclass
class MatchedOrder:
    price: float
    quantity: float
    buy_order: Order
    sell_order: Order
