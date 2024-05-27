# This file includes metrics that can be used for learning in markets.

import numpy as np

def midprice_move(market, lookback=20):
    midprices = market.get_midprices()
    if len(midprices) >= lookback:
        return midprices[-1] - np.mean(midprices[-lookback:-1])
    elif len(midprices) >= 2:
        return midprices[-1] - np.mean(midprices[:-1])
    else:
        return 0.0

def volume_imbalance(market):
    # The ratio of the difference between buy and sell volumes to their sum.
    order_book = market.order_book
    sell_book_size = order_book.sell_unmatched.count()
    buy_book_size = order_book.buy_unmatched.count()
    if sell_book_size + buy_book_size == 0:
        return 0.0
    imbalance = (sell_book_size - buy_book_size) / (sell_book_size + buy_book_size)
    return imbalance


def queue_imbalance(market):
    # The ratio of the difference between the number of buy and sell orders to their sum.
    order_book = market.order_book
    sell_book_size = len(order_book.sell_unmatched.order_dict)
    buy_book_size = len(order_book.buy_unmatched.order_dict)
    if sell_book_size + buy_book_size == 0:
        return 0.0
    imbalance = (sell_book_size - buy_book_size) / (sell_book_size + buy_book_size)
    return imbalance


def signed_volume(market):
    """
    Traded volume in a limit order book is either buyer or seller initiated;
    for example, if a resting ask is matched by an incoming buy (market or limit order)
    then the resulting traded volume is buyer initiated. Signed volume just associated a + with buyer-initiated volume
    and a - with seller initiated volume.
    """
    return market.get_signed_volume()

#TODO: Can think about tuning the lookback amount
def realized_volatility(market, lookback=20):
    """
    (RV) is an assessment of variation for assets by analyzing its historical returns within a defined period, it can be calculated by:
    sqrt(\sum_{i=1}^n (log(pt) - log(pt-1))^2)
    """
    midprices = market.get_midprices()
    if len(midprices) >= lookback:
        prices = np.array(midprices)[-lookback:]
    elif len(midprices) <= 1:
        return 0.0
    else:
        prices = np.array(midprices)

    price_ratio = np.log(prices[1:] / prices[:-1]) ** 2
    rv = np.sqrt(np.sum(price_ratio))

    return rv


def relative_strength_index(market, lookback=20):
    """
    (RSI) is a technical indicator used in momentum trading that measures
    the speed of a security’s recent price changes to evaluate overvalued or undervalued
    conditions in the price of that security. (Rescaled by 100)
    """
    midprices = market.get_midprices()
    if len(midprices) >= lookback:
        prices = np.array(midprices)[-lookback:]
        deltas = np.diff(prices)
        up = deltas[deltas >= 0].sum() / lookback
        down = -deltas[deltas < 0].sum() / lookback
    else:
        prices = np.array(midprices)
        deltas = np.diff(prices)
        up = deltas[deltas >= 0].sum() / len(midprices)
        down = -deltas[deltas < 0].sum() / len(midprices)

    if down == 0:
        return 100

    rs = up / down
    rsi = 100. - 100. / (1. + rs)

    return rsi


