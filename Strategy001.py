
# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from collections import deque

class Strategy0011(IStrategy):
    """
    Strategy 001
    author@: Gerald Lonlas
    github@: https://github.com/freqtrade/freqtrade-strategies

    How to use it?
    > python3 ./freqtrade/main.py -s Strategy001
    """
    P=14;
    order=5; 
    K=2;
    INTERFACE_VERSION: int = 3
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "60":  0.01,
        "30":  0.03,
        "20":  0.04,
        "0":  0.05
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.10

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    # run "populate_indicators" only for new candle
    process_only_new_candles = True

    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    # Optional order type mapping
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    


    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return [("ETH/USDT", "5m"), ("BTC/TUSD", "15m", "spot"), ("BTC/TUSD", "4h", "spot")]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        #dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        #dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        #dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        #heikinashi = qtpylib.heikinashi(dataframe)
        #dataframe['ha_open'] = heikinashi['open']
        #dataframe['ha_close'] = heikinashi['close']
           # Print the Analyzed pair
        
        self.getPeaks(dataframe, key='close', order=self.order, K=self.K)
        self.calcRSI(dataframe, P=self.P)
        self.getPeaks(dataframe, key='RSI', order=self.order, K=self.K)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        position = np.zeros(dataframe.shape[0])
        for i, (t, row) in enumerate(dataframe.iterrows()):
            if np.isnan(row['RSI']):
                continue
            # If no position is on
            if position[i-1] == 0:
            # Buy if indicator to higher low and price to lower low
                if row['close_lows'] == -1 and row['RSI_lows'] == 1:
                    if row['RSI'] < 50:
                        position[i] = 1
                        dataframe.loc[i,'enter_long'] = 1
                        #entry_rsi = row['RSI'].copy()
                        entry_rsi = row['RSI']

            # If current position is long
            elif position[i-1] == 1:
                if row['RSI'] < 50 and row['RSI'] < entry_rsi:
                    position[i] = 1
                    dataframe.loc[i, 'exit_long'] = 1


        #dataframe['position'] = position
        
        #print(f"result for {metadata['pair']}")

        # Inspect the last 5 rows
        #print(dataframe.tail())

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        #dataframe.loc[
        #    (
        #        qtpylib.crossed_above(dataframe['ema50'], dataframe['ema100']) &
        #        (dataframe['ha_close'] < dataframe['ema20']) &
        #        (dataframe['ha_open'] > dataframe['ha_close'])  # red bar
        #    ),
        #    'exit_long'] = 1
        return dataframe
    
    def getHigherLows(self, data: np.array, order=5, K=2):
        '''
        Finds consecutive higher lows in price pattern.
        Must not be exceeded within the number of periods indicated by the width 
        parameter for the value to be confirmed.
        K determines how many consecutive lows need to be higher.
        '''
        # Get lows
        low_idx = argrelextrema(data, np.less, order=order)[0]
        lows = data[low_idx]
        # Ensure consecutive lows are higher than previous lows
        extrema = []
        ex_deque = deque(maxlen=K)
        for i, idx in enumerate(low_idx):
            if i == 0:
                ex_deque.append(idx)
                continue
            if lows[i] < lows[i-1]:
                ex_deque.clear()

            ex_deque.append(idx)
            if len(ex_deque) == K:
                extrema.append(ex_deque.copy())

        return extrema

    def getLowerHighs(self, data: np.array, order=5, K=2):
        '''
        Finds consecutive lower highs in price pattern.
        Must not be exceeded within the number of periods indicated by the width 
        parameter for the value to be confirmed.
        K determines how many consecutive highs need to be lower.
        '''
        # Get highs
        high_idx = argrelextrema(data, np.greater, order=order)[0]
        highs = data[high_idx]
        # Ensure consecutive highs are lower than previous highs
        extrema = []
        ex_deque = deque(maxlen=K)
        for i, idx in enumerate(high_idx):
            if i == 0:
                ex_deque.append(idx)
                continue
            if highs[i] > highs[i-1]:
                ex_deque.clear()

            ex_deque.append(idx)
            if len(ex_deque) == K:
                extrema.append(ex_deque.copy())

        return extrema

    def getHigherHighs(self, data: np.array, order=5, K=2):
        '''
        Finds consecutive higher highs in price pattern.
        Must not be exceeded within the number of periods indicated by the width 
        parameter for the value to be confirmed.
        K determines how many consecutive highs need to be higher.
        '''
        # Get highs
        high_idx = argrelextrema(data, np.greater, order=5)[0]
        highs = data[high_idx]
        # Ensure consecutive highs are higher than previous highs
        extrema = []
        ex_deque = deque(maxlen=K)
        for i, idx in enumerate(high_idx):
            if i == 0:
                ex_deque.append(idx)
                continue
            if highs[i] < highs[i-1]:
                ex_deque.clear()

            ex_deque.append(idx)
            if len(ex_deque) == K:
                extrema.append(ex_deque.copy())

        return extrema

    def getLowerLows(self, data: np.array, order=5, K=2):
        '''
        Finds consecutive lower lows in price pattern.
        Must not be exceeded within the number of periods indicated by the width 
        parameter for the value to be confirmed.
        K determines how many consecutive lows need to be lower.
        '''
        # Get lows
        low_idx = argrelextrema(data, np.less, order=order)[0]
        lows = data[low_idx]
        # Ensure consecutive lows are lower than previous lows
        extrema = []
        ex_deque = deque(maxlen=K)
        for i, idx in enumerate(low_idx):
            if i == 0:
                ex_deque.append(idx)
                continue
            if lows[i] > lows[i-1]:
                ex_deque.clear()

            ex_deque.append(idx)
            if len(ex_deque) == K:
                extrema.append(ex_deque.copy())

        return extrema
    def getHHIndex(self, data: np.array, order=5, K=2):
        extrema = self.getHigherHighs(data, order, K)
        idx = np.array([i[-1] + order for i in extrema])
        return idx[np.where(idx<len(data))]

    def getLHIndex(self, data: np.array, order=5, K=2):
        extrema = self.getLowerHighs(data, order, K)
        idx = np.array([i[-1] + order for i in extrema])
        return idx[np.where(idx<len(data))]

    def getLLIndex(self, data: np.array, order=5, K=2):
        extrema = self.getLowerLows(data, order, K)
        idx = np.array([i[-1] + order for i in extrema])
        return idx[np.where(idx<len(data))]

    def getHLIndex(self, data: np.array, order=5, K=2):
        extrema = self.getHigherLows(data, order, K)
        idx = np.array([i[-1] + order for i in extrema])
        return idx[np.where(idx<len(data))]

    def calcRSI(self, data, P=14):
        data['diff_close'] = data['close'] - data['close'].shift(1)
        data['gain'] = np.where(data['diff_close']>0, data['diff_close'], 0)
        data['loss'] = np.where(data['diff_close']<0, np.abs(data['diff_close']), 0)
        data[['init_avg_gain', 'init_avg_loss']] = data[
            ['gain', 'loss']].rolling(P).mean()
        avg_gain = np.zeros(len(data))
        avg_loss = np.zeros(len(data))
        for i, _row in enumerate(data.iterrows()):
            row = _row[1]
            if i < P - 1:
                last_row = row.copy()
                continue
            elif i == P-1:
                avg_gain[i] += row['init_avg_gain']
                avg_loss[i] += row['init_avg_loss']
            else:
                avg_gain[i] += ((P - 1) * avg_gain[i-1] + row['gain']) / P
                avg_loss[i] += ((P - 1) * avg_loss[i-1] + row['loss']) / P
                
            last_row = row.copy()
            
        data['avg_gain'] = avg_gain
        data['avg_loss'] = avg_loss
        data['RS'] = data['avg_gain'] / data['avg_loss']
        data['RSI'] = 100 - 100 / (1 + data['RS'])
        return data
    
    def getPeaks(self, data, key='close', order=5, K=2):
        vals = data[key].values
        hh_idx = self.getHHIndex(vals, order, K)
        lh_idx = self.getLHIndex(vals, order, K)
        ll_idx = self.getLLIndex(vals, order, K)
        hl_idx = self.getHLIndex(vals, order, K)

        data[f'{key}_highs'] = np.nan
        data[f'{key}_highs'][hh_idx] = 1
        data[f'{key}_highs'][lh_idx] = -1
        data[f'{key}_highs'] = data[f'{key}_highs'].ffill().fillna(0)
        data[f'{key}_lows'] = np.nan
        data[f'{key}_lows'][ll_idx] = 1
        data[f'{key}_lows'][hl_idx] = -1
        data[f'{key}_lows'] = data[f'{key}_highs'].ffill().fillna(0)
        return data