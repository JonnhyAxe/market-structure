
# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce
from pandas import DataFrame
# --------------------------------

import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from collections import deque

from scipy.signal import argrelextrema
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import sys
from freqtrade.persistence import Trade
from freqtrade.persistence import Order

import logging

from scipy.signal import argrelextrema

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


logger = logging.getLogger('freqtrade')

class Strategy001(IStrategy):
    """
    Strategy 001
    author@: Gerald Lonlas
    github@: https://github.com/freqtrade/freqtrade-strategies

    How to use it?
    > python3 ./freqtrade/main.py -s Strategy001
    """
    
    lastLHIdx = -1
    lastLHPrice = sys.float_info.max
    lastLLIdx = -1
    lastLLPrice = sys.float_info.max

    lastHHIdx = 0
    lastHHPrice = 0
    lastHLIdx = 0
    lastHLPrice = 0

    marketPrice = 0
    half_level = 0
    upTrend = False
    downTrend = False
    breakOfStructureDown = False
    breakOfStructureUP = False
    trendChange = False
    lastStrongLevel = 0
    lastStrongLevelIdx = np.nan
    lastStrongLevelDataFrame = np.nan
    sig_dir = 'EXIT'
        
    P: int = 14;
    order: int = 5; 
    K: int = 2;
    INTERFACE_VERSION: int = 3
    long_window: int = 100
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        #"640":  0.01,
        #"30":  0.03,
        #"20":  0.04,
        "0":  1000
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    #
    # stoploss = -0.10
    stoploss = -1000


    # Optimal timeframe for the strategy
    timeframe = '4h'
    #inf_tf ='15m'


    # trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    #startup_candle_count = 100
    position_adjustment_enable = True

    # Optional order type mapping
    order_types = {
        'entry': 'market',
        'exit': 'market',
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
        
        #self.getPeaks(dataframe, key='close', order=self.order, K=self.K)
        #self.calcRSI(dataframe, P=self.P)
        #self.getPeaks(dataframe, key='RSI', order=self.order, K=self.K)
        self.getPeaks(dataframe)
        return dataframe
    
    #@informative('15m')
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        try:
            position = np.zeros(dataframe.shape[0])
            
            for i, (t, row) in enumerate(dataframe.iterrows()):
                
                data = dataframe.head(i)
                if i > self.long_window:
                    data = data.tail(self.long_window) 

                if data is not None and data.size > 2:

                    #filter pnly new values when there is a new Strong level
                    data = data[self.lastStrongLevelIdx:] if not np.isnan(self.lastStrongLevelIdx) else data


                    data["valuesHH"] = np.where(data.close > data.high, data['close'], data['high'])
                    price_data = data['valuesHH'].values
                    hh_idx = self.getHHIndex(price_data, self.order, self.K)
                    hh_idx = self.filterHHIndex(data, hh_idx)
                    # hh_idx = argrelextrema(price_data, np.greater, order=order)[0]
                    if (self.downTrend):
                        hh_idx = self.filterUptrendInternalStructureInDowntrend(data, hh_idx, self.lastLHPrice)  # filter all LL before last HH index (INTERNAL STRUCTURE)

                    hh = data.iloc[hh_idx - self.order]['valuesHH']  # SUBSTRACT ORDER

                    data["valuesLH"] = np.where(data.close < data.low, data['close'],  data['low'])
                    # data["valuesHL"] = np.where(data.close > data.high, data['close'], data['close'])
                    # price_data = data['valuesLH'].values
                    price_data = data['close'].values
                    hl_idx = self.getLHIndex(price_data, 1, 2)
                    # hl_idx = argrelextrema(price_data, np.less, order=order)[0]
                    hl = data.iloc[hl_idx - 1]['valuesLH']  # SUBSTRACT ORDER

                    self.marketPrice = price_data[-1]
                    if len(hh) >= 2 and hh.iloc[-1] > hh.iloc[-2] and hh_idx[-1] != self.lastHHIdx and  hh.iloc[-1] > self.lastHHPrice:  # only new HH

                        self.upTrend = True
                        # self.downTrend = False
                        self.lastHHIdx = hh_idx[-1] - self.order
                        self.lastHHPrice = data.iloc[self.lastHHIdx]['valuesHH']
                        self.lastHLIdx = self.getLastHLBeforeIndex(hl_idx, self.lastHHIdx, self.lastHHPrice, data['close'], self.order-1)
                        # self.lastHLPrice = bars[self.lastHLIdx-order] # use data instead
                        self.lastHLPrice = data.iloc[self.lastHLIdx-1]['valuesLH'] #
                        self.lastStrongLevel = self.lastHLPrice
                        self.lastStrongLevelIdx = self.lastHLIdx
                        self.half_level = self.lastHHPrice - (self.lastHHPrice - self.lastHLPrice) * 0.5

                        # reset downtrend
                        # self.lastLLIdx = -1
                        #self.lastLLPrice =  self.lastHHPrice

                        print("Uprange [" + str(self.lastHLPrice) + "," + str(self.lastHHPrice) + "] halfprice:" + str(self.half_level))
                        #if self.sig_dir == 'EXIT':
                        dataframe.loc[i,  ['enter_long', 'enter_tag']] =  (1, 'HH')
                        print("Long: [" + str(dataframe.loc[i]["date"]) + "], price: " + str(self.marketPrice) )
                        self.sig_dir = 'LONG'

                        #self.displayCandlesUpTrend(data, hh, hl, self.half_level)

                        if self.downTrend and self.lastHHPrice < self.lastHLPrice:  ## (internal structure to ignore)
                            self.upTrend = False
                            self.downTrend = True
                            #self.events.put(signal)
                            #self.bought[symbol] = 'LONG'

                    if self.upTrend and self.marketPrice > self.lastHHPrice:  # current market is above last HH and No new HH
                        self.breakOfStructureDown = False
                        self.breakOfStructureUP = True
                        # BOS to the Upside

                    if self.upTrend and self.marketPrice < self.half_level:  # current market is above last HH and No new HH
                        # Signal to buy
                        # self.displayCandlesUpTrend(data, hh, hl, self.half_level)
                        self.breakOfStructureDown = False
                        # override last LL Level as HL of uptrend
                        #if self.sig_dir == 'EXIT':
                        #    self.sig_dir = 'LONG'
                            #signal = SignalEvent(strategy_id, symbol, dt, self.sig_dir, strength)
                        # dataframe.loc[i,'enter_long'] = 1
                        #    position[i] = 1

                    if self.upTrend and self.lastHLPrice is not None and self.marketPrice < self.lastHLPrice:  # below last HL
                        self.upTrend = False
                        self.downTrend = False
                        self.breakOfStructureUP = False
                        self.breakOfStructureDown = True  # liquidity grab if downtrend is not formed

                        #self.half_level = self.lastHHPrice - (self.lastHHPrice - self.lastHLPrice) * 0.5
                        #self.displayCandlesUpTrend(data, hh, hl, self.half_level)
                        #if self.sig_dir == 'LONG':
                        dataframe.loc[i,  ['exit_long', 'exit_tag']] =  (1, 'BOS')
                        print("LongExit: [" + str(dataframe.loc[i]["date"]) + "], price: " + str(self.marketPrice) )
                        #self.sig_dir = 'EXIT'

                        #if self.sig_dir == 'LONG':
                        # if position[i-1] == 1:
                        #    self.sig_dir = 'EXIT'
                            #signal = SignalEvent(strategy_id, symbol, dt, self.sig_dir, strength)
                            #self.events.put(signal)
                        #    position[i] = 1
                        #    dataframe.loc[i, 'exit_long'] = 1

                    # if self.upTrend and self.lastLHPrice is not None and self.marketPrice < self.half_level:  # below last HL
                        # self.displayCandlesUpTrend(data, hh, hl, self.half_level)

                    #try tail from lastStrongLevel

                    data["valuesLL"] = np.where(data.close < data.low, data['close'], data['low'])
                    price_data = data['valuesLL'].values
                    # price_data = data['close'].values
                    ll_idx = self.getLLIndex(price_data, 1, self.K) #TODO: why order here is 1
                    ll_idx = self.filterLLIndex(data, ll_idx) # replace with lastStrongLevel ???
                    if(self.upTrend):
                        #self.displayCandlesUpTrend(data, hh, hl, self.half_level)
                        ll_idx = self.filterDowntrendInternalStructureInUptrend(data, ll_idx,  self.lastHLPrice) # filter all LL before last HH index (INTERNAL STRUCTURE)

                    ll = data.iloc[ll_idx - 1]['valuesLL']

                    data["valuesHL"] = np.where(data.close > data.high, data['close'], data['high'])
                    price_data = data['valuesHL'].values
                    lh_idx = self.getLHIndex(price_data, 1, 1)
                    lh = data.iloc[lh_idx - 1]['valuesHL']  # SUBSTRACT ORDER
                    # if len(ll) >= 2 and ll[-1] < ll[-2] and ll_idx[-1] != self.lastLLIdx and ll[-1] < self.lastLLPrice and len(lh_idx) >= 1 :  # only new LL
                    if len(ll) >= 2 and ll.iloc[-1] < ll.iloc[-2] and ll_idx[-1] != self.lastLLIdx and len(lh_idx) >= 1 and ll.iloc[-1] < self.lastLLPrice:  # only new LL
                        self.downTrend = True

                        self.lastLLIdx = ll_idx[-1]
                        self.lastLLPrice = data.iloc[self.lastLLIdx - 1]['valuesLL']

                        self.lastLHIdx = self.getLastLHBeforeIndex(lh_idx, self.lastLLIdx, self.lastLLPrice, data['close'], self.order - 1)
                        #self.lastLHPrice = data.iloc[self.lastLHIdx - 1]['valuesHL']
                        self.lastLHPrice = data.iloc[self.lastLHIdx-1]['valuesHL'] # why -order again?
                        #self.lastLHPrice = data.iloc[self.lastLHIdx]['close']
                        #self.lastLHPrice = bars[self.lastLHIdx]
                        self.lastStrongLevel = self.lastLHPrice
                        self.lastStrongLevelIdx = self.lastLHIdx

                        print("Downrange [" + str(self.lastLHPrice) + "," + str(self.lastLLPrice) + "]")

                        self.half_level = self.lastLHPrice - (self.lastLHPrice - self.lastLLPrice) * 0.5
                        #self.displayCandlesDownTrend(data, ll, lh, self.half_level)


                        # reset uptrendtrend
                        # self.lastHHIdx = -1  this breaks last update
                        # self.lastHHPrice = self.lastLLPrice

                    if self.upTrend and self.lastLLPrice > self.lastLHPrice:  # internal Down structure in Uptrend (ignore)
                        self.downTrend = False
                        self.upTrend = True
                        self.breakOfStructureDown = True
                        self.breakOfStructureUP = False

                    if self.downTrend and self.marketPrice < self.lastLLPrice:  # current market is below last LL (Sbreak to downside)
                        self.breakOfStructureDown = True
                        self.breakOfStructureUP = False

                    if self.downTrend and self.lastLLPrice is not None and self.marketPrice > self.lastLHPrice:  # above last HL
                        self.trendChange = True
                        self.breakOfStructureUP = False
                        self.breakOfStructureDown = False
                        self.upTrend = False
                        self.downTrend = False
                        #self.displayCandlesDownTrend(data, ll, lh, self.half_level)
                        #if position[i-1] == 1:
                           # dataframe.loc[i,'exit_short'] = 1
                           # position[i] = 1

                    if self.downTrend and self.marketPrice > self.half_level:  # current market is above last HH and No new HH
                        # Signal to short (push phase of an dowtrend)
                        # self.displayCandlesDownTrend(data, ll, lh, self.half_level)
                        self.breakOfStructureUP = False
                        #self.displayCandlesDownTrend(data, ll, lh, self.half_level)
                        #if position[i-1] == 0:
                           # dataframe.loc[i,'enter_short'] = 1
                           # position[i] = 1



                #dataframe['position'] = position
                
                #print(f"result for {metadata['pair']}")

                # Inspect the last 5 rows
                # print(dataframe.tail()) print(dataframe)  dataframe['enter_long'].tolist() dataframe['exit_long'].tolist() 
                # print(dataframe.columns)
            
        except Exception as e:
            logger.error(str(e))

            print(e.__cause__)

        print(f"Generated {dataframe['enter_long'].sum()} entry signals")
        print(f"Generated {dataframe['exit_long'].sum()} exit signals")

        return dataframe
    
    def getPeaks(self, data, key='Close', order=5, K=2):
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

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: any, **kwargs) -> None:
        # Obtain pair dataframe (just to show how to access it)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        return None
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: any, entry_tag: any,
                            side: str, **kwargs) -> bool:
        
        print(f"Trade Entry {current_time}, amount {amount}, side: {side}")
        
        return True 
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: any, **kwargs) -> bool:
       
        print(f"Trade Exit {current_time}, reason: {exit_reason}")

    
        return True
    
    
    def adjust_trade_position(self, trade: Trade, current_time: any,
                            current_rate: float, current_profit: float,
                            min_stake: Optional[float], max_stake: float,
                            current_entry_rate: float, current_exit_rate: float,
                            current_entry_profit: float, current_exit_profit: float,
                            **kwargs
                            ) -> Union[Optional[float], Tuple[Optional[float], Optional[str]]]:
        
        # Obtain pair dataframe (just to show how to access it)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # Only buy when not actively falling price.
        ##last_candle = dataframe.iloc[-1].squeeze()
        ##previous_candle = dataframe.iloc[-2].squeeze()
        #if last_candle['close'] < previous_candle['close']:
        #    return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        # Allow up to 3 additional increasingly larger buys (4 in total)
        # Initial buy is 1x
        # If that falls to -5% profit, we buy 1.25x more, average profit should increase to roughly -2.2%
        # If that falls down to -5% again, we buy 1.5x more
        # If that falls once again down to -5%, we buy 1.75x more
        # Total stake for this trade would be 1 + 1.25 + 1.5 + 1.75 = 5.5x of the initial allowed stake.
        # That is why max_dca_multiplier is 5.5
        # Hope you have a deep wallet!
        try:
            # This returns first order stake size
            stake_amount = filled_entries[0].stake_amount
            # This then calculates current safety order size
            stake_amount = stake_amount * (1 + (count_of_entries * 0.25))
            return stake_amount, '1/3rd_increase'
        except Exception as exception:
            return None
        
        return None
     
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
    
    def filterLLIndex(self, data: np.array, ll_idx):
        index = 0
        ll = data.iloc[ll_idx - 1]['valuesLL']
        datetime = ll.index
        for i, idx in enumerate(ll.copy()):
            if i == 0:
                continue
            if ll.iloc[i-index] > ll.iloc[i-index - 1]: #
                ll = ll.drop(datetime[i])
                ll_idx = np.delete(ll_idx, i-index)
                index += 1
        return ll_idx

    def filterDowntrendInternalStructureInUptrend(self, data: np.array, ll_idx, lastHLPrice):
        index = 0
        ll = data.iloc[ll_idx - 1]['valuesLL']
        datetime = ll.index
        #lastStrongLevelDataFrame = data.iloc[np.array((self.lastStrongLevelIdx,), dtype=np.int64) - 1]
        #lastStrongLevelDataFrameDate = lastStrongLevelDataFrame.index

        for i, idx in enumerate(ll.copy()):
            if ll.iloc[i-index] > lastHLPrice: #if internal structure drop indexes
                ll = ll.drop(datetime[i])
                ll_idx = np.delete(ll_idx, i-index)
                index += 1
        return ll_idx

    def filterUptrendInternalStructureInDowntrend(self, data: np.array, ll_idx,  lastHLPrice):
        index = 0
        ll = data.iloc[ll_idx - 1]['valuesHH']
        datetime = ll.index
        #lastStrongLevelDataFrame = data.iloc[np.array((self.lastStrongLevelIdx,), dtype=np.int64) - 1]
        #lastStrongLevelDataFrameDate = lastStrongLevelDataFrame.index

        for i, idx in enumerate(ll.copy()):
            if ll.iloc[i-index] < lastHLPrice: #if internal structure drop indexes
                ll = ll.drop(datetime[i])
                ll_idx = np.delete(ll_idx, i-index)
                index += 1
        return ll_idx

    def filterHHIndex(self, data: np.array, hh_idx):
        index = 0
        hh = data.iloc[hh_idx - 1]['valuesHH']
        datetime = hh.index
        for i, idx in enumerate(hh.copy()):
            if i == 0:
                continue
            if hh.iloc[i-index] < hh.iloc[i-index - 1]:
                hh = hh.drop(datetime[i])
                hh_idx = np.delete(hh_idx, i-index)
                index += 1
        return hh_idx
    
    def getLastHLBeforeIndex(self, lHIndexes, hhIndex, lastHHPrice, bars, order):
        """
        Finds the first (closest) index of the Higher Low before the given Higher High

        Parameters
        lHIndexes - lower High indexes.
        hhIndex - the last Higher High to search for the previous Lower High
        """

        for hLIndex in lHIndexes[::-1]:
            if hLIndex < hhIndex and bars.iloc[hLIndex-order] < lastHHPrice:
                return hLIndex

    def getLastLHBeforeIndex(self, lHIndexes, llIndex, lastLLPrice, bars, order):
        """
        Finds the first (closest) index of the Higher Low before the given Higher High

        Parameters
        lHIndexes - lower High indexes.
        hhIndex - the last Higher High to search for the previous Lower High
        """

        for hLIndex in lHIndexes[::-1]:
            if hLIndex < llIndex and bars.iloc[hLIndex-order] > lastLLPrice:
                return hLIndex

    def display(self, data, hh_idx, lh_idx, order):

        plt.figure(figsize=(15, 8))
        plt.plot(data['close'])
        close = data['close'].values
        date = data.index
        # _ = [plt.plot(date[i-order], close[i-order], c=colors[1]) for i in hh_idx]
        # _ = [plt.plot(date[i-order], close[i-order], c=colors[2]) for i in lh_idx]
        plt.scatter(date[hh_idx - order], close[hh_idx - order], c=colors[1], marker='^', s=100)
        plt.scatter(date[lh_idx - order], close[lh_idx - order], c=colors[2], marker='^', s=100)

        plt.show()
        plt.close()

    def displayCandles(self, data, hh_idx, lh_idx, order):
        trace1 = go.Candlestick(x=data.index,
                                open=data['open'],
                                high=data['high'],
                                low=data['low'],
                                close=data['close'])

        trace2 = go.Scatter(x=data.close[hh_idx - order].index, y=data.close[hh_idx - order])
        trace3 = go.Scatter(x=data.close[lh_idx - order].index, y=data.close[lh_idx - order])

        # trace3 = go.Scatter(x=support_indices, y=price_data[support_indices], fill='tonexty', fillcolor='white', opacity=0.1, line=dict(width=0))
        layout = go.Layout(xaxis_rangeslider_visible=False)
        dataLayers = [trace1, trace2, trace3]
        figure = go.Figure(data=dataLayers, layout=layout)

        figure.show()
        # figure.close()

    def displayCandlesUpTrend(self, data, hh_idx, lh_idx, fib):
        trace1 = go.Candlestick(x=data.index,
                                open=data['open'],
                                high=data['high'],
                                low=data['low'],
                                close=data['close'])

        hh = go.Scatter(x=hh_idx.index, y=hh_idx, line=dict(color="#FF0000"))
        lh = go.Scatter(x=lh_idx.index, y=lh_idx, line=dict(color=colors[2]))

        # trace3 = go.Scatter(x=support_indices, y=price_data[support_indices], fill='tonexty', fillcolor='white', opacity=0.1, line=dict(width=0))
        layout = go.Layout(xaxis_rangeslider_visible=False)
        dataLayers = [trace1, hh, lh]
        figure = go.Figure(data=dataLayers, layout=layout)
        figure.add_hline(y=fib, line_dash="dot", row=3, col="all", annotation_text="Uptrend",annotation_position="bottom right")
        figure.add_hline(y=self.lastHHPrice, line_dash="dot", row=3, col="all", annotation_text="HH",annotation_position="bottom right")
        figure.add_hline(y=self.lastHLPrice, line_dash="dot", row=3, col="all", annotation_text="HL",annotation_position="bottom right")

        figure.show()
        # figure.close()

    def displayCandlesDownTrend(self, data, ll_idx, hl_idx, fib):
        trace1 = go.Candlestick(x=data.index,
                                open=data['open'],
                                high=data['high'],
                                low=data['low'],
                                close=data['close'])

        ll = go.Scatter(x=ll_idx.index, y=ll_idx, line=dict(color=colors[2]))
        hl = go.Scatter(x=hl_idx.index, y=hl_idx, line=dict(color="#FF0000"))

        # trace3 = go.Scatter(x=support_indices, y=price_data[support_indices], fill='tonexty', fillcolor='white', opacity=0.1, line=dict(width=0))
        layout = go.Layout(xaxis_rangeslider_visible=False)
        dataLayers = [trace1, ll, hl]
        figure = go.Figure(data=dataLayers, layout=layout)
        figure.add_hline(y=fib, line_dash="dot", row=3, col="all", annotation_text="DownTrend",annotation_position="bottom right")
        figure.add_hline(y=self.lastLLPrice, line_dash="dot", row=3, col="all", annotation_text="LL",annotation_position="bottom right")
        figure.add_hline(y=self.lastLHPrice, line_dash="dot", row=3, col="all", annotation_text="LH",annotation_position="bottom right")

        figure.show()
        # figure.close()