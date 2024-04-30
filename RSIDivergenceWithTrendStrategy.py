from datetime import datetime as dtimport numpy as npfrom backtest import Backtestfrom data import HistoricCSVDataHandlerfrom event import SignalEventfrom execution import SimulatedExecutionHandlerfrom portfolio import Portfoliofrom strategy import Strategyfrom scipy.signal import argrelextremafrom collections import dequeimport numpy as npimport matplotlib.pyplot as pltimport pandas as pdimport numpy as npfrom matplotlib.lines import Line2Dimport plotly.graph_objects as goimport sysfrom scipy.signal import argrelextremacolors = plt.rcParams['axes.prop_cycle'].by_key()['color']class RSIDivergenceWithTrendStrategy(Strategy):    """    Carries out a basic Moving Average Crossover strategy with a    short/long simple weighted moving average. Default short/long    windows are 100/400 periods respectively.    """    def __init__(self, bars, events, short_window=100, long_window=400):        """        Initialises the buy and hold strategy.        Parameters:        bars - The DataHandler object that provides bar information        events - The Event Queue object.        short_window - The short moving average lookback.        long_window - The long moving average lookback.        """        self.bars = bars        self.symbol_list = self.bars.symbol_list        self.events = events        self.short_window = short_window        self.long_window = long_window        # Set to True if a symbol is in the market        self.bought = self._calculate_initial_bought()        self.lastLHIdx = -1        self.lastLHPrice = sys.float_info.max        self.lastLLIdx = -1        self.lastLLPrice = sys.float_info.max        self.lastHHIdx = 0        self.lastHHPrice = 0        self.lastHLIdx = 0        self.lastHLPrice = 0        self.marketPrice = 0        self.upTrend = False        self.downTrend = False        self.breakOfStructureDown = False        self.breakOfStructureUP = False        self.trendChange = False        self.bought = self._calculate_initial_bought()        self.sig_dir = ""    def _calculate_initial_bought(self):        """        Adds keys to the bought dictionary for all symbols        and sets them to 'OUT'.        """        bought = {}        for s in self.symbol_list:            bought[s] = 'OUT'        return bought    def getHigherLows(self, data: np.array, order=5, K=2):        '''        Finds consecutive higher lows in price pattern.        Must not be exceeded within the number of periods indicated by the width        parameter for the value to be confirmed.        K determines how many consecutive lows need to be higher.        '''        # Get lows        low_idx = argrelextrema(data, np.less, order=order)[0]        lows = data[low_idx]        # Ensure consecutive lows are higher than previous lows        extrema = []        ex_deque = deque(maxlen=K)        for i, idx in enumerate(low_idx):            if i == 0:                ex_deque.append(idx)                continue            if lows[i] < lows[i - 1]:                ex_deque.clear()            ex_deque.append(idx)            if len(ex_deque) == K:                extrema.append(ex_deque.copy())        return extrema    def getLowerHighs(self, data: np.array, order=5, K=2):        '''        Finds consecutive lower highs in price pattern.        Must not be exceeded within the number of periods indicated by the width        parameter for the value to be confirmed.        K determines how many consecutive highs need to be lower.        '''        # Get highs        high_idx = argrelextrema(data, np.greater, order=order)[0]        highs = data[high_idx]        # Ensure consecutive highs are lower than previous highs        extrema = []        ex_deque = deque(maxlen=K)        for i, idx in enumerate(high_idx):            if i == 0:                ex_deque.append(idx)                continue            if highs[i] > highs[i - 1]:                ex_deque.clear()            ex_deque.append(idx)            if len(ex_deque) == K:                extrema.append(ex_deque.copy())        return extrema    def getHigherHighs(self, data: np.array, order=5, K=2):        '''        Finds consecutive higher highs in price pattern.        Must not be exceeded within the number of periods indicated by the width        parameter for the value to be confirmed.        K determines how many consecutive highs need to be higher.        '''        # Get highs        # use extend() to add all the elements of the tuple to the list        high_idx = argrelextrema(data, np.greater, order=5)[0]        highs = data[high_idx]        # Ensure consecutive highs are higher than previous highs        extrema = []        ex_deque = deque(maxlen=K)        for i, idx in enumerate(high_idx):            if i == 0:                ex_deque.append(idx)                continue            if highs[i] < highs[i - 1]:                ex_deque.clear()            ex_deque.append(idx)            if len(ex_deque) == K:                extrema.append(ex_deque.copy())        return extrema    def getLowerLows(self, data: np.array, order=5, K=2):        '''        Finds consecutive lower lows in price pattern.        Must not be exceeded within the number of periods indicated by the width        parameter for the value to be confirmed.        K determines how many consecutive lows need to be lower.        '''        # Get lows        indices = data.astype(np.int)        low_idx = argrelextrema(indices, np.less, order=order)[0]        lows = data[low_idx]        # Ensure consecutive lows are lower than previous lows        extrema = []        ex_deque = deque(maxlen=K)        for i, idx in enumerate(low_idx):            if i == 0:                ex_deque.append(idx)                continue            if lows[i] > lows[i - 1]:                ex_deque.clear()            ex_deque.append(idx)            if len(ex_deque) == K:                extrema.append(ex_deque.copy())        return extrema    def getHHIndex(self, data: np.array, order=5, K=2):        extrema = self.getHigherHighs(data, order, K)        idx = np.array([i[-1] + order for i in extrema])        return idx[np.where(idx < len(data))]    def getLHIndex(self, data: np.array, order=5, K=2):        extrema = self.getLowerHighs(data, order, K)        idx = np.array([i[-1] + order for i in extrema])        return idx[np.where(idx < len(data))]    def getLLIndex(self, data: np.array, order=5, K=2):        extrema = self.getLowerLows(data, order, K)        idx = np.array([i[-1] + order for i in extrema])        return idx[np.where(idx < len(data))]    def getHLIndex(self, data: np.array, order=5, K=2):        extrema = self.getHigherLows(data, order, K)        idx = np.array([i[-1] + order for i in extrema])        return idx[np.where(idx < len(data))]    def getPeaks(self, data, key='close', order=5, K=2):        vals = data        hh_idx = self.getHHIndex(vals, order, K)  # TypeError: list indices must be integers or slices, not str        lh_idx = self.getLHIndex(vals, order, K)        ll_idx = self.getLLIndex(vals, order, K)        hl_idx = self.getHLIndex(vals, order, K)        data[f'{key}_highs'] = np.nan        data[f'{key}_highs'][hh_idx] = 1        data[f'{key}_highs'][lh_idx] = -1        data[f'{key}_highs'] = data[f'{key}_highs'].ffill().fillna(0)        data[f'{key}_lows'] = np.nan        data[f'{key}_lows'][ll_idx] = 1        data[f'{key}_lows'][hl_idx] = -1        data[f'{key}_lows'] = data[f'{key}_highs'].ffill().fillna(0)        return data    def _calcEMA(self, P, last_ema, N):        return (P - last_ema) * (2 / (N + 1)) + last_ema    def calcEMA(self, data, N):        # Initialize series        data['SMA_' + str(N)] = data['close'].rolling(N).mean()        ema = np.zeros(len(data))        for i, _row in enumerate(data.iterrows()):            row = _row[1]            if i < N:                ema[i] += row['SMA_' + str(N)]            else:                ema[i] += self._calcEMA(row['close'], ema[i - 1], N)        data['EMA_' + str(N)] = ema.copy()        return data    def calcRSI(self, data, P=14):        data['diff_close'] = data['close'] - data['close'].shift(1)        data['gain'] = np.where(data['diff_close'] > 0, data['diff_close'], 0)        data['loss'] = np.where(data['diff_close'] < 0, np.abs(data['diff_close']), 0)        data[['init_avg_gain', 'init_avg_loss']] = data[            ['gain', 'loss']].rolling(P).mean()        avg_gain = np.zeros(len(data))        avg_loss = np.zeros(len(data))        for i, _row in enumerate(data.iterrows()):            row = _row[1]            if i < P - 1:                last_row = row.copy()                continue            elif i == P - 1:                avg_gain[i] += row['init_avg_gain']                avg_loss[i] += row['init_avg_loss']            else:                avg_gain[i] += ((P - 1) * avg_gain[i - 1] + row['gain']) / P                avg_loss[i] += ((P - 1) * avg_loss[i - 1] + row['loss']) / P            last_row = row.copy()        data['avg_gain'] = avg_gain        data['avg_loss'] = avg_loss        data['RS'] = data['avg_gain'] / data['avg_loss']        data['RSI'] = 100 - 100 / (1 + data['RS'])        return data    def getLastHLBeforeIndex(self, lHIndexes, hhIndex, lastHHPrice, bars, order):        """        Finds the first (closest) index of the Higher Low before the given Higher High        Parameters        lHIndexes - lower High indexes.        hhIndex - the last Higher High to search for the previous Lower High        """        for hLIndex in lHIndexes[::-1]:            if hLIndex < hhIndex and bars[hLIndex-order] < lastHHPrice:                return hLIndex    def display(self, data, hh_idx, lh_idx, order):        plt.figure(figsize=(15, 8))        plt.plot(data['close'])        close = data['close'].values        date = data.index        # _ = [plt.plot(date[i-order], close[i-order], c=colors[1]) for i in hh_idx]        # _ = [plt.plot(date[i-order], close[i-order], c=colors[2]) for i in lh_idx]        plt.scatter(date[hh_idx - order], close[hh_idx - order], c=colors[1], marker='^', s=100)        plt.scatter(date[lh_idx - order], close[lh_idx - order], c=colors[2], marker='^', s=100)        plt.show()        plt.close()    def displayCandles(self, data, hh_idx, lh_idx, order):        trace1 = go.Candlestick(x=data.index,                                open=data['open'],                                high=data['high'],                                low=data['low'],                                close=data['close'])        trace2 = go.Scatter(x=data.close[hh_idx - order].index, y=data.close[hh_idx - order])        trace3 = go.Scatter(x=data.close[lh_idx - order].index, y=data.close[lh_idx - order])        # trace3 = go.Scatter(x=support_indices, y=price_data[support_indices], fill='tonexty', fillcolor='white', opacity=0.1, line=dict(width=0))        layout = go.Layout(xaxis_rangeslider_visible=False)        dataLayers = [trace1, trace2, trace3]        figure = go.Figure(data=dataLayers, layout=layout)        figure.show()        # figure.close()    def displayCandlesUpTrend(self, data, hh_idx, lh_idx, fib):        trace1 = go.Candlestick(x=data.index,                                open=data['open'],                                high=data['high'],                                low=data['low'],                                close=data['close'])        hh = go.Scatter(x=hh_idx.index, y=hh_idx, line=dict(color="#FF0000"))        lh = go.Scatter(x=lh_idx.index, y=lh_idx, line=dict(color=colors[2]))        # trace3 = go.Scatter(x=support_indices, y=price_data[support_indices], fill='tonexty', fillcolor='white', opacity=0.1, line=dict(width=0))        layout = go.Layout(xaxis_rangeslider_visible=False)        dataLayers = [trace1, hh, lh]        figure = go.Figure(data=dataLayers, layout=layout)        figure.add_hline(y=fib, line_dash="dot", row=3, col="all")        figure.show()        # figure.close()    def displayCandlesDownTrend(self, data, ll_idx, hl_idx):        trace1 = go.Candlestick(x=data.index,                                open=data['open'],                                high=data['high'],                                low=data['low'],                                close=data['close'])        ll = go.Scatter(x=ll_idx.index, y=ll_idx, line=dict(color=colors[2]))        hl = go.Scatter(x=hl_idx.index, y=hl_idx, line=dict(color="#FF0000"))        # trace3 = go.Scatter(x=support_indices, y=price_data[support_indices], fill='tonexty', fillcolor='white', opacity=0.1, line=dict(width=0))        layout = go.Layout(xaxis_rangeslider_visible=False)        dataLayers = [trace1, ll, hl]        figure = go.Figure(data=dataLayers, layout=layout)        figure.show()        # figure.close()    def calculate_signals(self, event, P=14, order=2, K=2, EMA1=50, EMA2=200):        """        Generates a new set of signals based on the MAC        SMA with the short window crossing the long window        meaning a long entry and vice versa for a short entry.            Parameters        event - A MarketEvent object.         """        if event.type == 'MARKET':            for symbol in self.symbol_list:                bars = self.bars.get_latest_bars_values(symbol, "close", N=self.long_window)                data = self.bars.get_latest_bars(symbol, N=self.long_window)                index = [tup[0] for tup in data]                columns = [tup[1] for tup in data]                data = pd.DataFrame(columns, index, columns=["open", "high", "low", "close", "volume"])                # high = self.bars.get_latest_bars_values(symbol, "high", N=self.long_window)                # dates = np.array([b[0]for b in data])                strength = 1.0                strategy_id = 1                if bars is not None and bars.size > 2:                    data["valuesHH"] = np.where(data.close > data.high, data['close'], data['high'])                    price_data = data['valuesHH'].values                    hh_idx = self.getHHIndex(price_data, order, K)                    hh = data.iloc[hh_idx - order]['valuesHH']  # SUBSTRACT ORDER                    data["valuesHL"] = np.where(data.close < data.low, data['close'],  data['low'])                   # data["valuesHL"] = np.where(data.close > data.high, data['close'], data['close'])                    price_data = data['valuesHL'].values                    hl_idx = self.getHLIndex(price_data, order, K)                    hl = data.iloc[hl_idx - order]['valuesHL']  # SUBSTRACT ORDER                    self.marketPrice = bars[-1]                    if len(hh_idx) >= 2 and bars[hh_idx[-1]] > bars[hh_idx[-2]] and hh_idx[-1] != self.lastHHIdx and \                            bars[hh_idx[-1]] > self.lastHHPrice:  # only new HH                        self.upTrend = True                        # self.downTrend = False                        self.lastHHIdx = hh_idx[-1]-order                        # self.lastHHPrice = bars[self.lastHHIdx] # use data instead data.iloc[self.lastLLIdx]['valuesHH']                        self.lastHHPrice = data.iloc[self.lastHHIdx]['valuesHH']                        self.lastHLIdx = self.getLastHLBeforeIndex(hl_idx, self.lastHHIdx, self.lastHHPrice, bars, order)                        # self.lastHLPrice = bars[self.lastHLIdx-order] # use data instead                        self.lastHLPrice = data.iloc[self.lastHLIdx-order]['valuesHL'] # why -order again?                        half_level = self.lastHHPrice - (self.lastHHPrice - self.lastHLPrice) * 0.5                        print("Uprange [" + str(self.lastHLPrice) + "," + str(self.lastHHPrice) + "] " + str(half_level))                        self.displayCandlesUpTrend(data, hh, hl)                        if self.downTrend and self.lastHHPrice < self.lastHLPrice:  ## (internal structure to ignore)                            self.upTrend = False                            self.downTrend = True                        if self.bought[symbol] == "OUT":                            self.sig_dir = 'LONG'                            signal = SignalEvent(strategy_id, symbol, dt, self.sig_dir, strength)                            self.events.put(signal)                            self.bought[symbol] = 'LONG'                    if self.upTrend and self.marketPrice > self.lastHHPrice:  # current market is above last HH and No new HH                        self.breakOfStructureDown = False                        self.breakOfStructureUP = True                    if self.upTrend and self.lastLHPrice is not None and self.marketPrice < self.lastHLPrice:  # below last HL                        self.upTrend = False                        self.downTrend = False                        self.breakOfStructureUP = False                        self.breakOfStructureDown = True  # liquidity grab if downtrend is not formed                        self.displayCandlesUpTrend(data, hh, hl)                        if self.bought[symbol] == "LONG":                            self.sig_dir = 'EXIT'                            signal = SignalEvent(strategy_id, symbol, dt, self.sig_dir, strength)                            self.events.put(signal)                            self.bought[symbol] = 'OUT'                    data["valuesLL"] = np.where(data.close < data.low, data['close'], data['low'])                    price_data = data['valuesLL'].values                    ll_idx = self.getLLIndex(price_data, order, K)                    ll = data.iloc[ll_idx - order]['valuesLL']  # SUBSTRACT ORDER                    data["valuesLH"] = np.where(data.close > data.high, data['close'], data['high'])                    price_data = data['valuesLH'].values                    lh_idx = self.getLHIndex(price_data, order, K)                    lh = data.iloc[lh_idx - order]['valuesLH']  # SUBSTRACT ORDER                    if len(ll_idx) >= 2 and bars[ll_idx[-1]] < bars[ll_idx[-2]] and ll_idx[-1] != self.lastLLIdx and \                            bars[ll_idx[-1]] < self.lastLLPrice:  # only new LL                        self.downTrend = True                        self.lastLHIdx = lh_idx[-1]-order                        self.lastLHPrice = data.iloc[self.lastLHIdx]['valuesLH']                        #self.lastLHPrice = bars[self.lastLHIdx]                        self.lastLLIdx = ll_idx[-1]                        self.lastLLPrice = data.iloc[self.lastLLIdx - order]['valuesLL']                        print("Downrange [" + str(self.lastLHPrice) + "," + str(self.lastLLPrice) + "]")                        self.displayCandlesDownTrend(data, ll, lh)                        if self.upTrend and self.lastLLPrice > self.lastLHPrice:  # internal Down structure in Uptrend (ignore)                            self.downTrend = False                            self.upTrend = True                            self.breakOfStructureDown = True                            self.breakOfStructureUP = False                        if self.downTrend and self.marketPrice < self.lastLLPrice:  # current market is below last LL (Sbreak to downside)                            self.breakOfStructureDown = True                            self.breakOfStructureUP = False                        if self.downTrend and self.lastLLPrice is not None and self.marketPrice > self.lastLHPrice:  # above last HL                            self.trendChange = True                            self.breakOfStructureUP = False                            self.breakOfStructureDown = False                            self.downTrend = False                            self.displayCandlesDownTrend(data, ll, lh)if __name__ == "__main__":    csv_dir = "C:\\Users\\machadojo\\PycharmProjects\\SuccessfulAlgoTrading\\chapter-event-driven-trading\\"  # TODO: create file    # symbol_list = ['BTC_USDT-4h2']  # TODO: change to XOM    symbol_list = ['BTC_USDT-15m']  # TODO: change to XOM    initial_capital = 100000.0    start_date = dt(2022, 4, 11)    heartbeat = 0.0    backtest = Backtest(csv_dir,                        symbol_list,                        initial_capital,                        heartbeat,                        start_date,                        HistoricCSVDataHandler,                        SimulatedExecutionHandler,                        Portfolio,                        RSIDivergenceWithTrendStrategy)    backtest.simulate_trading()