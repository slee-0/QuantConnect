#region imports

from AlgorithmImports import *
#endregion

import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import math 
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

class PairsTradingAlgorithm(QCAlgorithm):
    
    def Initialize(self):
       
        self.SetStartDate(2023, 2, 21)
        self.SetEndDate(2023, 3, 21)
        
        self.SetCash(100000)
        
        self.enter = 2.0 # Set the enter threshold 
        self.exit = 0  # Set the exit threshold 
        self.lookback = 90  # Set the loockback period 90 days
        
        self.pairs = ['MSFT','GOOG']
        self.symbols = []
        
        for ticker in self.pairs:            
            self.AddEquity(ticker, Resolution.Hour)
            self.symbols.append(self.Symbol(ticker))

        self.sym1 = self.symbols[0]
        self.sym2 = self.symbols[1]
        
    def stats(self, symbols):
        
        #Use Statsmodels package to compute linear regression and ADF statistics

        self.df = self.History(symbols, self.lookback)
        self.dg = self.df["open"].unstack(level=0)
        
        #self.Debug(self.dg)
        
        ticker1 = str(symbols[0])
        ticker2 = str(symbols[1])

        Y = self.dg[ticker1].apply(lambda x: math.log(x))
        X = self.dg[ticker2].apply(lambda x: math.log(x))
        
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        results = model.fit()
        sigma = math.sqrt(results.mse_resid)
        slope = results.params[1]
        intercept = results.params[0]
        res = results.resid
        zscore = res/sigma
        adf = adfuller (res)
        
        return [adf, zscore, slope]
     
    def OnData(self, data):

        self.IsInvested = (self.Portfolio[self.sym1].Invested) or (self.Portfolio[self.sym2].Invested)
        self.ShortSpread = self.Portfolio[self.sym1].IsShort
        self.LongSpread = self.Portfolio[self.sym1].IsLong
        
        stats = self.stats([self.sym1, self.sym2])
        self.beta = stats[2]
        zscore = stats[1][-1]
        
        self.wt1 = 1/(1+self.beta)
        self.wt2 = self.beta/(1+self.beta)
        
        self.pos1 = self.Portfolio[self.sym1].Quantity
        self.px1 = self.Portfolio[self.sym1].Price
        self.pos2 = self.Portfolio[self.sym2].Quantity
        self.px2 = self.Portfolio[self.sym2].Price
        
        self.equity = self.Portfolio.TotalPortfolioValue
        
        if self.IsInvested:
           
            if self.ShortSpread and zscore <= self.exit or \
                self.LongSpread and zscore >= self.exit:
                self.Liquidate()
                
        else:
           
            if zscore > self.enter:
                # short the spread
            
                self.SetHoldings(self.sym1, -self.wt1)
                self.SetHoldings(self.sym2, self.wt2)   
            
            if zscore < -self.enter:
                # long the spread
                
                self.SetHoldings(self.sym1, self.wt1)
                self.SetHoldings(self.sym2, -self.wt2) 

        self.pos1 = self.Portfolio[self.sym1].Quantity
        self.pos2 = self.Portfolio[self.sym2].Quantity
    