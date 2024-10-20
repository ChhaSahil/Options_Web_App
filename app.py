from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
import os

import math
import streamlit as st 
from scipy.stats import norm
import yfinance as yf
import numpy as np
import pandas as pd
import math
import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="Eqity & Derivatives")
st.header("Calculation of Option Price and Options Strategy for European Option Chains on NSE")

ind_symbol = {"NIFTY50":"^NSEI", "NIFTYBANK":"^NSEBANK", "SENSEX":"^BSESN", "^BANKNIFTY":"^NSEBANK","NIFTYMIDCAP100":"NIFTY_MIDCAP_100.NS","NIFTYMIDCAP50":"^NSEMDCP50",
              "FINNIFTY":"NIFTY_FIN_SERVICE.NS","NIFTYFINSERVICE":"NIFTY_FIN_SERVICE.NS","NIFTYMIDSELECT":"NIFTY_MID_SELECT.NS","NIFTYMIDCAPSELECT":"NIFTY_MID_SELECT.NS",
              "NIFTYSMLCAP100":"^CNXSC","NIFTYSMALLCAP100":"^CNXSC","NIFTYSMLCAP50":"NIFTYSMLCAP50.NS","NIFTYSMALLCAP50":"NIFTYSMLCAP50.NS","NIFTYNEXT50":"^NSMIDCP"}

strategy = {"Covered Call":"You Should select 1 OTM/ATM options and CALL Options for this strategy",
            "Protective Put":"You Should select 1 OTM/ATM options and PUT Options for this strategy",
            "Bull Call Spread":"You should select 1 ITM and 1 OTM/ATM options and CALL Options for this strategy",
            "Bear Put Spread":"You should select 2 PUT Options for this strategy",
            "Protective Collar":"You should choose 1 OTM CALL and 1 OTM PUT options",
            "Long Straddle":"You should choose 1 ATM CALL & PUT options each",
            "Long Strangle":"You should choose 1 OTM CALL and 1 OTM PUT options",
            "Long Call Butterfly Spread":"You should select 1 ITM, 2 ATM & 1 OTM CALL Options. Maintain same spread between ITM/ATM & ATM/OTM",
            "Iron Condor":"You should select 2 OTM PUT & 2 OTM CALL Options. Maintain the same even spread between each call & put options",
            "Iron Butterfly":"You should select 1 ATM CALL & PUT each and 1 OTM CALL & PUT Options each"}

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
def volatility(symbol, start_date, end_date):
    history = yf.download(symbol,start_date, end_date)
    daily_return = np.log(history['Close']/history['Close'].shift(1))
    daily_volatility = daily_return.std()
    annual_volatility = daily_volatility*np.sqrt(252)
    return annual_volatility
@st.cache_data
def black_scholes(S, K, T, r, sigma, options = 'call'):
    d1 = (math.log(S/K)+(r+0.5*sigma**0.5)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if options == "call":
        return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    elif options == "put":
        return -S*norm.cdf(-d1)+K*math.exp(-r*T)*norm.cdf(-d2)
    else:
        return "Please choose either call or put"

def binomial_options_pricing(S, K, T, r, sigma, n, options = 'call'):
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt)-d)/(u-d)

    stock_tree = np.zeros((n+1,n+1))
    stock_tree[0,0] = S
    for i in range(1,n+1):
        for j in range(i+1):
            stock_tree[i,j] = stock_tree[i-1,j-1]*u if j>0 else stock_tree[i-1,j]*d
    options_tree = np.zeros((n+1,n+1))
    for i in range(n+1):
        options_tree[n,i] = max(0,stock_tree[n,i]-K) if options == "call" else max(0,K-stock_tree[n,i])
    
    for i in range(n-1,-1,-1):
        for j in range(i+1):
            options_tree[i,j] = np.exp(-r*dt)*(p*options_tree[i+1,j+1]+(1-p)*options_tree[i+1,j])
    return options_tree[0,0]
# st.set_option('deprecation.showPyplotGlobalUse', False)
def monte_carlo_simulations(S,K,T,r,vol,N,M,options):
    dt = T/N
    nudt = (r - 0.5*vol**2)*dt
    volsdt = vol*np.sqrt(dt)
    lnS = np.log(S)

    # Monte Carlo Method
    Z = np.random.normal(size=(N, M))
    delta_lnSt = nudt + volsdt*Z
    lnSt = lnS + np.cumsum(delta_lnSt, axis=0)
    lnSt = np.concatenate( (np.full(shape=(1, M), fill_value=lnS), lnSt ) )

    # Compute Expectation and SE
    ST = np.exp(lnSt)
    if options == 'call':
        CT = np.maximum(0, ST - K)
    else:
        CT = np.maximum(0, K - ST)
    C0 = np.exp(-r*T)*np.sum(CT[-1])/M

    sigma = np.sqrt( np.sum( (CT[-1] - C0)**2) / (M-1) )
    SE = sigma/np.sqrt(M)
    return round(C0,2)
def coveredCall(s,price,premium,options):
    profit = [premium-max(0,i-K)+(i-S) for i in price]
    if len(profit)==0 or options == "put":
        st.write("You Should Choose OTM/ATM options and CALL Options Strategy for profit earning for the selected options strategy")
        return []
    return profit
def protected_put(price,premium,options):
    profit = [-premium+(i-S)+max(0,K-i) for i in price]
    if len(profit)==0 or options=='call':
        st.write("You Should Choose OTM/ATM options and PUT Options Strategy for profit earning for the selected options strategy")
        return []
    return profit

def bull_call_spread(price,premium,premium2,K,K2,options):
    profit = [-abs(premium2-premium)+max(0,i-min(K,K2))-max(0,i-max(K2,K)) for i in price]
    if ((K<S<=K2) or (K2<S<K)) and options == 'call':
        return profit
    st.write("You should select ITM and OTM/ATM options and CALL Options Strategy for profit earnng for the selected options strategy")
    st.write(profit,S)
    return []

def bear_put_spread(price,premium,premium2,K,K2,options):
    profit = [-abs(premium2-premium)+max(0,max(K,K2)-i)-max(0,min(K,K2)-i) for i in price]
    if options == "call":
        st.write("You should choose PUT options for profit earning for this options strategy")
        return []
    return profit

def protective_collar(price,premium,premium2,K,K2):
    profit = [(premium-premium2)-max(0,i-K)+max(0,K2-i)+(i-S) for i in price]
    if S<=K and K2<=S:
        return profit
    st.write("You should choose OTM CALL and OTM PUT options")
    return []

def long_straddle(price,premium,premium2,K):
    profit = [-(premium+premium2)+max(0,i-K)+max(0,K-i) for i in price]
    if K>1.05*S or K<1.05*S:
        st.write("You should choose ATM CALL & PUT options")
        return []
    return profit

def long_strangle(price,premium,premium2,K,K2):
    profit = [-(premium+premium2)+max(0,i-K)+max(0,K2-i) for i in price]
    if S<=K and K2<=S:
        return profit
    st.write("You should choose OTM CALL and OTM PUT options")
    return []

def long_call_butterfly_spread(price,K,K2,K3,K4,premium,premium2,premium3,premium4,options):
    d = {}
    d[K]=premium
    d[K2]=premium2
    d[K3]=premium3
    d[K4]=premium4
    s = sorted(d.items(),key = lambda x:x[0])
    profit = [-s[0][1]-s[-1][1]+s[1][1]+s[2][1]+max(0,i-s[0][0])+max(0,i-s[-1][0])-max(0,i-s[1][0])-max(0,i-s[2][0]) for i in price]
    if options=='put':
        st.write("You should choose CALL Options for this strategy")
        return []
    return profit,s

def iron_condor(price,dic):
    call_s = []
    put_s = []

    call_p = []
    put_p = []

    for i in dic['call']:
        call_s.append(i[0])
        call_p.append(i[1])
    for i in dic['put']:
        put_s.append(i[0])
        put_p.append(i[1])
    profit = [(max(put_p)+max(call_p)-min(call_p)-min(put_p))+max(0,min(put_s)-i)-max(0,max(put_s)-i)+max(0,i-max(call_s))-max(0,i-min(call_s))for i in price]
    
    if max(put_s)<=S and S<=min(call_s) and len(call_p)==2 and len(put_p)==2:
        return profit,put_p,put_s,call_p,call_s
    st.write("You should select 2 OTM PUT & 2 OTM CALL Options")
    return []


def plot(options_strategy,s,price,premium,K,options):
    if options_strategy=='Covered Call':
        profit = coveredCall(s,price,premium,options)
        if len(profit)!=0:
            fig,ax = plt.subplots()
            ax.plot(price,profit)
            ax.axhline(0,color = 'black', linestyle = '--',label='Breakeven')
            ax.axvline(S-premium,color = 'violet', linestyle = '--')
            ax.text(price[0],10,f"Breakeven : ₹{(S-premium):.2f}")
            ax.axhline(profit[-1], color = 'green', linestyle = '--', label = 'Maximum Profit')
            ax.text(price[0],profit[-1]+2,f"Max Profit : ₹{profit[-1]:.2f}",va="bottom",fontsize = 10)
            # plt.scatter(0,K,color = 'white')
        
            st.pyplot(fig)
    elif options_strategy == "Protected Put":
        profit = protected_put(price,premium,options)
        if len(profit)!=0:
            fig,ax = plt.subplots()
            ax.plot(price,profit)
            ax.axhline(0,color = 'black', linestyle = '--',label='Breakeven')
            ax.axvline(K,color = 'yellow', linestyle = '--', label = 'Strike Price')
            ax.axhline(profit[0], color = 'red', linestyle = '--', label = 'Maximum Loss')
            ax.text(price[0],profit[0]+2,f"Max Loss : ₹{profit[0]:.2f}",va="bottom",fontsize = 10)
            # plt.scatter(0,K,color = 'white')
            ax.legend()
            st.pyplot(fig)

def plot2(options_strategy,K,K2,price,premium,premium2,options):
    if options_strategy=='Bull Call Spread':
        profit = bull_call_spread(price,premium,premium2,K,K2,options)
        if len(profit)!=0:
            st.write(profit[0],profit[-1])
            fig,ax = plt.subplots()
            ax.plot(price,profit)
            ax.axhline(0,color = 'black', linestyle = '--')
            # plt.axvline(K,color = 'yellow', linestyle = '--', label = 'Strike Price')
            ax.axhline(profit[-1], color = 'green', linestyle = '--', label = 'Maximum Profit')
            ax.text(price[0],profit[-1]+2,f"Max Profit : ₹{profit[-1]:.2f}",fontsize = 10)
            ax.axhline(profit[0], color = 'red', linestyle = '--', label = 'Maximum Loss')
            ax.text(price[0],profit[0]+2,f"Max Loss : ₹{profit[0]:.2f}",va="bottom",fontsize = 10)
            ax.axvline(min(K,K2)+abs(premium-premium2),color = 'violet', linestyle = '--', label = 'Breakeven')
            ax.text(price[0],10,f"Breakeven Point : ₹{min(K,K2)+abs(premium-premium2):.2f}")
            # plt.scatter(0,K,color = 'white')
            # plt.legend()
            st.pyplot(fig)
    if options_strategy=="Bear Put Spread":
        profit = bear_put_spread(price,premium,premium2,K,K2,options)
        if len(profit)!=0:
            st.write(profit[0],profit[-1])
            fig,ax = plt.subplots()
            ax.plot(price,profit)
            ax.axhline(0,color = 'black', linestyle = '--')
            ax.axhline(profit[0], color = 'green', linestyle = '--', label = 'Maximum Profit')
            ax.text(price[0],profit[0]+2,f"Max Profit : ₹{profit[0]:.2f}",fontsize = 10)
            ax.axhline(profit[-1], color = 'red', linestyle = '--', label = 'Maximum Loss')
            ax.text(price[0],profit[-1]+2,f"Max Loss : ₹{profit[-1]:.2f}",va="bottom",fontsize = 10)
            ax.axvline(max(K,K2)-abs(premium-premium2),color = 'violet', linestyle = '--', label = 'Breakeven')
            ax.text(price[0],10,f"Breakeven Point : ₹{max(K,K2)-abs(premium-premium2):.2f}") 
            st.pyplot(fig)
    
    if options_strategy=="Protective Collar":
        profit = protective_collar(price,premium,premium2,K,K2)
        if len(profit)!=0:
            st.write(profit[0],profit[-1])
            fig,ax = plt.subplots()
            ax.plot(price,profit)
            ax.axhline(0,color = 'black', linestyle = '--')
            # plt.axvline(K,color = 'yellow', linestyle = '--', label = 'Strike Price')
            ax.axhline(profit[-1], color = 'green', linestyle = '--', label = 'Maximum Profit')
            ax.text(price[0],profit[-1]+2,f"Max Profit : ₹{profit[-1]:.2f}",fontsize = 10)
            ax.axhline(profit[0], color = 'red', linestyle = '--', label = 'Maximum Loss')
            ax.text(price[0],profit[0]+2,f"Max Loss : ₹{profit[0]:.2f}",va="bottom",fontsize = 10)
            ax.axvline(S-(premium-premium2),color = 'violet', linestyle = '--', label = 'Breakeven')
            ax.text(price[0],10,f"Breakeven Point : ₹{S-(premium-premium2):.2f}")
            st.pyplot(fig)
    if options_strategy=="Long Straddle":
        profit = long_straddle(price,premium,premium2,K)
        if len(profit)!=0:
            st.write(profit[0],profit[-1])
            fig,ax = plt.subplots()
            ax.plot(price,profit)
            ax.axhline(0,color = 'black', linestyle = '--')
            ax.axvline(K-(premium+premium2),color = 'violet', linestyle = '--', label = 'Breakeven')
            ax.text(price[0],10,f"Breakeven Point 1 : ₹{K-(premium+premium2):.2f}",fontsize=7)
            ax.axvline(K+(premium+premium2),color = 'violet', linestyle = '--', label = 'Breakeven')
            ax.text(price[-1]-10000,10,f"Breakeven Point 2 : ₹{K+(premium+premium2):.2f}",fontsize=7)
            ax.axhline(profit[profit.index(min(profit))], color = 'red', linestyle = '--', label = 'Maximum Loss')
            ax.text(price[0],profit[profit.index(min(profit))]+2,f"Max Loss : ₹{profit[profit.index(min(profit))]:.2f}",va="bottom",fontsize = 7)
            st.pyplot(fig)
    
    if options_strategy=="Long Strangle":
        profit = long_strangle(price,premium,premium2,K,K2)
        if len(profit)!=0:
            st.write(profit[0],profit[-1])
            fig,ax = plt.subplots()
            ax.plot(price,profit)
            ax.axhline(0,color = 'black', linestyle = '--')
            ax.axvline(K2-(premium+premium2),color = 'violet', linestyle = '--', label = 'Breakeven')
            ax.text(price[0],30,f"Breakeven Point 1 : ₹{K2-(premium+premium2):.2f}",fontsize=7)
            ax.axvline(K+(premium+premium2),color = 'violet', linestyle = '--', label = 'Breakeven')
            ax.text(price[-1]-3000,30,f"Breakeven Point 2 : ₹{K+(premium+premium2):.2f}",fontsize=7)
            ax.axhline(profit[profit.index(min(profit))], color = 'red', linestyle = '--', label = 'Maximum Loss')
            ax.text(price[0],profit[profit.index(min(profit))]+2,f"Max Loss : ₹{profit[profit.index(min(profit))]:.2f}",va="bottom",fontsize = 7)
            # plt.scatter(price[len(price)//2],-profit[-1],color = 'white')
            st.pyplot(fig)

def plot3(options_strategy,K,K2,K3,K4, price,premium,premium2,premium3,premium4,options):
    if options_strategy == "Long Call Butterfly Spread":
        profit,s = long_call_butterfly_spread(price,K,K2,K3,K4,premium,premium2,premium3,premium4,options)
        if len(profit)!=0:
            st.write(profit[0],profit[-1])
            fig,ax = plt.subplots()
            ax.plot(price,profit)
            ax.axhline(0,color = 'black', linestyle = '--',label='Breakeven')
            ax.axhline(profit[profit.index(min(profit))], color = 'red', linestyle = '--', label = 'Maximum Loss')
            ax.text(price[0],profit[profit.index(min(profit))]+2,f"Max Loss : ₹{profit[profit.index(min(profit))]:.2f}",va="bottom",fontsize = 7)
            ax.axhline(profit[profit.index(max(profit))], color = 'green', linestyle = '--', label = 'Maximum Profit')
            ax.text(price[0],profit[profit.index(max(profit))]+2,f"Max Profit : ₹{profit[profit.index(max(profit))]:.2f}",va="bottom",fontsize = 7)
            ax.text(price[0],30,f"Breakeven Point 1 : ₹{-(-s[0][1]-s[-1][1]+s[1][1]+s[2][1])+s[0][0]:.2f}",fontsize=7)
            ax.axvline(-(-s[0][1]-s[-1][1]+s[1][1]+s[2][1])+s[0][0],color = 'violet', linestyle = '--', label = 'Breakeven')
            ax.axvline(s[-1][0]+(-s[0][1]-s[-1][1]+s[1][1]+s[2][1]),color = 'violet', linestyle = '--', label = 'Breakeven')
            ax.text(price[-1]-10000,10,f"Breakeven Point 2 : ₹{s[-1][0]+(-s[0][1]-s[-1][1]+s[1][1]+s[2][1]):.2f}",fontsize=7)
            st.pyplot(fig)


def plot4(options_strategy,dic,price):
    if options_strategy=="Iron Condor":
        if len(dic)<2:
            st.write("You should select 2 OTM PUT & 2 OTM CALL Options for this strategy")
            return
        profit,put_p,put_s,call_p,call_s = iron_condor(price,dic)
        if len(profit)!=0:
            st.write(profit)
            st.write(price)
            fig,ax = plt.subplots()
            ax.plot(price,profit)
            ax.text(price[0],10,f"Breakeven Point 1 : ₹{max(put_s)-(max(put_p)+max(call_p)-min(call_p)-min(put_p)):.2f}",fontsize=7)
            ax.axvline(max(put_s)-(max(put_p)+max(call_p)-min(call_p)-min(put_p)),color = 'violet', linestyle = '--', label = 'Breakeven')
            ax.axvline((max(put_p)+max(call_p)-min(call_p)-min(put_p))+min(call_s),color = 'violet', linestyle = '--', label = 'Breakeven')
            ax.text(price[-1]-1500,10,f"Breakeven Point 2 : ₹{(max(put_p)+max(call_p)-min(call_p)-min(put_p))+min(call_s):.2f}",fontsize=7)
            ax.axhline(0,color = 'black', linestyle = '--',label='Breakeven')
            ax.axhline(profit[profit.index(max(profit))], color = 'green', linestyle = '--', label = 'Maximum Profit')
            ax.text(price[0],profit[profit.index(max(profit))]+2,f"Max Profit : ₹{profit[profit.index(max(profit))]:.2f}",va="bottom",fontsize = 7)
            ax.axhline(profit[profit.index(min(profit))], color = 'red', linestyle = '--', label = 'Maximum Loss')
            ax.text(price[0],profit[profit.index(min(profit))]+2,f"Max Loss : ₹{profit[profit.index(min(profit))]:.2f}",va="bottom",fontsize = 7)
            st.pyplot(fig)
# options = st.radio("Options :",["Call","Put"] ,horizontal=True)
with st.sidebar.container():
    st.header("Options Pricing Formula")
    options_formula = st.radio('Choose One of the formulas',["Black & Scholes","Binomial Options Pricing","Monte Carlo Simulation"],horizontal=True)
    # options = st.radio('Type of Options',['call','put'],horizontal=True)
    st.header("Option Strategy")
    options_strategy = st.selectbox("Choose One of the Options Pricing Strategies",("Covered Call","Protected Put","Bull Call Spread", "Bear Put Spread","Protective Collar","Long Straddle","Long Strangle","Long Call Butterfly Spread","Iron Condor"))
    # write_or_buy = st.radio("Do you want to Write(Sell) or Buy the options",["Buy","Write"],horizontal=True)
    st.write(strategy[options_strategy])
st.text(f"Calculate Options Price Using {options_formula}")
display_volatility = False
#----------------------------------Black & Scholes--------------------------------------------
S=0.0
if options_formula == "Black & Scholes":
    symbol = st.text_input("Symbol of the Underlying(Equity) - ",placeholder="Enter the NSE symbol of the equity")
    if symbol:
        if symbol in ind_symbol:
            symbol_NS = ind_symbol[symbol]
        elif ".NS" not in symbol:
           symbol_NS = symbol+".NS"
        else:
           symbol_NS = symbol
        try:
            tk = yf.Ticker(symbol_NS)
            data = yf.download(symbol_NS,'2019-1-1', datetime.date.today())
            csv = convert_df(data)
            S = tk.history("1d")['Close'][-1]
            annul_vol = volatility(symbol_NS, '2019-1-1',datetime.date.today())
        except:
            try:
                tk = yf.Ticker(ind_symbol[symbol.upper().replace(" ",'')])
                data = yf.download(ind_symbol[symbol.upper().replace(" ",'')],'2019-1-1', datetime.date.today())
                csv = convert_df(data)
                S = tk.history("1d")['Close'][-1]
                annul_vol = volatility(ind_symbol[symbol.upper().replace(" ",'')], '2019-1-1',datetime.date.today())
            except:
                st.write(f"Cannot get info on {symbol}. Try removing spaces in the symbol or use the chatbot")
                st.write("CAUTION : Chatbot may give wrong symbol")
                
        # display_volatility = st.checkbox("Show Annual Volatility")
        # if display_volatility:
        #     st.write(f"Annual Volatility of {symbol} is {round(annul_vol*100,2)}%")  
        col1,col2,col3 = st.columns(3)
        s = str(round(S,2))
        with col1:
            st.write("CMP")
            st.write(f":blue[₹{s}]")
        a_vol = str(round(annul_vol*100,2))
        with col2:
            st.write("Annual Volatility")
            st.write(f":blue[{a_vol}%]")
        with col3:
            st.write("Last 5 Year Close Price")
            st.download_button(label = 'Download',data = csv)
        options = st.radio('Options',['call','put'],horizontal = True) 
        K = st.number_input("Strike Price",value = 0)
        T = st.number_input("Number of Days left to expiry",value = 0)
        premium = black_scholes(S, K, T/365, 0.071,annul_vol, options)
        show_price = st.button("Calculate Premium")
        
        s = np.arange(int(S)-int(S)//10,int(S)+int(S)//10,int(math.ceil(0.002*int(S))))
        price = [i if i>0 else 0 for i in s] 
        if show_price:
            st.write(f"Calculated Options Premium is ₹{round(premium,2)}")
        if options_strategy == "Covered Call" or options_strategy == "Protected Put":
            show_pnl = st.toggle("Show PnL Chart",value=False)
            if show_pnl:    
                plot(options_strategy,s,price,premium,K,options)
          
        
        if options_strategy=="Bull Call Spread" or options_strategy=="Bear Put Spread":
            opt2 = st.radio('Options',['call','put'],horizontal=True,key = 'spread')
            K2 = st.number_input("Strike Price 2",value = 0)
            T = st.number_input("Number of Days left to expiry",value = 0,key = "K2")
            premium2 = black_scholes(S, K2, T/365, 0.071,annul_vol, opt2)
            show_price2 = st.button("Calculated 2nd Premium")
            if show_price2:
                st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
            show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
            if show_pnl2:
                plot2(options_strategy,K,K2,price,premium,premium2,options)
        if options_strategy=="Protective Collar" or options_strategy=="Long Strangle":
            if options=="call":
                opt2 = st.radio('Options',['call','put'],horizontal = True,key = "C2")
                K2 = st.number_input("Strike Price 2(Enter OTM PUT Strike)",value = 0)
                T = st.number_input("Number of Days left to expiry",value = 0,key = "K2")
                premium2 = black_scholes(S, K2, T/365, 0.071,annul_vol, opt2)
                show_price2 = st.button("Calculated 2nd Premium")
                if show_price2:
                    st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                if show_pnl2:
                    plot2(options_strategy,K,K2,price,premium,premium2,options)
            elif options=="put":
                K2 = st.number_input("Strike Price 2(Enter OTM CALL Strike)",value=0)
                T = st.number_input("Number of Days left to expiry",value = 0,key = "K2")
                premium2 = black_scholes(S, K2, T/365, 0.071,annul_vol, 'call')
                show_price2 = st.button("Calculated 2nd Premium")
                if show_price2:
                    st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                if show_pnl2:
                    plot2(options_strategy,K2,K,price,premium2,premium,options)
        
        if options_strategy == "Long Straddle":
            st.write("Please remember to choose an ATM option")
            if options=='call':
                premium2 = black_scholes(S,K,T/365,0.071,annul_vol,"put")
                show_price2 = st.button("Calculated 2nd Premium")
                if show_price2:
                    st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                if show_pnl2:
                    plot2(options_strategy,K,K,price,premium,premium2,options)

        if options_strategy == "Long Call Butterfly Spread":
            st.write("You Should Choose 1 ITM, 2 ATM & 1 OTM CALL Options for this strategy")
            if options == "call":
                opt2 = st.radio('',['call','put'],horizontal=True,key='C2')
                K2 = st.number_input("Strike Price 2",value = 0)
                T = st.number_input("Number of Days left to expiry",value = 0,key = "K2")
                premium2 = black_scholes(S, K2, T/365, 0.071,annul_vol, opt2)
                show_price2 = st.button("Calculated 2nd Premium")
                if show_price2:
                    st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                opt3 = st.radio('',['call','put'],horizontal=True,key='C3')
                K3 = st.number_input("Strike Price 3",value = 0)
                T = st.number_input("Number of Days left to expiry",value = 0,key="K3")
                premium3 = black_scholes(S, K3, T/365, 0.071,annul_vol, opt3)
                show_price3 = st.button("Calculated 3rd Premium")
                if show_price3:
                    st.write(f"Calculated Options Premium is ₹{round(premium3,2)}")
                opt4 = st.radio('',['call','put'],horizontal=True,key='C4')
                K4 = st.number_input("Strike Price 4", value = 0)
                T = st.number_input("Number of Days left to expiry",value = 0,key = "K4")
                premium4 = black_scholes(S, K4, T/365, 0.071,annul_vol, opt4)
                show_price4 = st.button("Calculated 4th Premium")
                if show_price4:
                    st.write(f"Calculated Options Premium is ₹{round(premium4,2)}")
                show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                if show_pnl2:
                    plot3(options_strategy,K,K2,K3,K4,price,premium,premium2,premium3,premium4,options)
                
        if options_strategy == "Iron Condor":
            st.write("You should choose 2 OTM & 2 OTM call options")
            dic = defaultdict(list)
            if options=="call" or options=='put':
                dic[options].append((K,premium))
                opt2 = st.radio('',['call','put'],horizontal=True,key='C2')
                K2 = st.number_input("Strike Price 2",value = 0)
                T = st.number_input("Number of Days left to expiry",value = 0,key="K2")
                premium2 = black_scholes(S, K2, T/365, 0.071,annul_vol, opt2)
                dic[opt2].append((K2,premium2))
                show_price2 = st.button("Calculated 2nd Premium")
                if show_price2:
                    st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                opt3 = st.radio('',['call','put'],horizontal=True,key='C3')
                K3 = st.number_input("Strike Price 3",value = 0)
                T = st.number_input("Number of Days left to expiry",value = 0,key = "K3")
                premium3 = black_scholes(S, K3, T/365, 0.071,annul_vol, opt3)
                dic[opt3].append((K3,premium3))
                show_price3 = st.button("Calculated 3rd Premium")
                if show_price3:
                    st.write(f"Calculated Options Premium is ₹{round(premium3,2)}")
                opt4 = st.radio('',['call','put'],horizontal=True,key='C4')
                K4 = st.number_input("Strike Price 4", value = 0)
                T = st.number_input("Number of Days left to expiry",value = 0,key = "K4")
                premium4 = black_scholes(S, K4, T/365, 0.071,annul_vol, opt4)
                dic[opt4].append((K4,premium4))
                show_price4 = st.button("Calculated 4th Premium")
                if show_price4:
                    st.write(f"Calculated Options Premium is ₹{round(premium4,2)}")
                show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                if show_pnl2:
                    # price.append(K)
                    # price.append(K2)
                    # price.append(K3)
                    # price.append(K4)
                    plot4(options_strategy,dic,price)


#-------------------Binomial Options Pricing-----------------------------------------------
if options_formula == "Binomial Options Pricing":
    symbol = st.text_input("Symbol of the Underlying(Equity) - ", placeholder = "Enter the NSE symbol of the equity")
    if symbol:
        if ".NS" not in symbol:
           symbol_NS = symbol+".NS"
        else:
           symbol_NS = symbol
        try:
            tk = yf.Ticker(symbol_NS)
            data = yf.download(symbol_NS,'2019-1-1', datetime.date.today())
            csv = convert_df(data)
            S = tk.history("1d")['Close'][-1]
            annul_vol = volatility(symbol_NS, '2019-1-1',datetime.date.today())
        except:
            try:
                tk = yf.Ticker(ind_symbol[symbol.upper().replace(" ",'')])
                data = yf.download(ind_symbol[symbol.upper().replace(" ",'')],'2019-1-1', datetime.date.today())
                csv = convert_df(data)
                S = tk.history("1d")['Close'][-1]
                annul_vol = volatility(ind_symbol[symbol.upper().replace(" ",'')], '2019-1-1',datetime.date.today())
            except:
                st.write(f"Cannot get info on {symbol}. Try removing spaces in the symbol or use the chatbot")
                st.write("CAUTION : Chatbot may give wrong symbol")

        n = st.number_input("Number of Steps(Should be greater than Zero)",value = 0)
        r = 0.071
        col1,col2,col3 = st.columns(3)
        s = str(round(S,2))
        with col1:
            st.write("CMP")
            st.write(f":blue[₹{s}]")
        a_vol = str(round(annul_vol*100,2))
        with col2:
            st.write("Annual Volatility")
            st.write(f":blue[{a_vol}%]")
        with col3:
            st.write("Last 5 Year Close Price")
            st.download_button(label = 'Download',data = csv)
        options = st.radio('Options',['call','put'],horizontal = True) 
        K = st.number_input("Strike Price", value = 0)
        T = st.number_input("Number of Days Left to Expiry", value = 0)
        if n!=0:
            premium = binomial_options_pricing(S,K,T/365,r,annul_vol,n, options)
            show_price = st.button("Calculate Premium")
            if show_price:
               st.write(f"Calculated Options Premium is ₹{round(premium,2)}")
            s = np.arange(int(S)-int(S)//10,int(S)+int(S)//10,int(math.ceil(0.002*int(S))))
            price = [i if i>0 else 0 for i in s] 
            if options_strategy == "Covered Call" or options_strategy == "Protected Put":
               show_pnl = st.toggle("Show PnL Chart",value=False)
               if show_pnl:    
                    plot(options_strategy,s,price,premium,K,options)
            s = np.arange(int(S)-int(S)//10,int(S)+int(S)//10,int(math.ceil(0.002*int(S))))
            price = [i if i>0 else 0 for i in s]   
            
            if options_strategy=="Bull Call Spread" or options_strategy=="Bear Put Spread":
                opt2 = st.radio('Options',['call','put'],horizontal=True)
                K2 = st.number_input("Strike Price 2",value = 0)
                T = st.number_input("Number of Days left to expiry",value = 0,key="K2")
                premium2 = binomial_options_pricing(S, K2, T/365, 0.071,annul_vol,n, opt2)
                show_price2 = st.button("Calculated 2nd Premium")
                if show_price2:
                    st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                if show_pnl2:
                    plot2(options_strategy,K,K2,price,premium,premium2,options)
            if options_strategy=="Protective Collar" or options_strategy=="Long Strangle":
                if options=="call":
                    opt2 = st.radio('Options',['call','put'],horizontal = True,key = "C2")
                    K2 = st.number_input("Strike Price 2(Enter OTM PUT Strike)",value = 0)
                    T = st.number_input("Number of Days left to expiry",value = 0,key="K2")
                    premium2 = binomial_options_pricing(S, K2, T/365, 0.071,annul_vol,n, opt2)
                    show_price2 = st.button("Calculated 2nd Premium")
                    if show_price2:
                        st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                    show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                    if show_pnl2:
                        plot2(options_strategy,K,K2,price,premium,premium2,options)
                elif options=="put":
                    K2 = st.number_input("Strike Price 2(Enter OTM CALL Strike)",value=0)
                    T = st.number_input("Number of Days left to expiry",value = 0,key="K2")
                    premium2 = binomial_options_pricing(S, K2, T/365, 0.071,annul_vol, n,'call')
                    show_price2 = st.button("Calculated 2nd Premium")
                    if show_price2:
                        st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                    show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                    if show_pnl2:
                        plot2(options_strategy,K2,K,price,premium2,premium,options)
            
            if options_strategy == "Long Straddle":
                st.write("Please remember to choose an ATM option")
                if options=='call':
                    premium2 = binomial_options_pricing(S,K,T/365,0.071,annul_vol,n,"put")
                    show_price2 = st.button("Calculated 2nd Premium")
                    if show_price2:
                        st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                    show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                    if show_pnl2:
                        plot2(options_strategy,K,K,price,premium,premium2,options)

            if options_strategy == "Long Call Butterfly Spread":
                st.write("You Should Choose 1 ITM, 2 ATM & 1 OTM CALL Options for this strategy")
                if options == "call":
                    opt2 = st.radio('',['call','put'],horizontal=True,key='C2')
                    K2 = st.number_input("Strike Price 2",value = 0)
                    T = st.number_input("Number of Days left to expiry",value = 0,key="K2")
                    premium2 = binomial_options_pricing(S, K2, T/365, 0.071,annul_vol, n,opt2)
                    show_price2 = st.button("Calculated 2nd Premium")
                    if show_price2:
                        st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                    opt3 = st.radio('',['call','put'],horizontal=True,key='C3')
                    K3 = st.number_input("Strike Price 3",value = 0)
                    T = st.number_input("Number of Days left to expiry",value = 0,key="K3")
                    premium3 = binomial_options_pricing(S, K3, T/365, 0.071,annul_vol, n,opt3)
                    show_price3 = st.button("Calculated 3rd Premium")
                    if show_price3:
                        st.write(f"Calculated Options Premium is ₹{round(premium3,2)}")
                    opt4 = st.radio('',['call','put'],horizontal=True,key='C4')
                    K4 = st.number_input("Strike Price 4", value = 0)
                    T = st.number_input("Number of Days left to expiry",value = 0,key="K4")
                    premium4 = binomial_options_pricing(S, K4, T/365, 0.071,annul_vol,n, opt4)
                    show_price4 = st.button("Calculated 4th Premium")
                    if show_price4:
                        st.write(f"Calculated Options Premium is ₹{round(premium4,2)}")
                    show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                    if show_pnl2:
                        plot3(options_strategy,K,K2,K3,K4,price,premium,premium2,premium3,premium4,options)
                    
            if options_strategy == "Iron Condor":
                st.write("You should choose 2 OTM & 2 OTM call options")
                dic = defaultdict(list)
                if options=="call" or options=='put':
                    dic[options].append((K,premium))
                    opt2 = st.radio('',['call','put'],horizontal=True,key='C2')
                    K2 = st.number_input("Strike Price 2",value = 0)
                    T = st.number_input("Number of Days left to expiry",value = 0,key="K2")
                    premium2 = binomial_options_pricing(S, K2, T/365, 0.071,annul_vol, n,opt2)
                    dic[opt2].append((K2,premium2))
                    show_price2 = st.button("Calculated 2nd Premium")
                    if show_price2:
                        st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                    opt3 = st.radio('',['call','put'],horizontal=True,key='C3')
                    K3 = st.number_input("Strike Price 3",value = 0)
                    T = st.number_input("Number of Days left to expiry",value = 0,key="K3")
                    premium3 = binomial_options_pricing(S, K3, T/365, 0.071,annul_vol,n, opt3)
                    dic[opt3].append((K3,premium3))
                    show_price3 = st.button("Calculated 3rd Premium")
                    if show_price3:
                        st.write(f"Calculated Options Premium is ₹{round(premium3,2)}")
                    opt4 = st.radio('',['call','put'],horizontal=True,key='C4')
                    K4 = st.number_input("Strike Price 4", value = 0)
                    T = st.number_input("Number of Days left to expiry",value = 0,key="K4")
                    premium4 = binomial_options_pricing(S, K4, T/365, 0.071,annul_vol,n, opt4)
                    dic[opt4].append((K4,premium4))
                    show_price4 = st.button("Calculated 4th Premium")
                    if show_price4:
                        st.write(f"Calculated Options Premium is ₹{round(premium4,2)}")
                    show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                    if show_pnl2:
                        # price.append(K)
                        # price.append(K2)
                        # price.append(K3)
                        # price.append(K4)
                        plot4(options_strategy,dic,price)

#-----------------------------------Monte Carlo Simulation---------------------------------------
if options_formula == "Monte Carlo Simulation":
    symbol = st.text_input("Symbol of the Underlying(Equity) - ", placeholder="Enter the NSE symbol of the equity")
    if symbol:
        if ".NS" not in symbol:
           symbol_NS = symbol+".NS"
        else:
           symbol_NS = symbol
        try:
            tk = yf.Ticker(symbol_NS)
            data = yf.download(symbol_NS,'2019-1-1', datetime.date.today())
            csv = convert_df(data)
            S = tk.history("1d")['Close'][-1]
            annul_vol = volatility(symbol_NS, '2019-1-1',datetime.date.today())
        except:
            try:
                tk = yf.Ticker(ind_symbol[symbol.upper().replace(" ",'')])
                data = yf.download(ind_symbol[symbol.upper().replace(" ",'')],'2019-1-1', datetime.date.today())
                csv = convert_df(data)
                S = tk.history("1d")['Close'][-1]
                annul_vol = volatility(ind_symbol[symbol.upper().replace(" ",'')], '2019-1-1',datetime.date.today())
            except:
                st.write(f"Cannot get info on {symbol}. Try removing spaces in the symbol or use the chatbot")
                st.write("CAUTION : Chatbot may give wrong symbol")

        n = st.number_input("Number of Steps(Should be greater than Zero)",value = 0)
        M = st.number_input("Number of Simulations(Should be greater than Zero)", value = 0)
        r = 0.071
        col1,col2,col3 = st.columns(3)
        s = str(round(S,2))
        with col1:
            st.write("CMP")
            st.write(f":blue[₹{s}]")
        a_vol = str(round(annul_vol*100,2))
        with col2:
            st.write("Annual Volatility")
            st.write(f":blue[{a_vol}%]")
        with col3:
            st.write("Last 5 Year Close Price")
            st.download_button(label = 'Download',data = csv)
        options = st.radio('Options',['call','put'],horizontal = True)
        K = st.number_input("Strike Price", value = 0)
        T = st.number_input("Number of Days Left to Expiry", value = 0)
        
        if n!=0 and M != 0:
            premium = monte_carlo_simulations(S,K,T/365,r,annul_vol,n,M,options) 
            show_price = st.button("Calculate Price")
            if show_price:
               st.write(f"Calculated Options Price is ₹{round(premium,2)}")
            s = np.arange(int(S)-int(S)//10,int(S)+int(S)//10,int(math.ceil(0.002*int(S))))
            price = [i if i>0 else 0 for i in s] 
            if options_strategy == "Covered Call" or options_strategy == "Protected Put":
               show_pnl = st.toggle("Show PnL Chart",value=False)
               if show_pnl:    
                    plot(options_strategy,s,price,premium,K,options)
            s = np.arange(int(S)-int(S)//10,int(S)+int(S)//10,int(math.ceil(0.002*int(S))))
            price = [i if i>0 else 0 for i in s]   
            
            if options_strategy=="Bull Call Spread" or options_strategy=="Bear Put Spread":
                opt2 = st.radio('Options',['call','put'],horizontal=True)
                K2 = st.number_input("Strike Price 2",value = 0)
                T = st.number_input("Number of Days left to expiry",value = 0,key="K2")
                premium2 = monte_carlo_simulations(S, K2, T/365, 0.071,annul_vol,n,M, opt2)
                show_price2 = st.button("Calculated 2nd Premium")
                if show_price2:
                    st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                if show_pnl2:
                    plot2(options_strategy,K,K2,price,premium,premium2,options)
            if options_strategy=="Protective Collar" or options_strategy=="Long Strangle":
                if options=="call":
                    opt2 = st.radio('Options',['call','put'],horizontal = True,key = "C2")
                    K2 = st.number_input("Strike Price 2(Enter OTM PUT Strike)",value = 0)
                    T = st.number_input("Number of Days left to expiry",value = 0,key="K2")
                    premium2 = monte_carlo_simulations(S, K2, T/365, 0.071,annul_vol,n,M ,opt2)
                    show_price2 = st.button("Calculated 2nd Premium")
                    if show_price2:
                        st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                    show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                    if show_pnl2:
                        plot2(options_strategy,K,K2,price,premium,premium2,options)
                elif options=="put":
                    K2 = st.number_input("Strike Price 2(Enter OTM CALL Strike)",value=0)
                    T = st.number_input("Number of Days left to expiry",value = 0,key="K2")
                    premium2 = monte_carlo_simulations(S, K2, T/365, 0.071,annul_vol,n,M, 'call')
                    show_price2 = st.button("Calculated 2nd Premium")
                    if show_price2:
                        st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                    show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                    if show_pnl2:
                        plot2(options_strategy,K2,K,price,premium2,premium,options)
            
            if options_strategy == "Long Straddle":
                st.write("Please remember to choose an ATM option")
                if options=='call':
                    premium2 = monte_carlo_simulations(S,K,T/365,0.071,annul_vol,n,M,"put")
                    show_price2 = st.button("Calculated 2nd Premium")
                    if show_price2:
                        st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                    show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                    if show_pnl2:
                        plot2(options_strategy,K,K,price,premium,premium2,options)

            if options_strategy == "Long Call Butterfly Spread":
                st.write("You Should Choose 1 ITM, 2 ATM & 1 OTM CALL Options for this strategy")
                if options == "call":
                    opt2 = st.radio('',['call','put'],horizontal=True,key='C2')
                    K2 = st.number_input("Strike Price 2",value = 0)
                    T = st.number_input("Number of Days left to expiry",value = 0,key="K2")
                    premium2 = monte_carlo_simulations(S, K2, T/365, 0.071,annul_vol,n,M, opt2)
                    show_price2 = st.button("Calculated 2nd Premium")
                    if show_price2:
                        st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                    opt3 = st.radio('',['call','put'],horizontal=True,key='C3')
                    K3 = st.number_input("Strike Price 3",value = 0)
                    T = st.number_input("Number of Days left to expiry",value = 0,key="K3")
                    premium3 = monte_carlo_simulations(S, K3, T/365, 0.071,annul_vol,n,M, opt3)
                    show_price3 = st.button("Calculated 3rd Premium")
                    if show_price3:
                        st.write(f"Calculated Options Premium is ₹{round(premium3,2)}")
                    opt4 = st.radio('',['call','put'],horizontal=True,key='C4')
                    K4 = st.number_input("Strike Price 4", value = 0)
                    T = st.number_input("Number of Days left to expiry",value = 0,key="K4")
                    premium4 = monte_carlo_simulations(S, K4, T/365, 0.071,annul_vol,n,M, opt4)
                    show_price4 = st.button("Calculated 4th Premium")
                    if show_price4:
                        st.write(f"Calculated Options Premium is ₹{round(premium4,2)}")
                    show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                    if show_pnl2:
                        plot3(options_strategy,K,K2,K3,K4,price,premium,premium2,premium3,premium4,options)
                    
            if options_strategy == "Iron Condor":
                st.write("You should choose 2 OTM & 2 OTM call options")
                dic = defaultdict(list)
                if options=="call" or options=='put':
                    dic[options].append((K,premium))
                    opt2 = st.radio('',['call','put'],horizontal=True,key='C2')
                    K2 = st.number_input("Strike Price 2",value = 0)
                    T = st.number_input("Number of Days left to expiry",value = 0,key="K2")
                    premium2 = monte_carlo_simulations(S, K2, T/365, 0.071,annul_vol,n,M, opt2)
                    dic[opt2].append((K2,premium2))
                    show_price2 = st.button("Calculated 2nd Premium")
                    if show_price2:
                        st.write(f"Calculated Options Premium is ₹{round(premium2,2)}")
                    opt3 = st.radio('',['call','put'],horizontal=True,key='C3')
                    K3 = st.number_input("Strike Price 3",value = 0)
                    T = st.number_input("Number of Days left to expiry",value = 0,key="K3")
                    premium3 = monte_carlo_simulations(S, K3, T/365, 0.071,annul_vol,n,M, opt3)
                    dic[opt3].append((K3,premium3))
                    show_price3 = st.button("Calculated 3rd Premium")
                    if show_price3:
                        st.write(f"Calculated Options Premium is ₹{round(premium3,2)}")
                    opt4 = st.radio('',['call','put'],horizontal=True,key='C4')
                    K4 = st.number_input("Strike Price 4", value = 0)
                    T = st.number_input("Number of Days left to expiry",value = 0,key="K4")
                    premium4 = monte_carlo_simulations(S, K4, T/365, 0.071,annul_vol,n,M, opt4)
                    dic[opt4].append((K4,premium4))
                    show_price4 = st.button("Calculated 4th Premium")
                    if show_price4:
                        st.write(f"Calculated Options Premium is ₹{round(premium4,2)}")
                    show_pnl2 = st.toggle(f"{options_strategy} Chart",value=False)
                    if show_pnl2:
                        # price.append(K)
                        # price.append(K2)
                        # price.append(K3)
                        # price.append(K4)
                        plot4(options_strategy,dic,price)
               
#---------------------------------------------------------------------------------------------

model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])



def get_gemini_response(question,prompt):
    response = model.generate_content([prompt[0],question])
    return response

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
prompt = [f"""
You are an expert in NSE Options trading strategy making and know the latest 
NSE symbols that can be used for yfinance library. For example if a user asks you
'What is the yfinance symbol for NIFTY 50' then you can either refer to the keys of 
the dictionary - {ind_symbol} or if not found you can generate your answer.
"""]
with st.sidebar.container():
    input = st.text_input("Chat: ",key = "Input")
    submit = st.button("Ask the Question")
    if submit and input:
        response = get_gemini_response(input,prompt)
        st.session_state['chat_history'].append(("You",input))
        st.subheader("The Response is")
        for chunk in response:
            st.write(chunk.text)
            st.session_state['chat_history'].append(("Bot",chunk.text))
    
