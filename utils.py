import yfinance as yf
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import google.generativeai as genai
import pandas as pd


class StockDataAnalyzer:
    def __init__(self, symbol, genai_api_key, news_api_key):
        self.symbol = symbol
        self.genai_api_key = genai_api_key
        self.news_api_key = news_api_key
        self.ticker = None
        self._initialize()
        genai.configure(api_key=self.genai_api_key)
        self.newsapi = NewsApiClient(api_key=self.news_api_key)

    def _initialize(self):
        """
        Validate and fetch stock ticker information.
        """
        try:
            self.ticker = yf.Ticker(self.symbol)
            print(f"Successfully initialized ticker: {self.symbol}")
        except Exception as e:
            raise ValueError(f"Error initializing ticker for {self.symbol}: {e}")

    def fetch_stock_info(self):
        """
        Fetch basic stock information such as company name, market cap, etc.
        """
        try:
            info = self.ticker.info
            return {
                "Company Name": info.get("longName", "N/A"),
                "Symbol": self.symbol,
                "Current Price": info.get("currentPrice", "N/A"),
                "Market Cap": info.get("marketCap", "N/A"),
                "Exchange": info.get("exchange", "N/A"),
                "PE Ratio": info.get("trailingPE", "N/A"),
                "Dividend Yield": info.get("dividendYield", "N/A"),
            }
        except Exception as e:
            print(f"Error fetching stock info: {e}")
            return {}

    def fetch_stock_performance(self):
        """
        Fetch stock performance metrics such as total return, volatility, etc.
        """
        try:
            hist = self.ticker.history(period="1mo")
            if hist.empty:
                return {}
            total_return = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
            daily_returns = hist["Close"].pct_change()
            volatility = daily_returns.std() * (252**0.5)  # Annualized volatility
            max_drawdown = ((hist["Close"] / hist["Close"].cummax()) - 1).min() * 100
            avg_daily_return = daily_returns.mean() * 100
            return {
                "Total Return": f"{total_return:.2f}%",
                "Volatility": f"{volatility:.2f}%",
                "Max Drawdown": f"{max_drawdown:.2f}%",
                "Average Daily Return": f"{avg_daily_return:.2f}%",
            }
        except Exception as e:
            print(f"Error fetching stock performance: {e}")
            return {}

    def fetch_related_news(self):
        """
        Fetch recent news articles related to the stock symbol or company name.
        """
        try:
            # Fetch the company name to broaden the search query
            company_name = self.ticker.info.get("longName", self.symbol)
            
            # Try fetching news using company name first
            articles = self.newsapi.get_everything(
                q=company_name,
                language="en",
                sort_by="relevancy",
                page=1
            )
            news_items = [
                {"title": article["title"], "link": article["url"]}
                for article in articles.get("articles", [])
            ]
            
            # Fallback to stock symbol search if no news is found
            if not news_items:
                print(f"No news found for company name '{company_name}', trying stock symbol...")
                articles = self.newsapi.get_everything(
                    q=self.symbol,
                    language="en",
                    sort_by="relevancy",
                    page=1
                )
                news_items = [
                    {"title": article["title"], "link": article["url"]}
                    for article in articles.get("articles", [])
                ]
            
            # Limit to top 5 articles
            return news_items[:5]
        
        except Exception as e:
            print(f"Error fetching news for {self.symbol}: {e}")
            return []


    def fetch_historical_data(self, start_date, end_date):
        """
        Fetch historical stock data between the given dates.
        """
        try:
            hist_data = self.ticker.history(start=start_date, end=end_date, interval="1d")
            if hist_data.empty:
                print(f"No historical data found for {self.symbol}")
                return None
            return hist_data
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None

    def generate_prediction_prompt(self, historical_data, news_articles):
        """
        Construct a prompt for Gemini AI using historical data and news.
        """
        historical_table = historical_data.to_string(index=False)
        news_titles = "\n".join([f"- {item['title']} (Link: {item['link']})" for item in news_articles])
        return f"""
        Based on the following stock data and news, predict the next day's stock price 
        and advise if it's a good time to buy:

        Historical Stock Data:
        {historical_table}

        News:
        {news_titles}

        Provide:
        1. Predicted stock price.
        2. Buy/Hold/Sell recommendation.
        3. Brief reasoning.
        """

    def predict_with_gemini(self, historical_data, news_articles):
        """
        Predict stock price and recommendation using Gemini AI.
        """
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = self.generate_prediction_prompt(historical_data, news_articles)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error using Gemini for prediction: {e}")
            return None
