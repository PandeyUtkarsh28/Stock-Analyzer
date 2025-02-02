from datetime import datetime, timedelta
from utils import StockDataAnalyzer


def display_stock_details(stock_info, performance, news):
    print("\n--- Stock Information ---")
    for key, value in stock_info.items():
        print(f"{key}: {value}")

    print("\n--- Financial Highlights ---")
    print(f"PE Ratio: {stock_info.get('PE Ratio', 'N/A')}")
    print(f"Dividend Yield: {stock_info.get('Dividend Yield', 'N/A')}")

    print("\n--- Stock Performance ---")
    for key, value in performance.items():
        print(f"{key}: {value}")

    print("\n--- Recent News ---")
    for item in news:
        print(f"Title: {item['title']}\nLink: {item['link']}\n---")


def main():
    symbol = input("Enter stock symbol: ").strip().upper()
    genai_api_key = "AIzaSyDCRZOok-jCCpl-q5kCSXR9fhA3lpdflvY"
    news_api_key = "bc31920a622745e088b1880b550165a3"

    analyzer = StockDataAnalyzer(symbol, genai_api_key, news_api_key)

    stock_info = analyzer.fetch_stock_info()
    performance = analyzer.fetch_stock_performance()
    news = analyzer.fetch_related_news()

    display_stock_details(stock_info, performance, news)

    # Fetch historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    historical_data = analyzer.fetch_historical_data(start_date, end_date)

    if historical_data is not None and news:
        prediction = analyzer.predict_with_gemini(historical_data, news)
        print("\n--- Prediction and Recommendation ---")
        print(prediction)
    else:
        print("Not enough data for prediction.")


if __name__ == "__main__":
    main()
