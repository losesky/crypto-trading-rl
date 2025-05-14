import ccxt
import csv
import datetime
import time

# Configuration
symbol = 'BTC/USDT'
timeframe = '1h'  # 1 hour
since_datetime_str = '2017-01-01T00:00:00Z'
to_datetime_str = '2025-01-01T00:00:00Z' # Note: Binance might not have data up to 2025 yet.
output_csv_file = 'btc_hourly_data_2017_2025.csv'
exchange_id = 'binance'

# Initialize exchange
exchange = getattr(ccxt, exchange_id)()

# Convert string dates to milliseconds
since_timestamp = exchange.parse8601(since_datetime_str)
to_timestamp = exchange.parse8601(to_datetime_str)

# Prepare CSV file
csv_header = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
with open(output_csv_file, 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(csv_header)

    current_timestamp = since_timestamp
    limit = 1000  # Number of candles per request (Binance limit can be up to 1000 for 1m)

    print(f"Starting data download for {symbol} from {since_datetime_str} to {to_datetime_str}")

    while current_timestamp < to_timestamp:
        try:
            print(f"Fetching data from: {exchange.iso8601(current_timestamp)}")
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_timestamp, limit=limit)

            if not ohlcv:
                print("No more data returned or an issue occurred. Stopping.")
                break

            # Write data to CSV
            with open(output_csv_file, 'a', newline='') as f_append:
                csv_writer_append = csv.writer(f_append)
                for candle in ohlcv:
                    # Ensure the candle's timestamp is within the desired range
                    if candle[0] >= to_timestamp:
                        current_timestamp = to_timestamp # Stop if we've reached the end date
                        break
                    # Convert timestamp to human-readable format (optional, but good for verification)
                    # candle[0] = exchange.iso8601(candle[0])
                    csv_writer_append.writerow(candle)
                
                if not ohlcv or candle[0] >= to_timestamp: # check again in case inner loop broke
                    break


            # Move to the next batch of data
            # The next 'since' should be the timestamp of the last candle + timeframe duration
            current_timestamp = ohlcv[-1][0] + exchange.parse_timeframe(timeframe) * 1000
            
            # Respect API rate limits (Binance has a weight limit, be cautious)
            # A small delay can help prevent getting banned.
            # Adjust the sleep time as necessary. 0.2 seconds is a conservative value.
            time.sleep(0.2) 

        except ccxt.NetworkError as e:
            print(f"Network error: {e}. Retrying in 30 seconds...")
            time.sleep(30)
        except ccxt.ExchangeError as e:
            print(f"Exchange error: {e}. Retrying in 60 seconds...")
            time.sleep(60)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

print(f"Data download complete. Saved to {output_csv_file}")
