import pandas as pd
import numpy as np

# Define the input and output file paths
input_csv_path = '/Users/melihkarakose/Desktop/EC 581/data/btc_hourly_data_filtered_from_existing.csv'
output_csv_path = '/Users/melihkarakose/Desktop/EC 581/btc_hourly_data_with_volatility.csv'

# Read the CSV file
df = pd.read_csv(input_csv_path)

# Ensure the 'close' column exists
if 'close' not in df.columns:
    raise ValueError("CSV file must contain a 'close' column.")

# Calculate the 30-period rolling volatility
# Volatility is typically calculated on returns, so first calculate log returns
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
df['volatility_30_period'] = df['log_returns'].rolling(window=30).std() * np.sqrt(30) # Annualized volatility if data is daily, adjust sqrt factor if needed for hourly

# Select relevant columns to save, or save the whole DataFrame
# If you want to save only specific columns:
# output_df = df[['timestamp', 'close', 'volatility_30_period']].copy()
# output_df.to_csv(output_csv_path, index=False)

# Save the entire DataFrame with the new volatility column
df.to_csv(output_csv_path, index=False)

print(f"Successfully calculated volatility and saved to {output_csv_path}")
