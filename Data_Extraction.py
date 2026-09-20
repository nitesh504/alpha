import pandas as pd
from Z_Score_Calculations import calculate_z_score

# Your existing code
file_path = "SPY_15min_5year_data.xlsx"
df = pd.read_excel(file_path)

if 'Datetime' in df.columns:
    df['Datetime'] = pd.to_datetime(df['Datetime'])

# Call the imported function
df_with_z_scores = calculate_z_score(df)

# Now df_with_z_scores contains the original data plus the Z-Score column
print(df_with_z_scores.columns.tolist())
print(df_with_z_scores[['Datetime', 'Adj Close', 'Z_Score']].head())

# Display the first few rows of the data
print(df.head(50))

# Get basic information about the data
# print("\nData Information:")
# print(f"Number of rows: {df.shape[0]}")
# print(f"Number of columns: {df.shape[1]}")
# print(f"Column names: {df.columns.tolist()}")
# print("\nBasic Statistics:")
# print(df.describe())


