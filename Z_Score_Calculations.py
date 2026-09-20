def calculate_z_score(df, column='Adj Close', window=20):
    """
    Calculate Z-scores for financial data and add it as a new column.
    
    Parameters:
    df (DataFrame): DataFrame containing the financial data
    column (str): Column to calculate Z-scores for (default: 'Adj Close')
    window (int): Rolling window size for calculating mean and std dev (default: 20)
    
    Returns:
    DataFrame: Original dataframe with additional Z-score column
    """
    # Calculate the rolling mean
    rolling_mean = df[column].rolling(window=window).mean()
    
    # Calculate the rolling standard deviation
    rolling_std = df[column].rolling(window=window).std()
    
    # Calculate Z-score: (price - mean) / standard deviation
    df['Z_Score'] = (df[column] - rolling_mean) / rolling_std
    
    return df