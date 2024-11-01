import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt

def load_and_clean_data(file_path):
    """
    Load Brent oil price data, parse dates, handle missing values.
    """
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  
    
    
    data.fillna(method='ffill', inplace=True)
    return data

def plot_oil_price(data):
    
    plt.figure(figsize=(12, 6))
    plt.plot(data['Price'], label='Brent Oil Price')
    plt.title('Brent Oil Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

def change_point_detection(data):
  
    signal = data['Price'].values
    algo = rpt.Pelt(model="rbf").fit(signal)
    change_points = algo.predict(pen=10)  
    
  
    rpt.display(signal, change_points, figsize=(12, 6))
    plt.title("Change Point Detection in Brent Oil Prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.show()
    
    return change_points





def summarize_data(data):
    
    summary = {
        "Metric": [
            "Start Date", "End Date", "Total Observations", 
            "Missing Values", "Price Mean", 
            "Price Standard Deviation", "Price Min", "Price Max"
        ],
        "Value": [
            data.index.min(), data.index.max(), data.shape[0],
            data.isnull().sum().sum(), data['Price'].mean(), 
            data['Price'].std(), data['Price'].min(), data['Price'].max()
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    print(summary_df)
    return summary_df









