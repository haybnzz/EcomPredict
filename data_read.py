import pandas as pd

# Load the dataset
file_path = "data.csv"
df = pd.read_csv(file_path)

# Display basic info and first few rowss
df.info(), df.head()
