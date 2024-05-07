import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv("losses_data.csv")

# Plotting
plt.figure(figsize=(10, 6))

# Plotting each column against Epochs
for column in df.columns:
    if column != 'Epochs':  # Exclude 'Epochs' column
        plt.plot(df['Epochs'], df[column], label=column)

plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig(losses_per_epoch.png)
# Show plot
plt.show()
