import matplotlib.pyplot as plt
import numpy as np

def plot_prices(y, y_hat, num_houses_to_plot=20):
    plt.figure(figsize=(10, 6))
    plt.title("Comparison of Actual and Predicted House Prices")
    plt.xlabel("House Index")
    plt.ylabel("House Price")
    plt.xticks(np.arange(num_houses_to_plot))
    bar_width = 0.4
    plt.bar(np.arange(num_houses_to_plot), y[:num_houses_to_plot], width=bar_width, label="Actual Prices", color="blue")
    plt.bar(np.arange(num_houses_to_plot) + bar_width, y_hat[:num_houses_to_plot], width=bar_width, label="Predicted Prices", color="red")

    plt.legend()
    plt.show()