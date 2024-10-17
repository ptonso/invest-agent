import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

class DescriptorControl:
    def __init__(self, n_stocks, plot_baselines=False):
        self.plot_baselines = plot_baselines

        self.allocations = []
        self.cumulative_values = []
        self.dates = []
        self.n_stocks = n_stocks

        plt.ion()  # Turn on interactive mode

        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(12, 8),
            gridspec_kw={'height_ratios': [1, 3]}
        )

        # Initialize the cumulative wallet value plot
        self.line1, = self.ax1.plot([], [], color='blue', label='Agent Portfolio')
        self.ax1.set_title('Cumulative Wallet Value')
        self.ax1.set_ylabel('Value')
        self.ax1.grid(True)

        # Initialize the heatmap
        self.im = None
        self.ax2.set_title('Portfolio Allocation Over Time')
        self.ax2.set_ylabel('Stocks')
        self.ax2.set_xlabel('Time Steps')
        self.ax2.set_yticks(range(self.n_stocks))
        self.ax2.set_yticklabels([f'Stock {i}' for i in range(self.n_stocks)])

        if self.plot_baselines:
            self.cumulative_values_stock1 = []
            self.cumulative_values_allstocks = []
            self.line_baseline1, = self.ax1.plot([], [], color='red', label='Stock 1 Baseline')
            self.line_baseline2, = self.ax1.plot([], [], color='green', label='Equal Weights Baseline')
            self.ax1.legend()

        plt.tight_layout()
        plt.show()

    def update(self, allocation, cumulative_value, date, baseline_values=None):
        self.allocations.append(allocation.detach().cpu().numpy())
        self.cumulative_values.append(float(cumulative_value))
        self.dates.append(date)

        if self.plot_baselines and baseline_values:
            baseline_value1, baseline_value2 = baseline_values
            self.cumulative_values_stock1.append(float(baseline_value1))  # Ensure float type
            self.cumulative_values_allstocks.append(float(baseline_value2))  # Ensure float type

        self.plot()

    def plot(self):
        self.line1.set_data(range(len(self.cumulative_values)), self.cumulative_values)
        self.ax1.relim()
        self.ax1.autoscale_view()

        if self.plot_baselines:
            x_range = range(len(self.cumulative_values_stock1))
            self.line_baseline1.set_data(x_range, self.cumulative_values_stock1)
            self.line_baseline2.set_data(x_range, self.cumulative_values_allstocks)

            all_values = self.cumulative_values + self.cumulative_values_stock1 + self.cumulative_values_allstocks
            self.ax1.set_ylim(min(all_values), max(all_values))

        # Prepare allocation data for heatmap
        allocations_array = np.array(self.allocations).T  # Shape: (n_stocks, n_time_steps)

        if self.im is None:
            self.im = self.ax2.imshow(
                allocations_array, aspect='auto', cmap='viridis', interpolation='none'
            )
            self.fig.colorbar(self.im, ax=self.ax2, label='Allocation Percentage')
        else:
            self.im.set_data(allocations_array)
            self.im.set_extent([0, allocations_array.shape[1], 0, allocations_array.shape[0]])

        self.ax2.set_xlim(0, allocations_array.shape[1])

        plt.draw()
        plt.pause(0.001)
