import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

class Descriptor:
    def __init__(self, n_stocks, baseline_flag=False):
        self.baseline_flag = baseline_flag

        self.allocations = []
        self.cumulative_values = []
        self.dates = []
        self.n_stocks = n_stocks

        plt.ion()
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(12, 8),
            gridspec_kw={'height_ratios': [1, 3]}
        )

        self.line1, = self.ax1.plot([], [], color='blue', label='Agent Portfolio')
        self.ax1.set_title('Cumulative Wallet Value')
        self.ax1.set_ylabel('Value')
        self.ax1.grid(True)

        self.im = None
        self.ax2.set_title('Portfolio Allocation Over Time')
        self.ax2.set_ylabel('Stocks')
        self.ax2.set_xlabel('Time Steps')

        if self.baseline_flag:
            self.cumulative_values_baseline = []
            self.line_baseline, = self.ax1.plot([], [], color='green', label='Equal Weights Baseline')
            self.ax1.legend()

        plt.tight_layout()
        plt.show()

    def update(self, allocation, cumulative_value, current_date, baseline_value=None):
        allocation = allocation.squeeze()
        self.allocations.append(allocation.detach().cpu().numpy())
        self.cumulative_values.append(float(cumulative_value))
        self.dates.append(current_date)

        if self.baseline_flag and baseline_value:
            self.cumulative_values_baseline.append(float(baseline_value))

        self.plot()

    def plot(self):
        self.line1.set_data(self.dates, self.cumulative_values)
        self.ax1.relim()
        self.ax1.autoscale_view()

        if self.baseline_flag:
            self.line_baseline.set_data(self.dates, self.cumulative_values_baseline)
            all_values = self.cumulative_values + self.cumulative_values_baseline
            self.ax1.set_ylim(min(all_values), max(all_values))

        self.ax1.xaxis.set_major_locator(mdates.YearLocator())
        self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(self.ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)

        allocations_array = np.array(self.allocations).T  # Shape: (n_stocks, n_time_steps)
        date_nums = mdates.date2num(self.dates)

        if len(date_nums) > 1:
            extent = [date_nums[0], date_nums[-1], 0, allocations_array.shape[0]]
        else:
            extent = [date_nums[0] - 1, date_nums[0] + 1, 0, allocations_array.shape[0]]

        if self.im is None:
            self.im = self.ax2.imshow(
                allocations_array,
                aspect='auto',
                cmap='viridis',
                interpolation='none',
                extent=extent
            )
            self.fig.colorbar(self.im, ax=self.ax2, label='Allocation Percentage')
        else:
            self.im.set_data(allocations_array)
            self.im.set_extent([date_nums[0], date_nums[-1], 0, allocations_array.shape[0]])

        self.ax2.set_title('Portfolio Allocation Over Time')
        self.ax2.set_ylabel('Stocks')
        self.ax2.set_xlabel('Date')

        if self.n_stocks > 20:
            sampled_indices = np.linspace(0, self.n_stocks - 1, 20, dtype=int)
        else:
            sampled_indices = range(self.n_stocks)

        self.ax2.set_yticks(sampled_indices)
        self.ax2.set_yticklabels([f'Stock {i}' for i in sampled_indices], fontsize=10)

        self.ax2.xaxis_date()
        self.ax2.xaxis.set_major_locator(mdates.YearLocator())
        self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(self.ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)

        plt.draw()
        plt.pause(0.001)


    def save_fig(self, filename):
        self.fig.savefig(filename, bbox_inches='tight')
