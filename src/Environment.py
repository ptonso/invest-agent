import numpy as np
import pandas as pd
import torch

class PortfolioEnv:
    def __init__(self, stock_data_csv, ipca_csv, window_size, n_stocks=-1, baseline_flag=False):
        stock_data = pd.read_csv(stock_data_csv, parse_dates=['Date'])
        stock_data.set_index('Date', inplace=True)

        if n_stocks == -1:
            self.stock_data = stock_data
        else:
            self.stock_data = stock_data.iloc[:, :n_stocks].copy()
        self.stock_data.fillna(0, inplace=True)
        
        self.ipca_data = pd.read_csv(ipca_csv, parse_dates=['Date'])
        self.ipca_data.set_index('Date', inplace=True)

        self.baseline_flag = baseline_flag

        self.window_size = window_size
        self.cumulative_wallet_value = 1  # Starts with 100% initial investment
        self.cumulative_baseline = 1
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.portfolio_allocation = torch.zeros(self.stock_data.shape[1])
        self.portfolio_allocation[0] = 1.  # All initial investment in the first asset
        
        self.cumulative_wallet_value = 1.
        self.cumulative_baseline = 1.

        self.initial_prices = torch.tensor(self.stock_data.iloc[self.current_step].values, dtype=torch.float32)
        self.portfolio_value = 1

        return self._get_state()

    def _get_state(self):
        stock_window = self.stock_data.iloc[self.current_step - self.window_size:self.current_step].values
        state = torch.tensor(stock_window.flatten(), dtype=torch.float32)
        return state

    def compute_portifolio_var(self, action, next_changes, inflation_rate):
        portifolio_var = torch.dot(action, next_changes)
        real_portifolio_var = (1 + portifolio_var) / (1 + inflation_rate) - 1
        return real_portifolio_var

    def compute_portifolio_reward(self, portifolio_value, allocation):
        diversification_penalty = torch.sum(allocation ** 2)
        
        lambda_penalty=0.01
        
        reward = portifolio_value = lambda_penalty * diversification_penalty
        reward = portifolio_value
        return reward
        

    def step(self, action):
        assert torch.isclose(torch.sum(action), torch.tensor(1.), atol=1e-4), f"Action must sum to 1, but got {np.sum(action)}"

        if self.current_step + 1 >= len(self.stock_data)-self.window_size:
            done = True
            return self._get_state(), 0, done

        next_changes = torch.tensor(self.stock_data.iloc[self.current_step + 1].values, dtype=torch.float32)
        inflation_rate = torch.tensor(self.ipca_data.iloc[self.current_step + 1].values[0], dtype=torch.float32)
    
        real_portfolio_var = self.compute_portifolio_var(action, next_changes, inflation_rate)
        self.cumulative_wallet_value *= (1 + real_portfolio_var)

        # reward = real_portfolio_var
        reward = self.compute_portifolio_reward(real_portfolio_var, action)
        reward = reward.item()


        if self.baseline_flag:
            n_assets = len(action)
            baseline_action = torch.full_like(action, 1.0 / n_assets)
            real_portfolio_var = self.compute_portifolio_var(baseline_action, next_changes, inflation_rate)
            self.cumulative_baseline *= (1 + real_portfolio_var)

        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - self.window_size - 1
        self.portfolio_allocation = action

        return self._get_state(), reward, done

    def get_baseline_cumulative_value(self):
        return self.cumulative_baseline

    def get_cumulative_value(self):
        return float(self.cumulative_wallet_value)
    
    def get_current_date(self):
        return self.stock_data.index[self.current_step]
    
    def get_window_date_string(self):
        start_date = self.stock_data.index[self.current_step - self.window_size]
        end_date = self.stock_data.index[self.current_step - 1]
        return f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"

    def get_state_dataframe(self, window_date=None):
        if window_date is None:
            # Use current window
            start_idx = self.current_step - self.window_size
            end_idx = self.current_step
            df = self.stock_data.iloc[start_idx:end_idx]
        else:
            start_str, end_str = window_date.split('_')
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)
            df = self.stock_data.loc[start_date:end_date]
        return df

    def render(self):
        start_date = self.stock_data.index[self.current_step - self.window_size]
        end_date = self.stock_data.index[self.current_step - 1]
        print(f"Current Step: {self.current_step-30}")
        print(f"Date Range: {start_date.date()} to {end_date.date()}")
        print(f"Portfolio Allocation: {[round(k, 4) for k in self.portfolio_allocation.tolist()]}")
        print(f"Cumulative Wallet Value: {self.cumulative_wallet_value:.4f}")

