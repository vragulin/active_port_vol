import numpy as np
import matplotlib.pyplot as plt

# Global Parameters
line_colors = ['blue', 'red']
# line_styles = ['-', '--']
line_styles = ['-', '-']
fill_colors = ['blue', 'red']


# colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

class Sim:
    def __init__(self, years=10, steps_per_year=12, risk_free=0.05, n_paths=10, risk_aversion=2):
        self.years = years
        self.steps_per_year = steps_per_year
        self.dt = 1 / self.steps_per_year
        self.steps = self.steps_per_year * self.years
        self.n_paths = n_paths
        self.risk_free = risk_free
        self.risk_aversion = risk_aversion
        self.dz = np.random.normal(loc=0, scale=1, size=(self.steps, self.n_paths))


class Asset:
    def __init__(self, sharpe_ratio, volatility=0.2, label='Portfolio'):
        self.sharpe_ratio = sharpe_ratio
        self.volatility = volatility
        self.lxret = self.volatility * self.sharpe_ratio - 0.5 * self.volatility ** 2  # log return over risk-free rate
        self.xret = self.volatility * self.sharpe_ratio  # arithmetic return over risk-free rate
        self.label = label
        self.rets = None
        self.cum_rets = None

    def simulate_portfolio(self, s: Sim):
        dt = s.dt

        self.w = min(1, self.sharpe_ratio / self.volatility / s.risk_aversion)  # Merton portfolio weight
        asset_rets = (self.lxret + s.risk_free) * dt + self.volatility * np.sqrt(dt) * s.dz
        port_rets = self.w * asset_rets + (1 - self.w) * s.risk_free * dt
        cum_rets = np.ones((s.steps + 1, s.n_paths))
        cum_rets[1:, :] = np.exp(np.cumsum(port_rets, axis=0))
        self.rets = port_rets
        self.cum_rets = cum_rets

    def plot_portfolio(self, s: Sim, ax, idx=0):
        lb = [25]
        ub = [75]
        alpha = [0.2]
        years = np.arange(len(self.cum_rets)) / s.steps_per_year

        n = len(self.cum_rets)
        mean_returns = np.mean(self.cum_rets, axis=1)
        plt
        ax.plot(years, mean_returns, label=f'{self.label}: w={self.w * 100:.0f}%, E[final wealth]={mean_returns[-1]:.2f}',
                color=line_colors[idx], linestyle=line_styles[idx])

        for i in range(len(lb)):
            ax.fill_between(years,
                            np.percentile(self.cum_rets, lb[i], axis=1),
                            np.percentile(self.cum_rets, ub[i], axis=1),
                            alpha=alpha[i], color=fill_colors[idx])


if __name__ == '__main__':

    # Set up the environment
    s = Sim(years=10, steps_per_year=12, risk_free=0.05, n_paths=1000, risk_aversion=1)

    # Portfolio parameters
    labels = ['Market', 'Active']
    sharpe_ratios = np.array([0.4, 0.3])  # Example Sharpe ratios for two stocks
    vols = [0.16, 0.3]
    rets = [sharp * vol for sharp, vol in zip(sharpe_ratios, vols)]
    # Plotting the fan chart
    plt.rc('font', family='monospace')
    fig, ax = plt.subplots(figsize=(10, 6))
    suptitle_string : str = f'Simulated Portfolio Returns, risk-aversion={s.risk_aversion}\n'
    for i in range(len(sharpe_ratios)):
        suptitle_string += (f'{labels[i]}: Return={rets[i]*100:.1f}%, Vol={vols[i] * 100:.0f}, '
                            f'Sharpe={sharpe_ratios[i]:.1f}\n')
    plt.suptitle(suptitle_string)

    percentiles = [5, 50, 95]
    a = []
    for i in range(len(sharpe_ratios)):
        a.append(Asset(sharpe_ratios[i], vols[i], label=labels[i]))
        a[i].simulate_portfolio(s)
        a[i].plot_portfolio(s, ax=ax, idx=i)

    # Add title and legend
    ax.set_xlabel('Years')
    ax.set_ylabel('Cumulative Returns')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()
