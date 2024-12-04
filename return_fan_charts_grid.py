import numpy as np
import matplotlib.pyplot as plt

# Global Parameters
line_colors = ['blue', 'red']
line_styles = ['-', '--']
fill_colors = ['blue', 'red']
N_PATHS = 10_000


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

    def simulate_portfolio(self, s: Sim, no_leverage: bool = True):
        dt = s.dt
        w_merton = self.sharpe_ratio / self.volatility / s.risk_aversion  # Merton portfolio weight
        self.w = min(1, w_merton) if no_leverage else min(1.5, w_merton)
        asset_rets = (self.lxret + s.risk_free) * dt + self.volatility * np.sqrt(dt) * s.dz
        port_rets = self.w * asset_rets + (1 - self.w) * s.risk_free * dt
        cum_rets = np.ones((s.steps + 1, s.n_paths))
        cum_rets[1:, :] = np.exp(np.cumsum(port_rets, axis=0))
        self.rets = port_rets
        self.cum_rets = cum_rets
        self.port_ret = self.w * self.xret + s.risk_free
        self.port_ret_ra = self.port_ret - 0.5 * s.risk_aversion * (self.w * self.volatility) ** 2

    def plot_portfolio(self, s: Sim, ax, idx=0, no_leverage: bool = True):
        lb = [25]
        ub = [75]
        alpha = [0.2]
        years = np.arange(len(self.cum_rets)) / s.steps_per_year
        mean_returns = np.mean(self.cum_rets, axis=1)
        plt.rc('font', family='monospace')
        color = line_colors[idx] if no_leverage else 'black'
        linestyle = line_styles[idx] if no_leverage else ':'
        ax.plot(years, mean_returns,
                label=f'{self.label}:w={self.w * 100:3.0f}%, E[W(T)]={mean_returns[-1]:.1f}, '
                      + f'r={self.port_ret * 100:.1f}%, ' + r'$r_{ra}$' + f'={self.port_ret_ra * 100:.1f}%',
                color=color, linestyle=linestyle)
        if no_leverage:
            for i in range(len(lb)):
                ax.fill_between(years,
                                np.percentile(self.cum_rets, lb[i], axis=1),
                                np.percentile(self.cum_rets, ub[i], axis=1),
                                alpha=alpha[i], color=fill_colors[idx])


if __name__ == '__main__':
    risk_aversions = [1, 2.5, 5, 10]
    labels = ['Market ', 'Active ']
    sharpe_ratios = np.array([0.44, 0.4])
    vols = [0.18, 0.3]
    rets = [sharp * vol for sharp, vol in zip(sharpe_ratios, vols)]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    plt.rc('font', family='monospace')
    suptitle_string = f'Active v. Market Optimal Portfolio Returns vs Investor Risk-Aversion\n'
    for j in range(len(sharpe_ratios)):
        suptitle_string += (f'{labels[j]}: Sharpe={sharpe_ratios[j]:.2f}, '
                            f'Return={rets[j] * 100:4.1f}%, Vol={vols[j] * 100:.0f}%\n')
    plt.rc('font', family='monospace')
    plt.suptitle(suptitle_string)

    axs = axs.flatten()
    for i, risk_aversion in enumerate(risk_aversions):
        s = Sim(years=10, steps_per_year=12, risk_free=0.05, n_paths=N_PATHS, risk_aversion=risk_aversion)
        for j in range(len(sharpe_ratios)):
            a = Asset(sharpe_ratios[j], vols[j], label=labels[j])
            a.simulate_portfolio(s)
            a.plot_portfolio(s, ax=axs[i], idx=j)
        if i in [0]:
            lev_mkt = Asset(sharpe_ratios[0], vols[0], label='Lev Mkt')
            lev_mkt.simulate_portfolio(s, no_leverage=False)
            lev_mkt.plot_portfolio(s, ax=axs[i], idx=j, no_leverage=False)
        axs[i].set_title(f'Relative Risk Aversion={risk_aversion}')
        axs[i].set_xlabel('Years')
        axs[i].set_ylabel('Cumulative Returns')
        axs[i].legend(loc='upper left')

    plt.tight_layout()
    plt.show()
