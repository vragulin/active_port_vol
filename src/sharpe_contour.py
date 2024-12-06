import numpy as np
import matplotlib.pyplot as plt

# Parameters
return_value = 0.064
mkt_risk = 0.16
fees_range = np.linspace(-0.02, 0.02, 101)  # Fees from 0% to 2%
active_risk_range = np.linspace(0, 0.20, 100)  # Active risk from 0% to 20%

# Create a meshgrid for fees and active risk
fees, active_risk = np.meshgrid(fees_range, active_risk_range)

# Calculate the ratio
ratio = (return_value - fees) / (np.sqrt(mkt_risk**2) + (active_risk**2))

# Plot heatmap
plt.figure(figsize=(10, 6))
heatmap = plt.pcolormesh(fees, active_risk, ratio, shading='auto', cmap='cool')
plt.colorbar(heatmap, label='Reward/Risk')

# Plot contour lines
contour = plt.contour(fees, active_risk, ratio, levels=10, colors='navy')
plt.clabel(contour, inline=True, fontsize=8, fmt='%1.2f')

# Add a vertical line at x = 0
plt.axvline(x=0, color='black', linestyle=':', linewidth=1)

# Add titles and labels
plt.title('Sharpe Ratio for Different Levels of Net Fees and Active Risk\n'
          'Contour Plot and Heatmap, Net Fees = Fees - Alpha')
plt.xlabel('Fees - Alpha (%)')
plt.ylabel('Active Risk (%)')

# Format the x and y labels to show percentage
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: '{:.1f}%'.format(val * 100)))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: '{:.1f}%'.format(val * 100)))

plt.show()
