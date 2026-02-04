import numpy as np
import matplotlib.pyplot as plt
#
# mini simulation to plot equifinality and show how it would look on GLUE
# 
parameter1 = np.linspace(0.02, 0.08, 150)
paramater2 = np.linspace(0.1, 0.8, 150)
P1, P2 = np.meshgrid(parameter1, paramater2)

nse = (
    0.75 * np.exp(-((P1 - 0.04)**2 / 0.0008 + (P2 - 0.4)**2 / 0.05)) +
    0.72 * np.exp(-((P1 - 0.06)**2 / 0.0006 + (P2 - 0.25)**2 / 0.04)) +
    0.68 * np.exp(-((P1 - 0.035)**2 / 0.0007 + (P2 - 0.55)**2 / 0.06))
)

time = np.arange(0, 365, 1)
ObsDischarge = 50 + 30 * np.sin(2 * np.pi * time / 365) + 5 * np.random.normal(0, 1, len(time))
ObsDischarge = np.maximum(ObsDischarge, 5)
ensemblen = 100
PredEnsemble = np.zeros((ensemblen, len(time)))
for i in range(ensemblen):
    phase = np.random.uniform(0, 2*np.pi)
    amplitude = np.random.uniform(20, 40)
    trend = np.random.uniform(45, 55)
    PredEnsemble[i, :] = trend + amplitude * np.sin(2 * np.pi * time / 365 + phase) + np.random.normal(0, 2, len(time))
    PredEnsemble[i, :] = np.maximum(PredEnsemble[i, :], 5)
p5 = np.percentile(PredEnsemble, 5, axis=0)
p50 = np.percentile(PredEnsemble, 50, axis=0)
p95 = np.percentile(PredEnsemble, 95, axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

contourf = ax1.contourf(P1, P2, nse, levels=np.linspace(0.5, 0.78, 10), cmap='RdYlGn', alpha=0.9)
ax1.contour(P1, P2, nse, levels=[0.65], colors='black', linewidths=3)
thresh = nse >= 0.65
samplei = np.random.choice(np.sum(thresh), size=300, replace=False)
samplem = np.zeros(np.sum(thresh), dtype=bool)
samplem[samplei] = True
ax1.scatter(P1[thresh][samplem], P2[thresh][samplem], c='navy', s=8, alpha=0.6, label='Acceptable fits')
cbar1 = plt.colorbar(contourf, ax=ax1)
cbar1.set_label('Model Performance', fontsize=12)
ax1.set_xlabel('Parameter 1', fontsize=12, fontweight='bold')
ax1.set_ylabel('Parameter 2', fontsize=12, fontweight='bold')
ax1.set_title('(a) Multiple Parameter Sets Fit Data Equally Well', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=11)

ax2.fill_between(time, p5, p95, alpha=0.3, color='steelblue', label='Uncertainty range')
ax2.plot(time, p50, 'b-', linewidth=2.5, label='Average prediction')
ax2.plot(time, ObsDischarge, 'r-', linewidth=1.5, label='Observations', alpha=0.8)
ax2.set_xlabel('Time (days)', fontsize=12, fontweight='bold')
ax2.set_ylabel('River Discharge (m³/s)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Prediction Uncertainty from Multiple Parameter Sets', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=11)
ax2.set_xlim(0, 365)

plt.tight_layout()
plt.savefig('equifinalityfig.png', dpi=300, bbox_inches='tight')
plt.show()
