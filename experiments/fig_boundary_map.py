"""Figure: R_{q,q}(x) - 1 against x for four levels q (from results/boundary_map.csv).
  python experiments/fig_boundary_map.py   -> results/fig_boundary_map.png
"""
import matplotlib
import pandas as pd

from _common import RESULTS

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

COLS = {.80: '#86b6ef', .85: '#3987e5', .90: '#1c5cab', .95: '#0d366b'}   # ordinal blue ramp
INK, MUTED, AXIS, GRID, SURF = '#1f1e1b', '#5c5b55', '#b9b8b1', '#ecebe6', '#fcfcfb'

d = pd.read_csv(RESULTS / 'boundary_map.csv')
fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.axhline(0, color='#8a8983', lw=1)
for q, g in d.groupby('q'):
    g = g.sort_values('x'); y = 100 * (g.R - 1)
    ax.plot(g.x, y, color=COLS[q], lw=2, label=f'q = {q:.2f}')
    i = int(y.values.argmax())
    ax.plot(g.x.values[i], y.values[i], 'o', ms=5, color=COLS[q], mec=SURF, mew=1.5)
ax.set_xscale('log'); ax.set_xlim(1e-4, 0.045); ax.set_ylim(-1.0, 0.45)
ax.text(1.2e-4, 0.37, 'above 0: widen (noisy calibration can undercover)', fontsize=8, color=MUTED)
ax.text(1.2e-4, -0.92, 'below 0: shrinking allowed', fontsize=8, color=MUTED)
ax.set_xlabel('noise-to-threshold variance ratio  x = D / t²', fontsize=9, color=INK)
ax.set_ylabel('sharp latent radius − 1  (%)', fontsize=9, color=INK)
ax.set_title('Log-concave latent laws: radius needed for latent coverage q', fontsize=10,
             color=INK, loc='left')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, labelsize=8); ax.grid(True, which='major', color=GRID, lw=.6)
ax.legend(frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(0.0, 0.08), labelcolor=INK)
fig.tight_layout(); fig.savefig(RESULTS / 'fig_boundary_map.png', facecolor=SURF)
