import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
import baggins as bgs
import ketjugw
import figure_config


bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot trajectory example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    default="INFO",
    choices=bgs.VERBOSITY,
    dest="verbosity",
    help="set verbosity level",
)
args = parser.parse_args()

SL = bgs.setup_logger("script", args.verbosity)


ketjufiles = bgs.utils.get_ketjubhs_in_dir(
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick"
)

fig, ax = plt.subplots()
fig.set_figheight(0.8 * fig.get_figheight())
inset_window = 0.9
axins = ax.inset_axes(
    [0.52, 0.03, 0.48, 0.48],
    xlim=(-inset_window, inset_window),
    ylim=(-inset_window - 0.2, inset_window - 0.2),
    xticklabels=[],
    yticklabels=[],
    xticks=[],
    yticks=[],
)
ax.set_xlabel(r"$x/\mathrm{kpc}$")
ax.set_ylabel(r"$z/\mathrm{kpc}$")

xcom = None
step = 100
cmap = sns.cubehelix_palette(start=0.5, rot=-0.75, hue=0.9, reverse=True, as_cmap=True)
gradline_kwargs = {"ls": "-", "capstyle": "round"}

for i, kf in enumerate(ketjufiles):
    kv = i * 60
    if kv == 0:
        bh1, bh2 = ketjugw.load_hdf5(kf).values()
        bh = bh1 if len(bh1) > len(bh2) else bh2
        xcom = np.median(bh.x, axis=0)
    if kv == 780:
        SL.info(f"Doing {kv} km/s simulation")
        bh1, bh2 = ketjugw.load_hdf5(kf).values()
        bh = bh1 if len(bh1) > len(bh2) else bh2
        bh = bh[bh.t / bgs.general.units.Gyr < 0.75]
        glp = bgs.plotting.GradientLinePlot(ax=ax, cmap=cmap)
        glp.add_data(
            x=(bh.x[::step, 0] - xcom[0]) / bgs.general.units.kpc,
            y=(bh.x[::step, 2] - xcom[2]) / bgs.general.units.kpc,
            c=bh.t[::step] / bgs.general.units.Myr,
        )
        glp.plot(ax=ax, **gradline_kwargs)
        glp.plot(ax=axins, **gradline_kwargs)
glp.draw_arrow_on_series(ax, 0, 650)
for axi in (ax, axins):
    axi.set_aspect("equal")
ax.set_xlim(-15, 21)
ax.set_ylim(-18, 14)
ax.indicate_inset_zoom(axins, ec="k")

# add circle to show binary core radius
rb0 = 0.58
core_circle = Circle((0, 0), rb0, ec="k", ls=":", lw=1, fill=None)
axins.add_artist(core_circle)
text_angle = np.pi / 4
axins.text(
    rb0 * np.cos(text_angle),
    rb0 * np.cos(text_angle),
    r"$r_\mathrm{b,0}$",
    ha="left",
    va="bottom",
)

# add scale bar
bgs.plotting.draw_sizebar(
    axins, 0.5, "kpc", borderpad=0.25, remove_ticks=False, size_vertical=0.01
)

glp.add_cbar(ax=ax, label=r"$t/\mathrm{Myr}$", shrink=0.7, extend="max")


bgs.plotting.savefig(figure_config.fig_path("trajectory.pdf"), force_ext=True)
