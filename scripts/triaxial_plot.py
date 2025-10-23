import argparse
import os.path
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import baggins as bgs
import figure_config


parser = argparse.ArgumentParser(
    description="Plot triaxiality of merger remnants given triaxial data",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-c", "--combine", action="store_true", dest="combine", help="combine datasets"
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

bgs.plotting.check_backend()

data_path = figure_config.data_path("triaxiality")

if args.combine:
    pickle_files = bgs.utils.get_files_in_dir(data_path, ext=".pickle")
    pickle_files = [p for p in pickle_files if "triax_v" in p]
    try:
        assert pickle_files
    except AssertionError:
        SL.exception(
            "No pickle files found with correct string pattern!", exc_info=True
        )
        raise
    data = {}
    for p in pickle_files:
        k = os.path.splitext(os.path.basename(p))[0].replace("triax_", "")
        data[k] = bgs.utils.load_data(p)
    bgs.utils.save_data(data, os.path.join(data_path, "triax_core-study.pickle"))
else:
    data = bgs.utils.load_data(os.path.join(data_path, "triax_core-study.pickle"))

# set up figure
fig, ax = plt.subplot_mosaic(
    """
    A
    B
    C
    C
    """,
    sharex=True,
)
fig.set_figheight(fig.get_figheight() * 1.5)
ax["A"].sharey(ax["B"])
vkcols = figure_config.VkickColourMap()

for k, v in data.items():
    if k == "__githash" or k == "__script":
        continue
    vv = v["ratios"]
    c = vkcols.get_colour(float(k[1:]))
    ax["A"].plot(
        vv["r"][0], gaussian_filter1d(vv["ba"][0], 2, mode="nearest"), c=c, ls="-"
    )
    ax["B"].plot(
        vv["r"][0], gaussian_filter1d(vv["ca"][0], 2, mode="nearest"), c=c, ls="-"
    )
    ax["C"].plot(
        vv["r"][0],
        gaussian_filter1d(
            (1 - vv["ba"][0] ** 2) / (1 - vv["ca"][0] ** 2), 2, mode="nearest"
        ),
        c=c,
        ls="-",
    )

# add colour bar and other labels
vkcols.make_cbar(list(ax.values()))

for k, lab in zip("ABC", (r"$b/a$", r"$c/a$", r"$T$")):
    ax[k].set_xscale("log")
    ax[k].set_ylabel(lab)
ax["C"].set_xlabel(r"$r/\mathrm{kpc}$")

# add some text for the T plot
ax["C"].set_ylim(0, 1)
ax["C"].axhline(0.5, c="k", lw=1, ls=":", zorder=0.5)
ax["C"].text(10, 0.55, r"$\mathrm{prolate}$", va="bottom")
ax["C"].text(10, 0.45, r"$\mathrm{oblate}$", va="top")

bgs.plotting.savefig(figure_config.fig_path("triaxiality.pdf"), force_ext=True)
