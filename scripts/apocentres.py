import argparse
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    from matplotlib import use

    use("Agg")
    import matplotlib.pyplot as plt
import baggins as bgs
import arviz as az
import figure_config

parser = argparse.ArgumentParser(
    description="Plot apocentre distribution",
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

orbitfilebases = [
    d.path
    for d in os.scandir(
        "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/orbit_analysis"
    )
    if d.is_dir() and "kick" in d.name
]
orbitfilebases.sort()

fig, ax = plt.subplots(1, 1)
vkcols = figure_config.VkickColourMap()

for obf in orbitfilebases:
    SL.debug(f"Reading {obf}")
    orbitcl = bgs.utils.get_files_in_dir(obf, ext=".cl", recursive=True)[0]
    kv = float(os.path.basename(obf).replace("kick-vel-", ""))
    res = bgs.analysis.orbits_radial_frequency(orbitcl, returnextra=True)
    mask = np.logical_and(
        res["apo"] < np.nanquantile(res["apo"], 0.975), res["apo"] > 0
    )
    # corresponds to extent of IFU maps
    mask = np.logical_and(mask, res["meanposrad"] < 3)
    az.plot_kde(
        res["apo"][mask],
        ax=ax,
        plot_kwargs={"lw": 2, "color": vkcols.get_colour(kv), "ls": "-"},
    )
vkcols.make_cbar(ax=ax)
ax.set_xlabel(r"$r_\mathrm{apo}/\mathrm{kpc}$")
ax.set_ylabel("PDF")

bgs.plotting.savefig(figure_config.fig_path("apo.pdf"), force_ext=True)
