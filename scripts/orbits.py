import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import figure_config


parser = argparse.ArgumentParser(
    description="Plot orbit families, based on `script_freq.py`",
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

# create logger
SL = bgs.setup_logger("script", args.verbosity)

bgs.plotting.check_backend()


orbitfilebases = [
    d.path
    for d in os.scandir(
        "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/orbit_analysis"
    )
    if d.is_dir() and "kick" in d.name
]
orbitfilebases.sort()

# create the merge mask
mergemask = bgs.analysis.MergeMask()
mergemask.add_family("pi-box", [21, 24, 25], r"$\pi\mathrm{-box}$")
mergemask.add_family("boxlet", [1, 5, 9, 13, 17], r"$\mathrm{boxlet}$")
mergemask.add_family(
    "x-tube-in", [4, 8, 12, 16, 20], r"$\mathrm{inner\,}x\mathrm{-tube}$"
)
mergemask.add_family(
    "x-tube-out", [3, 7, 11, 15, 19], r"$\mathrm{outer\,}x\mathrm{-tube}$"
)
mergemask.add_family("z-tube", [2, 6, 10, 14, 18], r"$z\mathrm{-tube}$")
mergemask.add_family("rosette", [26], r"$\mathrm{rosette}$")
mergemask.add_family("irreg", [22], r"$\mathrm{irregular}$")
mergemask.add_family("unclass", [0, 23], r"$\mathrm{unclassified}$")

# figure 1: plots of different orbital families
fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
fig.set_figwidth(2 * fig.get_figwidth())
vkcols = figure_config.VkickColourMap()


# figure 2: plots of different kick velocities
fig2, ax2 = plt.subplots(5, 4, sharex=True, sharey=True)
fig2.set_figwidth(2 * fig2.get_figwidth())
fig2.set_figheight(2.5 * fig2.get_figheight())

for j, (axj, orbitfilebase) in enumerate(zip(ax2.flat, orbitfilebases)):
    try:
        orbitcl = bgs.utils.get_files_in_dir(orbitfilebase, ext=".cl", recursive=True)[
            0
        ]
        classifier = bgs.analysis.OrbitClassifier(orbitcl, mergemask=mergemask)
        classifier.radial_frequency(radbins=np.geomspace(0.2, 11, 11))
        rosette_mask = classifier.classids == mergemask.get_family("rosette")
        for dist, arr in zip(
            ("Apocentre", "Pericentre"), (classifier.apocenter, classifier.pericenter)
        ):
            SL.info(
                f"{dist} IQR for rosettes: {np.nanquantile(arr[rosette_mask], 0.25):.2e} - {np.nanquantile(arr[rosette_mask], 0.75):.2e} (median: {np.median(arr[rosette_mask]):.2e})"
            )
    except ValueError:
        # ongoing analysis
        SL.error(f"Unable to read {orbitfilebase}: skipping")
        # continue
    vkick = float(orbitfilebase.split("/")[-1].split("-")[-1])
    for i, axi in enumerate(ax.flat):
        axi.semilogx(
            classifier.meanrads,
            classifier.classfrequency[:, i],
            label=vkick,
            c=vkcols.get_colour(vkick),
            ls="-",
        )
        if mergemask.families[i] == "unclass":
            SL.debug(
                f"Skipping family '{mergemask.families[i]}' kick-vel stratified plot..."
            )
            continue
        axj.semilogx(
            classifier.meanrads,
            classifier.classfrequency[:, i],
            label=mergemask.labels[i],
        )
    axj.text(
        0.95,
        0.9,
        f"${vkick:.0f}\, \mathrm{{km}}\,\mathrm{{s}}^{{-1}}$",
        ha="right",
        va="center",
        transform=axj.transAxes,
    )

# for first figure:
# make axis labels nice
for i in range(ax.shape[0]):
    ax[i, 0].set_ylabel(r"$f_\mathrm{orbit}$")
for i in range(ax.shape[1]):
    ax[1, i].set_xlabel(r"$r/\mathrm{kpc}$")
for axi, label in zip(ax.flat, mergemask.labels):
    axi.text(0.05, 0.86, label, ha="left", va="center", transform=axi.transAxes)

# add the colour bar in the top right subplot, hiding that subplot
vkcols.make_cbar(ax[:, -1].flat)

# for second figure
for i in range(ax2.shape[0]):
    ax2[i, 0].set_ylabel(r"$f_\mathrm{orbit}$")
for i in range(ax2.shape[1]):
    ax2[-1, i].set_xlabel(r"$r/\mathrm{kpc}$")
bbox = ax2[-1, -1].get_position()
fig2.legend(
    *ax2[0, 0].get_legend_handles_labels(),
    loc="center left",
    bbox_to_anchor=(bbox.x0 + bbox.width / 4, bbox.y0 + bbox.height / 4),
)
ax2[-1, -1].axis("off")

bgs.plotting.savefig(figure_config.fig_path("orbits.pdf"), fig=fig, force_ext=True)
bgs.plotting.savefig(figure_config.fig_path("orbits2.pdf"), fig=fig2, force_ext=True)
