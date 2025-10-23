import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import dask
from tqdm.dask import TqdmCallback
import baggins as bgs
import figure_config


bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot beta profile and box-tube ratio",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-e", "--extract", help="extract orbit data", action="store_true", dest="extract"
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


@dask.delayed
def dask_extractor(orbitcl, vkey, mergemask):
    """
    Helper function to extract box tube ratio using dask parallelism

    Parameters
    ----------
    orbitcl : str, path-like
        orbit classifiction file
    vkey : str
        kick velocity key to read core data
    mergemask : bgs.analysis.MergeMask
        how to merge different orbits

    Returns
    -------
    ratio_med : np.array
        median values of box tube ratio
    ratio_err : np.array
        lower and upper errors of box tube ratio
    """
    # construct the classifier
    classifier = bgs.analysis.OrbitClassifier(orbitcl, mergemask=mergemask)

    rb = np.nanmedian(core_data["rb"][vkey].flatten())
    Nbox = classifier.family_size_in_radius("box", rb)
    Ntube = classifier.family_size_in_radius("tube", rb)
    return rb, Nbox, Ntube


orbitfilebases = [
    d.path
    for d in os.scandir(
        "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/orbit_analysis"
    )
    if d.is_dir() and "kick" in d.name
]
orbitfilebases.sort()
core_data = bgs.utils.load_data(
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/core-kick.pickle"
)
data_file = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/box_tube_ratio.pickle"

mergemask = bgs.analysis.MergeMask.make_box_tube_mask()

if args.extract:
    data = {"vkick": [], "rb": [], "Nbox": [], "Ntube": []}
    dask_res = []
    for j, orbitfilebase in enumerate(orbitfilebases):
        orbitcl = bgs.utils.get_files_in_dir(orbitfilebase, ext=".cl", recursive=True)[
            0
        ]
        try:
            vkey = f"{orbitfilebase.split('/')[-1].split('-')[-1]}"
            # test that we have a valid key
            core_data["rb"][vkey]
            vkick = float(vkey)
            # store kick velocities to data
            data["vkick"].append(vkick)
        except KeyError:
            SL.debug(f"No key for {vkey}")
            break
        SL.info(f"Reading {orbitcl}")

        # parallel delayed computation
        dask_res.append(dask_extractor(orbitcl, vkey, mergemask))
    # compute
    with TqdmCallback(desc="Computing samples"):
        dask_res = dask.compute(dask_res)

    # store data
    for i, dkey in enumerate(("rb", "Nbox", "Ntube")):
        data[dkey] = [r[i] for r in dask_res[0]]
    bgs.utils.save_data(data, data_file, exist_ok=True)
else:
    data = bgs.utils.load_data(data_file)

# plot 1: mass in boxes and tubes within core
stellar_mass = 5e4
norm_factor = 1e8
fig, ax = plt.subplots()
ax.set_xlabel(
    f"$M_{{\star,\mathrm{{box}}}}/(10^{int(np.log10(norm_factor))}\, \mathrm{{M}}_\odot)$"
)
ax.set_ylabel(
    f"$M_{{\star,\mathrm{{tube}}}}/(10^{int(np.log10(norm_factor))}\, \mathrm{{M}}_\odot)$"
)
vkcols = figure_config.VkickColourMap()

# plot data
for vk, Nb, Nt in zip(data["vkick"], data["Nbox"], data["Ntube"]):
    ax.scatter(
        stellar_mass * Nb / norm_factor,
        stellar_mass * Nt / norm_factor,
        color=vkcols.get_colour(vk),
        **figure_config.marker_kwargs,
    )
    SL.debug(
        f"{int(vk):04d} has {stellar_mass*Nb:.2e} Msol in boxes and {stellar_mass*Nt:.2e} Msol in tubes."
    )
xlims = ax.get_xlim()
ylims = ax.get_ylim()

# plot guidelines
guide_kwargs = {"alpha": 0.4, "ls": ":", "zorder": 0.5, "lw": 1, "c": "k"}
x = stellar_mass * np.array([1e2, 1e6]) / norm_factor
labels = []
rotation = []
for i, grad in enumerate((0.5, 1, 2), start=1):
    ax.plot(x, grad * x, **guide_kwargs)
    labels.append(f"$M_{{\star,\mathrm{{tube}}}}={grad} M_{{\star,\mathrm{{box}}}}$")
    rotation.append(np.arctan(grad) * 180 / np.pi)
fkwargs = {
    "fontsize": "small",
    "color": "k",
    "alpha": guide_kwargs["alpha"],
    "va": "center",
    "ha": "center",
    "rotation_mode": "anchor",
    "transform_rotates_text": True,
}
ax.text(7.5, 3.5, labels[0], rotation=rotation[0], **fkwargs)
ax.text(4.6, 5, labels[1], rotation=rotation[1], **fkwargs)
ax.text(2.1, 5, labels[2], rotation=rotation[2], **fkwargs)
ax.set_xlim(xlims)
ax.set_ylim(ylims)
vkcols.make_cbar(ax=ax)

bgs.plotting.savefig(figure_config.fig_path("orbit_bt.pdf"), force_ext=True)
