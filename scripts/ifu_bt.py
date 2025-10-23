import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import baggins as bgs
import pygad
import figure_config


bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot IFU maps", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(dest="vel", help="Velocity to plot")
parser.add_argument(
    "-e", "--extract", help="extract data", action="store_true", dest="extract"
)
parser.add_argument(
    "-I", "--Inertia", action="store_true", help="align with inertia", dest="inertia"
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

suffix = "_I" if args.inertia else ""
data_file = os.path.join(
    figure_config.reduced_data_dir, f"IFU_bt/ifu_bt_{args.vel}{suffix}.pickle"
)

if args.extract:
    # XXX set the fraction of rhalf within which IFU maps are created for
    rhalf_factor = 0.25
    seeing = {"num": 25, "sigma": 0.3}

    # get the snapshot, all this is taken from IFU.py
    snapfiles = bgs.utils.read_parameters(
        os.path.join(
            bgs.HOME,
            "projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml",
        )
    )
    snapshots = dict()
    for k, v in snapfiles["snap_nums"].items():
        if v is None:
            continue
        snapshots[k] = os.path.join(
            snapfiles["parent_dir"],
            f"kick-vel-{k.lstrip('v')}/output/snap_{v:03d}.hdf5",
        )
    snap = pygad.Snapshot(snapshots[f"v{args.vel}"], physical=True)
    centre = pygad.analysis.shrinking_sphere(
        snap.stars, pygad.analysis.center_of_mass(snap.stars), 30
    )
    # move to CoM frame
    pygad.Translation(-centre).apply(snap, total=True)
    pre_ball_mask = pygad.BallMask(30)
    vcom = pygad.analysis.mass_weighted_mean(snap.stars[pre_ball_mask], "vel")
    pygad.Boost(-vcom).apply(snap, total=True)

    rhalf = pygad.analysis.half_mass_radius(snap.stars[pre_ball_mask])
    extent = rhalf_factor * rhalf
    n_regular_bins = int(2 * extent / pygad.UnitScalar(0.04, "kpc"))

    if args.inertia:
        pygad.Translation(-centre).apply(snap, total=True)
        pygad.analysis.orientate_at(snap.stars[pre_ball_mask], "red I", total=True)

    box_mask = pygad.ExprMask(f"abs(pos[:,1]) <= {extent}") & pygad.ExprMask(
        f"abs(pos[:,2]) <= {extent}"
    )

    # determine the correct orbit file
    orbitfilebase = [
        d.path
        for d in os.scandir(
            "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/orbit_analysis"
        )
        if d.is_dir() and "kick" in d.name and args.vel in d.name
    ][0]
    SL.info(f"Reading: {orbitfilebase}")
    orbitcl = bgs.utils.get_files_in_dir(orbitfilebase, ext=".cl", recursive=True)[0]

    # read in orbit classification data
    mergemask = bgs.analysis.MergeMask.make_box_tube_mask()
    classifier = bgs.analysis.OrbitClassifier(orbitcl, mergemask=mergemask)

    # all orbits
    voronoi_stats_all = bgs.analysis.voronoi_binned_los_V_statistics(
        x=snap.stars[box_mask]["pos"][:, 1],
        y=snap.stars[box_mask]["pos"][:, 2],
        V=snap.stars[box_mask]["vel"][:, 0],
        m=snap.stars[box_mask]["mass"],
        Npx=n_regular_bins,
        part_per_bin=2000 * seeing["num"],
        seeing=seeing,
    )

    # box orbits
    mask = pygad.IDMask(classifier.get_particle_ids_for_family("box")) & box_mask
    voronoi_stats_box = bgs.analysis.voronoi_binned_los_V_statistics(
        x=snap.stars[mask]["pos"][:, 1],
        y=snap.stars[mask]["pos"][:, 2],
        V=snap.stars[mask]["vel"][:, 0],
        m=snap.stars[mask]["mass"],
        Npx=n_regular_bins,
        part_per_bin=2000 * seeing["num"],
        seeing=seeing,
    )

    # tube orbits
    mask = pygad.IDMask(classifier.get_particle_ids_for_family("tube")) & box_mask
    voronoi_stats_tube = bgs.analysis.voronoi_binned_los_V_statistics(
        x=snap.stars[mask]["pos"][:, 1],
        y=snap.stars[mask]["pos"][:, 2],
        V=snap.stars[mask]["vel"][:, 0],
        m=snap.stars[mask]["mass"],
        Npx=n_regular_bins,
        part_per_bin=2000 * seeing["num"],
        seeing=seeing,
    )

    data = dict(
        voronoi_stats_all=voronoi_stats_all,
        voronoi_stats_box=voronoi_stats_box,
        voronoi_stats_tube=voronoi_stats_tube,
    )
    bgs.utils.save_data(data, data_file)
else:
    data = bgs.utils.load_data(data_file)

# now we will create three sets of IFU maps: one for all, one for tubes, one for boxes
gs_kwargs = {"height_ratios": [1, 0.1, 1, 1]}
fig, ax = plt.subplots(4, 4, sharex="all", sharey="all", gridspec_kw=gs_kwargs)
fig.set_figwidth(3 * fig.get_figwidth())
fig.set_figheight(1.5 * fig.get_figheight())

# set the second row to be invisible to get the right spacing
for axi in ax[1, :]:
    axi.set_visible(False)

ax[0, 0].text(0.1, 0.1, "all", ha="left", va="center", transform=ax[0, 0].transAxes)
ax[2, 0].text(0.1, 0.1, "box", ha="left", va="center", transform=ax[2, 0].transAxes)
ax[3, 0].text(0.1, 0.1, "tube", ha="left", va="center", transform=ax[3, 0].transAxes)
for axi in ax[-1, :]:
    axi.set_xlabel(r"$y/\mathrm{kpc}$")
for axi in ax[:, 0]:
    axi.set_ylabel(r"$z/\mathrm{kpc}$")
ax[0, 0].set_xlim(-2.6, 2.6)
ax[0, 0].set_ylim(-2.6, 2.6)

bgs.plotting.voronoi_plot(data["voronoi_stats_all"], ax=ax[0, :])


# determine the colour limits to use: consistent between box and tube plots
def _extreme_finder(k):
    vals = [
        np.max(np.abs(data[v][k])) for v in ["voronoi_stats_box", "voronoi_stats_tube"]
    ]
    return [np.min(vals), np.max(vals)]


clims = dict(
    V=[_extreme_finder("img_V")[1]],
    sigma=_extreme_finder("img_sigma"),
    h3=[_extreme_finder("img_h3")[1]],
    h4=[_extreme_finder("img_h4")[1]],
)

bgs.plotting.voronoi_plot(
    data["voronoi_stats_box"], ax=ax[2, :], clims=clims, desat=True
)

bgs.plotting.voronoi_plot(
    data["voronoi_stats_tube"], ax=ax[3, :], clims=clims, desat=True
)

# get the core radius
core_radius = np.nanmedian(
    bgs.utils.load_data(
        "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/core-kick.pickle"
    )["rb"][args.vel].flatten()
)
SL.debug(f"Using a core radius of {core_radius:.2e}")
# spin parameter
spin_func = bgs.analysis.lambda_R(data["voronoi_stats_all"])
for r in (1, 2):
    SL.info(f"Spin parameter at {r} core radii is {spin_func(r*core_radius):.2e}")
SL.info(f"Spin parameter at 2kpc is {spin_func(2):.2e}")

# add the core radius to all plots
for i, axi in enumerate(ax.flat):
    core_circle = Circle((0, 0), core_radius, fill=False, ec="k", ls="--")
    axi.add_artist(core_circle)

plt.subplots_adjust(left=0.03, right=0.95, top=0.97)
bgs.plotting.savefig(
    figure_config.fig_path(f"IFU_bt_{args.vel}{suffix}.pdf"), force_ext=True
)
