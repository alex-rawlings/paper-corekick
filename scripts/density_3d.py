import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs
import figure_config


parser = argparse.ArgumentParser(
    description="Plot 3D density",
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

bgs.plotting.check_backend()

snapfiles = bgs.utils.read_parameters(
    os.path.join(
        bgs.HOME,
        "projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml",
    )
)

fig, ax = plt.subplots(1, 1)

r_edges = np.geomspace(5e-2, 30, 100)
r_centres = bgs.mathematics.get_histogram_bin_centres(r_edges)
ball_mask = pygad.BallMask(r_edges[-1])

# create the colour scale
vkcols = figure_config.VkickColourMap()


for i, (k, v) in enumerate(snapfiles["snap_nums"].items()):
    if v is None:
        continue
    snapfile = os.path.join(
        snapfiles["parent_dir"], f"kick-vel-{k.lstrip('v')}/output/snap_{v:03d}.hdf5"
    )
    SL.debug(f"Reading: {snapfile}")
    snap = pygad.Snapshot(snapfile, physical=True)
    xcom = pygad.analysis.shrinking_sphere(
        snap.stars, pygad.analysis.center_of_mass(snap.stars[ball_mask]), 30
    )
    trans = pygad.Translation(-xcom)
    trans.apply(snap, total=True)

    dens = pygad.analysis.profile_dens(snap.stars, "mass", r_edges=r_edges)
    mask = dens > 0
    ax.loglog(
        r_centres[mask], dens[mask], ls="-", c=vkcols.get_colour(float(k.lstrip("v")))
    )

    # conserve memory
    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()

# add the r^-1 scaling line
xline = np.geomspace(2, 12, 4)
yline = ax.loglog(xline, 1e10 * xline**-1, c="k", lw=1)
ax.text(5, 3e9, r"$\propto r^{-1}$", ha="left")

ax.set_xlabel(r"$r/\mathrm{kpc}$")
ax.set_ylabel(r"$\rho(r)/(\mathrm{M}_\odot\,\mathrm{kpc}^{-3})$")
vkcols.make_cbar(ax)
bgs.plotting.savefig(figure_config.fig_path("density_3d.pdf"), fig=fig, force_ext=True)
plt.show()
