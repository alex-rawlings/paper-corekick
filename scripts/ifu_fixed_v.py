import argparse
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from datetime import datetime
import pygad
import baggins as bgs
import figure_config


parser = argparse.ArgumentParser(
    description="Plot IFU maps", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "-e", "--extract", help="extract data", action="store_true", dest="extract"
)
parser.add_argument(
    "-kv", "--kickvel", type=int, help="kick velocity", dest="kv", default=900
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

h4_file = os.path.join(
    bgs.DATADIR,
    f"mergers/processed_data/core-paper-data/h4_fixed_vel/h4_{args.kv}.pickle",
)

bgs.plotting.check_backend()

if args.extract:
    snapfiles = bgs.utils.read_parameters(
        os.path.join(
            bgs.HOME,
            "projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml",
        )
    )

    seeing = {"num": 25, "sigma": 0.3}

    i0 = snapfiles["snap_nums"][f"v{args.kv:04d}"]
    snapfiles["snap_nums"] = np.linspace(0.2 * i0, i0, 5, dtype=int)
    snapfiles["snap_nums"] = np.concatenate(
        (snapfiles["snap_nums"], [np.ceil(a * i0) for a in [1.1, 1.2, 1.3, 1.5]])
    )
    snapfiles["snap_nums"] = np.array(snapfiles["snap_nums"], dtype=int)

    snapshots = {}
    for v in snapfiles["snap_nums"]:
        if v is None:
            continue
        snapshots[f"snap-{v:03d}"] = os.path.join(
            snapfiles["parent_dir"],
            f"kick-vel-{args.kv:04d}/output/snap_{v:03d}.hdf5",
        )

    # dict to store radial h4 profiles
    h4_vals = {"para": {}, "ortho": {}, "t": [], "tfid": None}

    for k, v in snapshots.items():
        SL.info(f"Creating IFU maps for {k}")
        tstart = datetime.now()

        try:
            snap = pygad.Snapshot(v, physical=True)
            if len(snap.bh) > 1:
                continue
            h4_vals["t"].append(bgs.general.convert_gadget_time(snap))
            if str(i0) in os.path.basename(v).split("_")[-1]:
                h4_vals["tfid"] = h4_vals["t"][-1]
        except OSError as e:
            SL.exception(e)
            continue

        centre = pygad.analysis.shrinking_sphere(
            snap.stars, pygad.analysis.center_of_mass(snap.stars), 30
        )
        # move to CoM frame
        pygad.Translation(-centre).apply(snap, total=True)
        pre_ball_mask = pygad.BallMask(30)
        vcom = pygad.analysis.mass_weighted_mean(snap.stars[pre_ball_mask], "vel")
        pygad.Boost(-vcom).apply(snap, total=True)

        rhalf = pygad.analysis.half_mass_radius(snap.stars[pre_ball_mask])
        extent = 0.25 * rhalf
        n_regular_bins = int(2 * extent / pygad.UnitScalar(0.04, "kpc"))

        SL.debug(f"IFU extent is {extent:.2f} kpc")
        SL.debug(f"Number of regular bins is {n_regular_bins}^2")

        # as the BH kick direction is always along the x-axis, use
        # this axis as the LOS parallel direction
        # try two orientations:
        # 1: LOS parallel with BH motion
        # 2: LOS perpendicular to BH motion
        for orientation, x_axis, LOS_axis in zip(("para", "ortho"), (1, 0), (0, 1)):
            SL.info(f"Doing {orientation} orientation...")
            box_mask = pygad.ExprMask(
                f"abs(pos[:,{x_axis}]) <= {extent}"
            ) & pygad.ExprMask(f"abs(pos[:,2]) <= {extent}")
            voronoi_stats = bgs.analysis.voronoi_binned_los_V_statistics(
                x=snap.stars[box_mask]["pos"][:, x_axis],
                y=snap.stars[box_mask]["pos"][:, 2],
                V=snap.stars[box_mask]["vel"][:, LOS_axis],
                m=snap.stars[box_mask]["mass"],
                Npx=n_regular_bins,
                part_per_bin=2000 * seeing["num"],
                seeing=seeing,
            )

            SL.debug(f"Completed binning in {datetime.now()-tstart}")

            h4_vals[orientation][k] = {}
            (
                h4_vals[orientation][k]["R"],
                h4_vals[orientation][k]["h4"],
            ) = bgs.analysis.radial_profile_velocity_moment(voronoi_stats, "h4")
            SL.debug(f"Length of data: {len(h4_vals['t'])}")

        # conserve memory
        snap.delete_blocks()
        del snap
        pygad.gc_full_collect()

    bgs.utils.save_data(h4_vals, h4_file)
else:
    SL.warning(
        "Reading in saved dataset for h4 analysis, IFU maps will not be recreated!"
    )
    h4_vals = bgs.utils.load_data(h4_file)
    SL.debug(f"Times of analysis: {[h4_vals['t']]}")

time_vals = np.array(h4_vals["t"]) - h4_vals["tfid"]

# plot h4 radial profiles
fig, ax = plt.subplots(2, 1, sharex="all", sharey="all")

cmapper, sm = bgs.plotting.create_offcentre_diverging(
    vmin=min(time_vals), vmax=max(time_vals), cmap="icefire"
)


# helper function for plotting
def plot_helper(axi, t, v):
    r = v["R"]
    h4 = v["h4"]
    r_bins = np.linspace(0, 3, 11)
    h4_med, *_ = scipy.stats.binned_statistic(r, h4, statistic="median", bins=r_bins)
    if np.abs(t) < 1e-10:
        plt_kwargs = {"marker": "o", "markevery": 2}
    else:
        plt_kwargs = {}
    axi.plot(
        bgs.mathematics.get_histogram_bin_centres(r_bins),
        h4_med,
        c=cmapper(t),
        ls="-",
        **plt_kwargs,
    )


for t, (kp, vp), (ko, vo) in zip(
    time_vals, h4_vals["para"].items(), h4_vals["ortho"].items()
):
    plot_helper(ax[0], t, vp)
    plot_helper(ax[1], t, vo)

# fig.suptitle(f"Kick velocity: {args.kv} km/s")
ax[-1].set_xlabel(r"$R/\mathrm{kpc}$")
ax[0].set_ylabel(r"$\langle h_4 \rangle\;\mathrm{(parallel)}$")
ax[1].set_ylabel(r"$\langle h_4 \rangle\;\mathrm{(orthogonal)}$")
plt.colorbar(sm, ax=ax.flat, label=r"$(t-t_\mathrm{settle})/\mathrm{Gyr}$")
bgs.plotting.savefig(
    figure_config.fig_path(f"h4_{args.kv}.pdf"),
    force_ext=True,
)
