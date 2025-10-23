# For calculating the (velocity) anisotropy parameter beta as a funtion of radius
import numpy as np
from numpy import nanmedian as med
import matplotlib.pyplot as plt
import pygad
import gc
import time
import pickle
import figure_config

start_time = time.time()


# Options
read_betas = 1  # read in previously computed beta profiles
show_fig = 1
save_fig = 1  # save output figure
save_betas = 1  # save beta profiles to output data file (only if read_betas == 0)
sample_size = (
    10000  # number of core radii sampled per simulation (only if read_file == 0)
)
x_axis = (
    "v"  # whether to plot betas as a function of kick velocity ("v") or radius ("r")
)

input_file = "beta_profiles_vkick_N10000.pickle"  # file containing input beta profiles if read_file == 1
output_file = (
    "beta_profiles_vkick.pdf"  # base filename for the output figure and data file
)

# fig_path = this_dir + "/../figures/" # directory for output figure
data_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/"  # directory for input core radii, input beta profiles and output data file
snap_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/"  # directory for input snapshots
core_file = "core-kick.pickle"  # file containing core radii values


# Kick velocities and corresponding snapshot numbers to use
vels = [
    "0000",
    "0060",
    "0120",
    "0180",
    "0240",
    "0300",
    "0360",
    "0420",
    "0480",
    "0540",
    "0600",
    "0660",
    "0720",
    "0780",
    "0840",
    "0900",
    "0960",
    "1020",
]
snaps = [2, 4, 4, 9, 9, 13, 31, 50, 70, 60, 73, 74, 100, 143, 193, 199, 240, 275]


# Compute polar angle given xyz position
def theta(x, y, z):
    return np.arccos(z / np.sqrt(x**2 + y**2 + z**2))


# Compute azimuthal angle given xy position
def phi(x, y):
    return np.sign(y) * np.arccos(x / np.sqrt(x**2 + y**2))


# Compute spherical velocities (v_r, v_theta, v_phi) for a given (sub)snapshot
def spherical_velocity(snapshot):
    x, y, z = (
        snapshot["pos"][:, 0],
        snapshot["pos"][:, 1],
        snapshot["pos"][:, 2],
    )
    vx, vy, vz = (
        snapshot["vel"][:, 0],
        snapshot["vel"][:, 1],
        snapshot["vel"][:, 2],
    )
    t, p = theta(x, y, z), phi(x, y)  # theta, phi arrays
    v_r = (
        np.sin(t) * np.cos(p) * vx + np.sin(t) * np.sin(p) * vy + np.cos(t) * vz
    )  # v_r
    v_t = (
        np.cos(t) * np.cos(p) * vx + np.cos(t) * np.sin(p) * vy - np.sin(t) * vz
    )  # v_theta
    v_p = -np.sin(p) * vx + np.cos(p) * vy  # v_phi

    return v_r, v_t, v_p


# Compute anisotropy parameter beta for a given (sub)snapshot
def beta(snapshot):
    v_r, v_t, v_p = spherical_velocity(snapshot)
    return 1 - (np.std(v_t) ** 2 + np.std(v_p) ** 2) / (2 * np.std(v_r) ** 2 + 1e-16)


# Compute anisotropy parameter beta profile for a given (sub)snapshot.
# Profile is computed in 2 bins with respect to r_split in [kpc], i.e. r <= r_split and r > r_split.
def beta_profile(snapshot, r_split):
    b_arr = np.full(2, np.nan)
    bin_counts = np.full(2, np.nan)

    # radial masks for r <= r_split and r > r_split
    m1, m2 = pygad.ExprMask("r <= " + str(r_split)), pygad.ExprMask(
        "r > " + str(r_split)
    )
    m = [m1, m2]
    for i in range(2):
        b_arr[i] = beta(snapshot[m[i]])
        bin_counts[i] = len(snapshot[m[i]]["r"])
    # Compute mean radii for the bins
    r_arr = np.array([np.mean(snapshot[m1]["r"]), np.mean(snapshot[m2]["r"])])

    # Return bin radii and corresponding beta values and bin counts
    return (
        r_arr,
        b_arr,
        bin_counts,
    )


# wrap main functionality into a function so we can call it elsewhere
def main(ax=None, read_betas=1):
    # Create random number generator
    rgen = np.random.default_rng(seed=88965)

    # Extract input core radii
    with open(data_path + core_file, "rb") as f:
        core_radii = pickle.load(f)["rb"]

    # Compute new beta profiles...
    if read_betas == 0:
        print("Computing beta profiles...", flush=True)
        extracted_data = {
            vel: dict(
                r_in=np.full(sample_size, np.nan),
                r_out=np.full(sample_size, np.nan),
                b_in=np.full(sample_size, np.nan),
                b_out=np.full(sample_size, np.nan),
            )
            for vel in vels
        }
        for i in range(len(vels)):
            step_time = time.time()
            run = "kick-vel-" + vels[i] + "/"
            print("---", run, "---", flush=True)

            snap = pygad.Snapshot(
                snap_path
                + run
                + "output/snap_"
                + "0" * (3 - len(str(snaps[i])))
                + str(snaps[i])
                + ".hdf5",
                physical=True,
            )

            # Find center using shrinking sphere on the stars
            center = pygad.analysis.shrinking_sphere(
                snap.stars, center=pygad.analysis.center_of_mass(snap.stars), R=30
            )

            # Shift coordinates of all particles so that the snap is centered on
            # the shrinking sphere result
            pygad.Translation(-center).apply(snap)

            this_core_radii = core_radii[vels[i]].flatten()
            core_sample = rgen.choice(
                this_core_radii, size=sample_size, replace=True
            )  # select core radius sample with rgen
            for j in range(len(core_sample)):
                radii, betas, bincount = beta_profile(
                    snap.stars, r_split=core_sample[j]
                )
                r_in, r_out, b_in, b_out = radii[0], radii[-1], betas[0], betas[-1]
                extracted_data[vels[i]]["r_in"][j] = radii[0]
                extracted_data[vels[i]]["r_out"][j] = radii[-1]
                extracted_data[vels[i]]["b_in"][j] = betas[0]
                extracted_data[vels[i]]["b_out"][j] = betas[-1]

            del snap
            gc.collect()
            pygad.gc_full_collect()
            print(time.time() - step_time, "s", flush=True)

        # Save beta profiles
        if save_betas == 1:
            print("Saving beta profiles...", flush=True)
            print(data_path + output_file + ".pickle", flush=True)
            pickle.dump(extracted_data, open(data_path + output_file + ".pickle", "wb"))

    # ...or read in previously computed beta profiles
    elif read_betas == 1:
        print("Reading beta profiles...", flush=True)
        print(data_path + input_file, flush=True)
        extracted_data = pickle.load(open(data_path + input_file, "rb"))

    else:
        raise RuntimeError("ERROR: incorrect read_file, select 0 or 1")

    if ax is None:
        # Initialize figure
        fig, ax = plt.subplots()
    ax.axhline(
        y=0, xmin=0, xmax=1, linestyle=":", color="k", zorder=0, lw=1
    )  # zero level on the background

    # Plot the profiles
    cols = figure_config.color_cycle_shuffled.by_key()["color"][:2]
    for i in range(len(vels)):
        r_in, r_out = extracted_data[vels[i]]["r_in"], extracted_data[vels[i]]["r_out"]
        b_in, b_out = extracted_data[vels[i]]["b_in"], extracted_data[vels[i]]["b_out"]
        r_err, b_err = [np.nanstd(r_in), np.nanstd(r_out)], [
            np.nanstd(b_in),
            np.nanstd(b_out),
        ]
        core_r = np.mean(core_radii[vels[i]].flatten())
        if x_axis == "r":
            ax.errorbar(
                [med(r_in) / core_r, med(r_out) / core_r],
                [med(b_in), med(b_out)],
                xerr=r_err,
                yerr=b_err,
                fmt=".",
                label=vels[i],
            )
        elif x_axis == "v":
            med_b_in = med(b_in)
            med_b_out = med(b_out)
            yerr_in = np.atleast_2d(
                np.array(
                    [
                        med_b_in - np.nanquantile(b_in, 0.25),
                        np.nanquantile(b_in, 0.75) - med_b_in,
                    ]
                )
            ).T
            yerr_out = np.atleast_2d(
                np.array(
                    [
                        med_b_out - np.nanquantile(b_out, 0.25),
                        np.nanquantile(b_out, 0.75) - med_b_out,
                    ]
                )
            ).T
            # r <= r_core
            b_in_p = ax.errorbar(
                int(vels[i]),
                med_b_in,
                yerr=yerr_in,
                fmt="o",
                capthick=1,
                ecolor=cols[0],
                mfc=cols[0],
                mec="k",
            )
            # r > r_core
            b_out_p = ax.errorbar(
                int(vels[i]),
                med_b_out,
                yerr=yerr_out,
                fmt="o",
                capthick=1,
                ecolor=cols[1],
                mfc=cols[1],
                mec="k",
            )
        else:
            raise RuntimeError("ERROR: incorrect x_axis, select 'r' or 'v' ")

    if x_axis == "r":
        ax.set_xlabel("$r/r_{\mathrm{b}}$")
        ax.set_xscale("log")

    else:
        ax.set_xlabel(r"$v_{\mathrm{kick}}/\mathrm{km \, s^{-1}}$")
    ax.set_ylabel(r"$\beta$")
    ax.legend(
        handles=[b_in_p, b_out_p],
        labels=["$r \leq r_{\mathrm{b}}$", "$r > r_{\mathrm{b}}$"],
        loc="lower right",
    )
    return ax


if __name__ == "__main__":
    main()
    # Save and show figure
    if save_fig == 1:
        print("Saving figure...", flush=True)
        plt.savefig(figure_config.fig_path(output_file))
    print("Time taken:", time.time() - start_time, flush=True)

    if show_fig == 1:
        plt.show()
