import numpy as np
import matplotlib.pyplot as plt
import figure_config
from scipy.ndimage import median_filter
from matplotlib.patches import ConnectionPatch


def get_data_file(v_kick):
    return figure_config.data_path(f"lagrangian_files/data/kick-vel-{v_kick}.txt")


def get_infall_time(v_kick):
    data_file = get_data_file(v_kick)
    data = np.loadtxt(data_file, skiprows=1)
    t = data[:, 0]
    r = data[:, 1]
    v = data[:, 2]  # velocities are w.r.t CoM velocity

    for i in range(2, len(t) - 1):
        t0 = t[i]
        v_used = v[(t - t0 < 0.1) & (t >= t0)]
        r_used = r[(t - t0 < 0.1) & (t >= t0)]
        if not any(r_used > 0.1):
            if np.median(v_used) < 25:
                inds = v_used < 25
                ind = np.argmax(inds)
                print(v_kick, i + ind)

                return t[i + ind]
    print(v_kick, -1, t[-1])
    return t[-1]


def get_lagrangian_radii_evolution(v_kick):
    gad_t_fac = 0.978
    data_file = get_data_file(v_kick)
    data = np.loadtxt(data_file, skiprows=1)
    t = data[:, 0] * gad_t_fac
    r_mBH = data[
        :, 4
    ]  # radius enclosing stellar particles with mass equal to the BH mass

    return t, r_mBH


v_list = [
    0,
    60,
    120,
    180,
    240,
    300,
    360,
    420,
    480,
    540,
    600,
    660,
    720,
    780,
    840,
    900,
    960,
    1020,
    1080,
    1140,
    1200,
    1260,
    1320,
    1380,
    1440,
    1500,
    1560,
    1620,
    1680,
    1740,
    1800,
    2000,
]
# settled runs and largest kick
v_list = [
    0,
    60,
    120,
    180,
    240,
    300,
    360,
    420,
    480,
    540,
    600,
    660,
    720,
    780,
    840,
    900,
    960,
    1020,
    2000,
]

vkcols = figure_config.VkickColourMap()

fig, ax = plt.subplots(1, 1)
t_back_to_center = np.zeros(len(v_list))

xlim = (-136.26946625, 2662.49147125)
xlim = (-50, 2662.49147125)
ylim = (0.9885675684202839, 1.8)
plt.setp(ax, xlim=xlim, ylim=ylim)

# add inset axes
xlim_ins = (0, 100)
ylim_ins = (0.998, 1.45)
axins = ax.inset_axes([0.35, 0.1, 0.55, 0.3], xlim=xlim_ins, ylim=ylim_ins)

# want to set connector paths differently
rect = (xlim_ins[0], ylim_ins[0], xlim_ins[1] - xlim_ins[0], ylim_ins[1] - ylim_ins[0])
ax.indicate_inset(rect, edgecolor="k")
cp1 = ConnectionPatch(
    xyA=(100, 0.998),
    xyB=(0, 0),
    axesA=ax,
    axesB=axins,
    coordsA="data",
    coordsB="axes fraction",
    alpha=0.5,
    zorder=4.99,
)
cp2 = ConnectionPatch(
    xyA=(100, 1.45),
    xyB=(0, 1),
    axesA=ax,
    axesB=axins,
    coordsA="data",
    coordsB="axes fraction",
    alpha=0.5,
    zorder=4.99,
)
ax.add_patch(cp1)
ax.add_patch(cp2)

for i in range(len(v_list)):
    # Merger time acquired from ketju_bhs output with the following command
    # h5dump -d mergers /scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000/output/ketju_bhs.hdf5
    t_merge = 0.00905306
    v = v_list[i]
    v_kick = f"{v:04d}"
    t, r_lag = get_lagrangian_radii_evolution(v_kick)
    t_settle = get_infall_time(v_kick)

    r_after_merge = r_lag[t > t_merge]
    t_cut = t[t > t_merge]
    r_lag = r_after_merge[t_cut <= t_settle]

    r_smooth = median_filter(r_lag, 5, mode="nearest")
    t = t_cut[t_cut <= t_settle]
    r_lag = r_smooth
    ax.plot(
        (t[t <= t_settle] - t_merge) * 1000,
        r_lag[t <= t_settle],
        "-",
        color=vkcols.get_colour(v),
    )
    axins.plot(
        (t[t <= t_settle] - t_merge) * 1000,
        r_lag[t <= t_settle],
        "-",
        color=vkcols.get_colour(v),
    )


cbar = vkcols.make_cbar(ax)
ax.set_ylabel(r"$r(M_\star=M_\bullet)/\mathrm{kpc}$")
ax.set_xlabel("$t/\mathrm{Myr}$")

log = False
if log:
    ax.set_xscale("log")
    plt.savefig(figure_config.fig_path("lagrangian_radii_log.pdf"))
else:
    plt.savefig(figure_config.fig_path("lagrangian_radii.pdf"))
