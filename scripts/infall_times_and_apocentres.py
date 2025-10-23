import numpy as np
import matplotlib.pyplot as plt
import figure_config


def get_data_file(v_kick):
    return figure_config.data_path(f"lagrangian_files/data/kick-vel-{v_kick}.txt")


def get_infall_time(v_kick):
    data_file = get_data_file(v_kick)
    data = np.loadtxt(data_file, skiprows=1)
    gad_t_fac = 0.978
    t = data[:, 0] * gad_t_fac
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
                end_i = i + ind
                break
    t_end = t[end_i]

    return t_end, end_i


def get_apocenter_distance(v_kick, infall_index):
    data_file = get_data_file(v_kick)
    data = np.loadtxt(data_file, skiprows=1)
    r = data[:, 1]

    # the kick seems to happen between second and third snapshots (001 and 002)
    dists = r[2 : infall_index + 1]
    if len(dists) == 1:
        return dists[0]
    # We have a apocenter if there ever is a situation where distance to center decreases
    if not (all(dists[:-1] <= dists[1:])):
        return max(dists)

    print("No apocenter found for kick velocity ", v_kick)
    print(dists)
    quit()
    return -1


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
# settled runs
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
]

fig, ax = plt.subplots(2, 1, figsize=(3, 4))
times = np.zeros(len(v_list))
apocenters = np.zeros(len(v_list))
vkcols = figure_config.VkickColourMap()


for i in range(len(v_list)):
    v = v_list[i]
    # print(v)
    v_kick = f"{v:04d}"
    t_infall, infall_index = get_infall_time(v_kick)
    apo_dist = get_apocenter_distance(v_kick, infall_index)
    times[i] = t_infall
    apocenters[i] = apo_dist
    print(infall_index)


line_text = "$r_\mathrm{b,0}$"
# radius of stellar core scoured by the SMBH binary
rb = 0.58
sigma_1D = 270.0
sigma_text = "$\sigma_\star(r<r_\mathrm{b,0})$"
ax[0].axhline(rb, color="k", ls="dotted", lw=1)
ax[0].axvline(sigma_1D, color="k", ls="dotted", lw=1)
ax[0].text(sigma_1D + 15, 1.5e-2, sigma_text, rotation=90)
ax[0].text(800, rb + 0.11, line_text)

ax[1].scatter(
    v_list[1:],
    times[1:],
    label="Infall time",
    color=figure_config.col_list[1],
    **figure_config.marker_kwargs,
)

ax[0].scatter(
    v_list[1:],
    apocenters[1:],
    color=figure_config.col_list[1],
    **figure_config.marker_kwargs,
)
ax[0].set_yscale("log")
ax[1].set_yscale("log")

ax[0].set_xticklabels([])
ax[1].set_xlabel("$v_\mathrm{kick}/\mathrm{km}\,\mathrm{s}^{-1}$")
ax[1].set_ylabel("$t_\mathrm{settle}/\mathrm{Gyr}$")
ax[0].set_ylabel("$r_\mathrm{apo}/\mathrm{kpc}$")

plt.savefig(figure_config.fig_path("infall_times_and_apocenters.pdf"))
