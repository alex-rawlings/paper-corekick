import numpy as np
import baggins as bgs
import figure_config

bgs.plotting.check_backend()

data1 = bgs.literature.LiteratureTables.load_thomas_2016_data()

vkcols = figure_config.VkickColourMap()

obs_scatter_kwargs = {"alpha": 1, "c": None, "marker": "."}
obs_scatter_kwargs.update(figure_config.marker_kwargs)

ax = data1.scatter("BH_mass", "core_radius_kpc", scatter_kwargs=obs_scatter_kwargs)

# add the intrinsic scatter area from Thomas et al. 2016
# note that the relation is given for core radius as the dependent variable,
# whereas we have BH mass as the dependent variable
xhat = np.geomspace(
    data1.table.loc[:, "BH_mass"].min(), data1.table.loc[:, "BH_mass"].max(), 1000
)
slope = 1.17
intercept = 10.27
mean_relation = -intercept / slope + 1 / slope * np.log10(xhat)
intrinsic_scatter = -0.29 / slope
line1 = ax.plot(xhat, 10**mean_relation, c=figure_config.col_list[1], zorder=1)
ax.fill_between(
    xhat,
    10 ** (mean_relation - intrinsic_scatter),
    10 ** (mean_relation + intrinsic_scatter),
    alpha=0.3,
    fc=line1[-1].get_color(),
    zorder=1,
)

# add mergers
kicks = dict(
    v0000=0.580,
    v0060=0.632,
    v0120=0.737,
    v0180=0.757,
    v0240=0.832,
    v0300=1.13,
    v0360=1.07,
    v0420=1.22,
    v0480=1.22,
    v0540=1.40,
    v0600=1.33,
    v0660=1.40,
    v0720=1.51,
    v0780=1.58,
    v0840=1.56,
    v0900=1.58,
    v0960=1.46,
    v1020=1.64,
)
my_BH_mass = 5.86e9

for i, (vk, rb) in enumerate(kicks.items()):
    ax.scatter(
        my_BH_mass,
        rb,
        marker="s",
        color=vkcols.get_colour(float(vk.lstrip("v"))),
        label=(r"$\mathrm{Simulation}$" if i == 0 else ""),
        zorder=2.5,
        **figure_config.marker_kwargs,
    )

ax.set_xlabel(r"$M_\bullet/\mathrm{M}_\odot$")
ax.set_ylabel(r"$r_\mathrm{b}/\mathrm{kpc}$")
ax.set_xscale("log")
ax.set_yscale("log")
vkcols.make_cbar(ax=ax, extend=None)
ax.legend(fontsize=7)

bgs.plotting.savefig(figure_config.fig_path("core-obs.pdf"), force_ext=True)
