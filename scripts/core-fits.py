import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import figure_config
import arviz as az
import missing_mass as mm

parser = argparse.ArgumentParser(
    description="Plot core fits given a Stan sample",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-e", "--extract", help="extract data", action="store_true", dest="extract"
)
parser.add_argument(
    "-p",
    "--parameter",
    help="parameter to plot",
    choices=["Re", "rb", "n", "a", "log10densb", "g", "all", "OOS"],
    default="rb",
    dest="param",
)
parser.add_argument(
    "-d",
    "--diff",
    help="distribution difference plot",
    action="store_true",
    dest="diffplot",
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
data_file = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/core-kick.pickle"
rng = np.random.default_rng(42)
col_list = figure_config.color_cycle_shuffled.by_key()["color"]
ESCAPE_VEL = 1800

bgs.plotting.check_backend()

if args.extract:
    main_path = "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs"
    analysis_params = bgs.utils.read_parameters(
        "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml"
    )

    with os.scandir(main_path) as _it:
        subdirs = [entry.path for entry in _it if entry.is_dir() and "-v" in entry.name]
    subdirs.sort()
    for s in subdirs:
        SL.debug(f"Reading: {s}")

    figname_base = "ck"

    # put the data into a format we can pickle as numpy arrays for faster
    # plotting
    data = {
        "rb": {},
        "Re": {},
        "n": {},
        "log10densb": {},
        "g": {},
        "a": {},
        "R_OOS": {},
        "log10_surf_rho": {},
    }

    # load the fits
    for subdir in subdirs:
        csv_files = bgs.utils.get_files_in_dir(subdir, ext=".csv")[-4:]
        try:
            graham_model = bgs.analysis.GrahamModelHierarchy.load_fit(
                model_file=os.path.join(
                    bgs.HOME,
                    "projects/collisionless-merger-sample/code/analysis_scripts/hierarchical_models/stan/density/graham_hierarchy.stan",
                ),
                fit_files=csv_files,
                figname_base=figname_base,
            )
        except ValueError as e:
            SL.error(f"Unable to load data from directory: {subdir}: {e}. Skipping")
            continue
        SL.info(f"Loaded model from csv files {csv_files[0]}")

        graham_model.extract_data(analysis_params, None, binary=False)
        graham_model.set_stan_data()
        graham_model.sample_model(
            sample_kwargs=analysis_params["stan"]["density_sample_kwargs"],
            diagnose=False,
        )
        gid = graham_model.merger_id.split("-")[-1][1:]
        for k in data.keys():
            if k == "R_OOS":
                data[k][gid] = graham_model.stan_data[k]
            else:
                data[k][gid] = graham_model.sample_generated_quantity(
                    f"{k}_posterior", state="OOS"
                )
    bgs.utils.save_data(data, data_file)
else:
    SL.debug(f"Reading {data_file}")
    data = bgs.utils.load_data(data_file)


def _helper(param_name, ax):
    kick_vels = []
    param = []
    normalisation = (
        rng.permutation(data[param_name]["0000"].flatten()) if param_name == "rb" else 1
    )
    SL.warning(f"Determining distributions for parameter: {param_name}")
    for k, v in data[param_name].items():
        if k == "__githash" or k == "__script":
            continue
        if k == "2000":
            continue
        SL.info(f"Determining ratio for model {k}")
        kick_vels.append(float(k))
        # determine the ratio of rb / rb_initial
        val = v.flatten() / normalisation
        param.append(val[~np.isnan(val)])
    bp = ax.boxplot(
        param,
        positions=kick_vels,
        showfliers=False,
        whis=0,
        widths=50,
        manage_ticks=False,
        patch_artist=True,
        showcaps=False,
    )
    for p in bp["boxes"]:
        p.set_facecolor(col_list[0])
        p.set_edgecolor(p.get_facecolor())
        p.set_alpha(0.3)
    for m in bp["medians"]:
        m.set_color("#003A74")
        m.set_linewidth(2)
        m.set_alpha(1)
    for w in bp["whiskers"]:
        w.set_alpha(0)
    return np.nanmedian(normalisation), kick_vels


def distribution_diff_plot(param_name, bins=20):
    SL.warning(f"Determining distribution difference for parameter: {param_name}")
    dset = data[param_name]["1020"].flatten() - data[param_name]["0000"].flatten()
    dset = np.sort(dset)
    fig, ax = plt.subplots(1, 2, sharex="all")
    ax[0].hist(dset, bins=bins, density=True)
    t = np.linspace(np.min(dset), np.max(dset), 100)
    ecdf = list(map(lambda tt: bgs.mathematics.empirical_cdf(dset, tt), t))
    ax[1].plot(t, ecdf)
    SL.info(
        f"From the ECDF, 0 corresponds to the {bgs.mathematics.empirical_cdf(dset, 0):.2f} quantile"
    )
    label_dict = dict(
        Re=r"$R_\mathrm{e}$",
        rb=r"$r_\mathrm{b}$",
        g=r"$\gamma$",
        a=r"$\alpha$",
        log10densb=r"$\log_{10} \Sigma_\mathrm{b}$",
        n=r"$n",
    )
    for axi in ax:
        axi.set_xlabel(
            f"{label_dict[param_name]}$(1020)$ - {label_dict[param_name]}$(0)$"
        )
    ax[0].set_ylabel(r"$\mathrm{PDF}$")
    ax[1].set_ylabel(r"$\mathrm{ECDF}$")
    return ax


xlabel = r"$v_\mathrm{kick}/\mathrm{km}\,\mathrm{s}^{-1}$"
if args.param == "all":
    ylabs = dict(
        Re=r"$R_\mathrm{e}/\mathrm{kpc}$",
        rb=r"$r_\mathrm{b}/r_\mathrm{b,0}$",
        n=r"$n$",
        a=r"$\alpha$",
        log10densb=r"$\log_{10}\left(\Sigma_\mathrm{b}/(\mathrm{M}_\odot\mathrm{kpc}^{-2})\right)$",
        g=r"$\gamma$",
    )
    fig, ax = plt.subplots(3, 2, sharex="all")
    fig.set_figwidth(1.2 * fig.get_figwidth())
    fig.set_figheight(1.5 * fig.get_figheight())
    for axi in ax[-1, :]:
        axi.set_xlabel(xlabel)
    for i, pname in enumerate(("rb", "log10densb", "g", "Re", "a", "n")):
        SL.info(f"Making plot for {pname}")
        _helper(pname, ax.flat[i])
        ax.flat[i].set_ylabel(ylabs[pname])
    fname = f"{args.param}-kick.pdf"
elif args.param == "OOS":
    fig, ax = plt.subplots(1, 1)
    axins = ax.inset_axes(
        [0.07, 0.02, 0.6, 0.5],
        xlim=(-0.95, 0),
        ylim=(9.25, 9.85),
        xticklabels=[],
        yticklabels=[],
        xticks=[],
        yticks=[],
    )

    def cols():
        for c in figure_config.custom_colors_shuffled:
            yield c

    cgen = cols()
    for k, v in data["R_OOS"].items():
        if k == "__githash" or k == "__script":
            continue
        if k not in ("0000", "0480", "0600", "1020"):
            continue
        SL.info(f"Determining density for model {k}")
        c = next(cgen)

        def _dens_plotter(axi):
            az.plot_hdi(
                np.log10(v),
                data["log10_surf_rho"][k],
                hdi_prob=0.25,
                ax=axi,
                smooth=True,
                hdi_kwargs={"skipna": True},
                fill_kwargs={
                    "label": f"{float(k):.1f}",
                    "color": c,
                    "edgecolor": c,
                    "lw": 0.5,
                },
            )

        _dens_plotter(ax)
        _dens_plotter(axins)
    ax.indicate_inset_zoom(axins, edgecolor="k")
    ax.set_xlabel(r"$\log_{10}(R/\mathrm{kpc})$")
    ax.set_ylabel(
        r"$\log_{10}\left(\Sigma(R)/\mathrm{M}_\odot\,\mathrm{kpc}^{-2}\right)$"
    )
    ax.legend(
        title=r"$v_\mathrm{kick}/\mathrm{km}\,\mathrm{s}^{-1}$", loc="upper right"
    )
    fname = "density.pdf"
else:
    fig, ax = plt.subplots(2, 1, sharex="all")
    fig.set_figheight(1.5 * fig.get_figheight())
    # plot the core radius boxplots
    ax[-1].set_xlabel(xlabel)
    norm_val, sampled_kicks = _helper(args.param, ax[0])
    if args.param == "rb":
        ax[0].tick_params(axis="y", which="both", right=False)
        ax2 = ax[0].secondary_yaxis(
            "right", functions=(lambda x: x * norm_val, lambda x: x / norm_val)
        )
        ax[1].set_yscale("log")
        vkick = np.linspace(min(sampled_kicks), max(sampled_kicks), 500)
        # add best fit relations
        ax[0].plot(
            vkick,
            2.72 * (vkick / ESCAPE_VEL) ** 0.782 + 1,
            label=r"$\mathrm{Exponential}$",
            c=col_list[1],
        )
        ax[0].plot(
            vkick,
            3.03 * vkick / ESCAPE_VEL + 1,
            label=r"$\mathrm{Linear}$",
            c=col_list[2],
        )
        ax[0].plot(
            vkick,
            3.04 * (1 - np.exp(-1.56 * vkick / ESCAPE_VEL)) + 1,
            label=r"$\mathrm{Sigmoid}$",
            c=col_list[3],
        )
        ax[0].set_ylabel(r"$r_\mathrm{b}/r_{\mathrm{b},0}$")
        ax2.set_ylabel(r"$r_\mathrm{b}/\mathrm{kpc}$")
        ax[0].legend()
        # add Sonja's missing mass plot
        mm.missing_mass_plot(data, ax=ax[1], nro_iter=10000)
    elif args.param == "Re":
        ax.set_ylabel(r"$R_\mathrm{e}/\mathrm{kpc}$")
    elif args.param == "n":
        ax.set_ylabel(r"$n$")
    elif args.param == "a":
        ax.set_ylabel(r"$\alpha$")
    elif args.param == "g":
        ax.set_ylabel(r"$\gamma$")
    else:
        ax.set_ylabel(r"log($\Sigma(R)$/(M$_\odot$/kpc$^2$))")
    fname = f"{args.param}-kick.pdf"

bgs.plotting.savefig(figure_config.fig_path(fname), force_ext=True)

if args.diffplot:
    axd = distribution_diff_plot(args.param)
    plt.show()
