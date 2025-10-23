import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt

"""try:
    import matplotlib.pyplot as plt
except ImportError:
    from matplotlib import use
    use("Agg")
    import matplotlib.pyplot as plt"""
import arviz as az
import baggins as bgs
import figure_config

parser = argparse.ArgumentParser(
    "Determine core - kick relation",
    allow_abbrev=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-n", "--new", help="new sample", action="store_true", dest="new")
parser.add_argument("-s", "--single", help="single", action="store_true", dest="single")
parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="model type",
    choices=["exp", "lin", "sigmoid"],
    dest="model",
    default="exp",
)
parser.add_argument(
    "--nodiag", action="store_false", help="prevent Stan diagnosis", dest="diag"
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    choices=bgs.VERBOSITY,
    dest="verbosity",
    default="INFO",
    help="verbosity level",
)
args = parser.parse_args()

SL = bgs.setup_logger("script", args.verbosity)

bgs.plotting.check_backend()

# load necessary data
# data from previous core fitting routine
datafile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/core-kick.pickle"
# simulation output data at the moment just before merger
ketju_file = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000/output"
# set the stan model file base path
stan_file_base = "/users/arawling/projects/collisionless-merger-sample/code/analysis_scripts/core_kick_relation"
# set the escape velocity in km/s
ESCAPE_VEL = 1800
rng = np.random.default_rng(93849838)

if args.single:
    # usage 1: run fit (either new or loaded) for a given model
    # set the stan model file
    stan_file = os.path.join(stan_file_base, f"core-kick-{args.model}.stan")
    figname_base = f"core-study/rb-kick-models/{args.model}/{args.model}"
    stan_output = f"stan_files/core-kick-relation/{args.model}-model"
    if args.new:
        stan_new_kwargs = {
            "model_file": stan_file,
            "prior_file": "",
            "figname_base": figname_base,
            "escape_vel": ESCAPE_VEL,
            "premerger_ketjufile": ketju_file,
            "rng": rng,
        }
        if args.model == "exp":
            ck = bgs.analysis.CoreKickExp(**stan_new_kwargs)
        elif args.model == "lin":
            ck = bgs.analysis.CoreKickLinear(**stan_new_kwargs)
        else:
            ck = bgs.analysis.CoreKickSigmoid(**stan_new_kwargs)
    else:
        csv_files = bgs.utils.get_files_in_dir(
            os.path.join(bgs.DATADIR, stan_output), ext=".csv"
        )[-4:]
        stan_load_kwargs = {
            "model_file": stan_file,
            "fit_files": csv_files,
            "figname_base": figname_base,
            "escape_vel": ESCAPE_VEL,
            "premerger_ketjufile": ketju_file,
            "rng": rng,
        }
        if args.model == "exp":
            ck = bgs.analysis.CoreKickExp.load_fit(**stan_load_kwargs)
        elif args.model == "lin":
            ck = bgs.analysis.CoreKickLinear.load_fit(**stan_load_kwargs)
        else:
            ck = bgs.analysis.CoreKickSigmoid.load_fit(**stan_load_kwargs)

    ck.extract_data(d=datafile)

    if args.verbosity == "DEBUG":
        ck.print_obs_summary()
    ck.set_stan_data()

    sample_kwargs = {
        "output_dir": os.path.join(bgs.DATADIR, stan_output),
        "adapt_delta": 0.99,
        "max_treedepth": 15,
    }
    ck.sample_model(sample_kwargs=sample_kwargs, diagnose=args.diag)

    if args.verbosity == "DEBUG":
        plt.hist(ck.stan_data["x2"], 50, density=True)
        plt.xlabel(r"$v_\mathrm{kick}$")
        plt.ylabel("PDF")
        plt.show()

    ck.determine_loo()
    ck.all_plots()
    ck.print_parameter_percentiles(ck.latent_qtys)
else:
    # usage 2: run model comparison of most recently fit models
    sampled_data = figure_config.data_path("core_dist.pickle")

    fig, ax = plt.subplots()
    stan_load_kwargs = {
        "figname_base": "core-study/rb-kick-models/comparison/comp",
        "escape_vel": ESCAPE_VEL,
        "premerger_ketjufile": ketju_file,
        "rng": rng,
    }

    def yield_model():
        yield bgs.analysis.CoreKickSigmoid.load_fit(
            model_file=os.path.join(stan_file_base, "core-kick-sigmoid.stan"),
            fit_files=bgs.utils.get_files_in_dir(
                os.path.join(
                    bgs.DATADIR, "stan_files/core-kick-relation/sigmoid-model"
                ),
                ext=".csv",
            )[-4:],
            **stan_load_kwargs,
        ), "Sigmoid"
        yield bgs.analysis.CoreKickExp.load_fit(
            model_file=os.path.join(stan_file_base, "core-kick-exp.stan"),
            fit_files=bgs.utils.get_files_in_dir(
                os.path.join(bgs.DATADIR, "stan_files/core-kick-relation/exp-model"),
                ext=".csv",
            )[-4:],
            **stan_load_kwargs,
        ), "Exponential"
        yield bgs.analysis.CoreKickLinear.load_fit(
            model_file=os.path.join(stan_file_base, "core-kick-lin.stan"),
            fit_files=bgs.utils.get_files_in_dir(
                os.path.join(bgs.DATADIR, "stan_files/core-kick-relation/lin-model"),
                ext=".csv",
            )[-4:],
            **stan_load_kwargs,
        ), "Linear"

    def calculate_mode(y):
        # Function taken from StanModel, which follows the arviz implementation
        x, dens = az.kde(y)
        return x[np.nanargmax(dens)]

    def restrict_vel_dist(y, _m, ax, col):
        mask = _m.stan_data["vkick_OOS"] < _m.vmax
        az.plot_dist(_m.stan_data["vkick_OOS"][mask], ax=ax, plot_kwargs={"c": col})
        az.plot_dist(
            _m.stan_data["vkick_OOS"], ax=ax, plot_kwargs={"c": col, "ls": "--"}
        )
        SL.info(f"Sample size reduced from {len(mask)} to {np.sum(mask)}")
        SL.debug(
            f"Maximum velocity was {np.max(_m.stan_data['vkick_OOS']):.3e}, is now {np.max(_m.stan_data['vkick_OOS'][mask]):.3e}"
        )
        _y = y[:, mask]
        SL.debug(f"Posterior draws now have shape {_y.shape}")
        return _y

    loo_dict = {"Exponential": None, "Linear": None, "Sigmoid": None}
    # generator object
    models = yield_model()

    figH, axH = plt.subplots()
    rb_data = dict.fromkeys(loo_dict)
    rb_data["rb0"] = None
    rb_data["loo"] = None

    if args.new:
        for i in range(3):
            try:
                m, n = next(models)
            except StopIteration:
                break
            SL.info(f"Doing model: {n}")
            SL.debug(f"Model is {type(m)}")
            m.extract_data(d=datafile)
            m.set_stan_data(restrict_v=False)
            m.sample_model(diagnose=False)

            rb_data["rb0"] = m.rb0

            loo_dict[n] = m.determine_loo()

            rb_data[n] = {}
            rb_vals = m.sample_generated_quantity(
                m.folded_qtys_posterior[0], state="OOS"
            )
            rb_data[n]["rb"] = rb_vals
            rb_data[n]["vel"] = m.stan_data["vkick_OOS"]
            rb_data[n]["vel_max"] = m.vmax

            m.print_parameter_percentiles(m.latent_qtys)
            for lq in m.latent_qtys:
                hdi = az.hdi(m.sample_generated_quantity(lq))
                SL.info(f"1-sigma (68%) HDI for {lq} is {hdi}")

        rb_data["loo"] = loo_dict

        bgs.utils.save_data(
            rb_data, figure_config.data_path(sampled_data), exist_ok=True
        )
    else:
        rb_data = bgs.utils.load_data(sampled_data)

    rb0 = rb_data.pop("rb0")
    loo_dict = rb_data.pop("loo")
    rb_data.pop("__githash", None)
    rb_data.pop("__script", None)

    for (k, v), c in zip(rb_data.items(), figure_config.custom_colors_shuffled[1:]):
        SL.info(f"Doing model {k}")
        rb_vals = v["rb"]
        assert np.all(np.sign(np.diff(v["vel"])) > 0)

        # XXX for an unknown reason, when sampling the core radii above, the
        # order of the array of core radii values becomes jumbled, so masking
        # based off the kick velocity no longer works. As the tested functions
        # are monotonic, let's just take the cut to be when the median (across
        # draws and chains) core value exceeds a threshold, determined by when
        # the kick velocity exceeds the desired value of 1020 km/s
        sort_idxs = np.argsort(np.median(rb_vals, axis=0))
        rb_vals = rb_vals[:, sort_idxs]
        break_idx = np.argmax(v["vel"] > v["vel_max"])

        restricted_rb_vals = rb_vals[:, :break_idx]

        SL.debug(f"Core sample: {rb_vals.shape} --> {restricted_rb_vals.shape}")

        # plot restricted values
        az.plot_kde(
            restricted_rb_vals,
            ax=ax,
            label=f"$\mathrm{{{k}}}$",
            plot_kwargs={"color": c},
        )

        # plot unrestricted values
        az.plot_kde(rb_vals, ax=ax, plot_kwargs={"ls": "-.", "alpha": 0.4, "color": c})

        # print some diagnostics
        for s_rb, rb in zip(
            ("unrestricted", "restricted"), (rb_vals, restricted_rb_vals)
        ):
            rb_mode = calculate_mode(rb)
            SL.info(
                f"Mode of {s_rb} distribution is {rb_mode:.2f} rb0, or {rb_mode*rb0:.2f} kpc"
            )

    ax.legend()
    ax.set_xlabel(r"$r_\mathrm{b}/r_{\mathrm{b},0}$")
    ax.set_ylabel(r"$\mathrm{PDF}$")
    # set xlimits by hand
    ax.set_xlim(0, 6)
    # add a secondary axis, turning off ticks from the top axis (if they are there)
    ax.tick_params(axis="x", which="both", top=False)
    secax = ax.secondary_xaxis("top", functions=(lambda x: x * rb0, lambda x: x / rb0))
    secax.set_xlabel(r"$r_\mathrm{b}/\mathrm{kpc}$")
    bgs.plotting.savefig(figure_config.fig_path("rb_pdf.pdf"), fig=fig, force_ext=True)
    plt.close()

    # do LOO model comparison
    comp = az.compare(loo_dict, ic="loo")
    print("Model comparison")
    print(comp)
