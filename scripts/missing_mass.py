import numpy as np
import figure_config
import pickle
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm


col_list = figure_config.color_cycle_shuffled.by_key()["color"]


def idot(rb, re, n, gamma, alpha, log10densb):
    mub = 10**log10densb
    b = 2.0 * n - 0.33333333 + 0.009876 * (1 / n)

    return (
        mub
        * 2 ** (-gamma / alpha)
        * np.exp(b * (2 ** (1 / alpha) * rb / re) ** (1 / n))
    )


def i_ser(r, rb, re, n, gamma, alpha, i):
    b = 2.0 * n - 0.33333333 + 0.009876 * (1 / n)
    return (
        i
        * (1 + (rb / r) ** alpha) ** (gamma / alpha)
        * np.exp(-b * ((r**alpha + rb**alpha) / re**alpha) ** (1 / (alpha * n)))
    )


def mass_deficit(r, rb, re, n, log10densb, g, a):
    """
    :param r: radius
    :param rb: break radius
    :param re: effective radius
    :param n: Sersic index
    :param log10densb: normalisation factor Sigma_b
    :param g: gamma, inner profile slope index
    :param a: alpha, transition index
    :return: The integrand of the mass deficit equation
    """

    i_cs = idot(rb, re, n, g, a, log10densb)
    core_ser = i_ser(r, rb, re, n, g, a, i_cs)
    ser = i_ser(r, 0, re, n, g, a, i_cs)
    return (ser - core_ser) * r


def missing_mass_plot(filename, nro_iter=10000, min_r=1e-2, ax=None, debug_mode=False):
    """
    :param filename: The path and dictionary file of the data. Dictionary should include break radius,
    effective radius, sersic index, normalistion factor sigma_b, inner profile slope index and the transition index.
    Alternatively, the dictionary may already be loaded and the data used
    :param nro: number of iterations
    :param min_r: minimum radius for integration
    :param ax: plotting axis
    :return: plotting axis
    """
    if isinstance(filename, str):
        with open(filename, "rb") as f:
            data = pickle.load(f)
    else:
        data = filename

    # Kick velocities
    vkicks = data["rb"].keys()

    # An empty dictionry to save the data
    mdef_dict = dict()
    n_dict = dict()

    # Defining the random seed to get reproducible results
    rng = np.random.default_rng(seed=42)
    # Looping over all kick velocities
    for j, v in enumerate(vkicks):
        rb = data["rb"][v].flatten()  # break radius
        Re = data["Re"][v].flatten()  # effective radius
        n = data["n"][v].flatten()  # sersic index
        log10densb = data["log10densb"][v].flatten()  # normalisation factor Sigma_b
        g = data["g"][v].flatten()  # gamma, inner profile slope index
        a = data["a"][v].flatten()

        mdef_dict[v] = np.zeros(nro_iter)
        n_dict[v] = np.zeros(nro_iter)

        for i in tqdm(range(nro_iter), desc=f"Missing mass for case {v}"):
            rb_new, Re_new, n_new, log10densb_new, g_new, a_new = (
                rng.choice(rb),
                rng.choice(Re),
                rng.choice(n),
                rng.choice(log10densb),
                rng.choice(g),
                rng.choice(a),
            )

            m, abserr = integrate.quad(
                mass_deficit,
                min_r,
                5 * Re_new,
                args=(rb_new, Re_new, n_new, log10densb_new, g_new, a_new),
            )
            while np.isnan(m):
                rb_new, Re_new, n_new, log10densb_new, g_new, a_new = (
                    rng.choice(rb),
                    rng.choice(Re),
                    rng.choice(n),
                    rng.choice(log10densb),
                    rng.choice(g),
                    rng.choice(a),
                )

                m, abserr = integrate.quad(
                    mass_deficit,
                    min_r,
                    5 * Re_new,
                    args=(rb_new, Re_new, n_new, log10densb_new, g_new, a_new),
                )
            mdef_dict[v][i] = 2 * np.pi * m
            n_dict[v][i] = rb_new

    # Creating the figure
    new_figure = False
    if ax is None:
        fig, ax = plt.subplots()
        new_figure = True
    velocities = [int(key) for key in data["rb"].keys()]
    norm_val = np.median(mdef_dict["0000"])
    bp = ax.boxplot(
        [_v / norm_val for _v in mdef_dict.values()],
        positions=velocities,
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

    ax.tick_params(axis="y", which="both", right=False)
    # let's normalise by norm_val / 1e9
    new_norm_val = norm_val / 1e9
    print(f"We will normalise mass by {norm_val:.2e} Msol")
    print(
        f"The 0 km/s case has IQR spanning {np.nanquantile(mdef_dict['0000'], 0.25):.2e} - {np.nanquantile(mdef_dict['0000'], 0.75):.2e} Msol"
    )
    ax2 = ax.secondary_yaxis(
        "right", functions=(lambda x: x * new_norm_val, lambda x: x / new_norm_val)
    )
    try:
        ax2.ticklabel_format(style="sci", useMathText=True)
    except AttributeError:
        # we are using a log-scale, so no need for scientific format
        pass
    # axis labels
    ax.set_ylabel(r"$M_\mathrm{def} / M_\mathrm{def,0}$")
    ax2.set_ylabel(r"$M_\mathrm{def}/ (10^9\,\mathrm{M}_\odot)$")
    if new_figure:
        ax.set_xlabel(r"$v_\mathrm{kick}/\mathrm{kms}^{-1}$")
        plt.show()

    if debug_mode:
        fig3, ax3 = plt.subplots(1, 3, sharex="all", sharey="all")

        vk = ["0180", "0480", "0840"]
        for j in range(len(vk)):
            ax3[j].set_title(vk[j])
            positivemask = mdef_dict[vk[j]] > 0
            ntemp = n_dict[vk[j]][positivemask]
            mtemp = np.log10(mdef_dict[vk[j]][positivemask])

            nmask = np.logical_and(
                ntemp > np.quantile(ntemp, 0.25), ntemp < np.quantile(ntemp, 1 - 0.25)
            )
            mmask = np.logical_and(
                mtemp > np.quantile(mtemp, 0.25), mtemp < np.quantile(mtemp, 1 - 0.25)
            )

            bigmask = np.logical_and(nmask, mmask)

            h = ax3[j].hist2d(
                ntemp[bigmask], mtemp[bigmask], norm=colors.LogNorm(1e-1, 100), bins=10
            )

            ax3[j].set_ylabel(r"$M_\mathrm{def}$")
            ax3[j].set_xlabel(r"$r_\mathrm{b}$")

            fig3.colorbar(h[3], ax=ax3[j])
        plt.show()

    return ax


if __name__ == "__main__":
    # run the program independently
    data_file = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/core-kick.pickle"
    missing_mass_plot(data_file, nro_iter=1000, debug_mode=True)
