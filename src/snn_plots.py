import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.ticker import ScalarFormatter

from src.models import SimulationParams


def init_matplotlib_style():
    """
    Initialize the matplotlib style/settings.
    """
    mpl.rcParams.update(
        {
            "text.usetex": False,  # Disable LaTeX
            "font.family": "DeJavu Serif",  # Use serif font
            "font.serif": ["Times New Roman"],  # A LaTeX-like font
            "axes.labelsize": 26,  # Font size for axis labels
            "axes.titlesize": 26,  # Font size for titles
            "xtick.labelsize": 26,  # Font size for x-axis ticks
            "ytick.labelsize": 26,  # Font size for y-axis ticks
            "legend.fontsize": 18,  # Font size for legend
        }
    )


def plot_all_results(
    simulation_df: pd.DataFrame,
    simulation_params: SimulationParams,
    low_quantile: float = 0.01,
    high_quantile: float = 0.99,
):
    """Generate three plots: (1) Leak-free vs leak <W>(<ISI>) comparison,(2) number of spikes, and (3) LZW complexity."""

    # Raise error if any of "w_mean", "total_spikes", "lzw_complexity", or "mean_isi" columns are missing
    if not all(
        col in simulation_df.columns
        for col in ["w_mean", "total_spikes", "lzw_complexity", "mean_isi"]
    ):
        raise ValueError(
            "Missing columns in simulation_df: 'w_mean', 'total_spikes', 'lzw_complexity', 'mean_isi'"
        )
    init_matplotlib_style()

    membrane_threshold = simulation_params.membrane_threshold
    currents_period = simulation_params.currents_period
    external_current = simulation_params.external_current
    leak_coefficient = simulation_params.leak_coefficient
    num_neurons = simulation_params.num_neurons
    small_world_graph_k = simulation_params.small_world_graph_k

    def w_leak_free(delta):
        return (
            membrane_threshold * delta
            - (external_current * delta**2) / (currents_period * num_neurons)
        ) / (0.5 * small_world_graph_k * (delta - 1))

    def w_leak(delta):
        return (
            (leak_coefficient * membrane_threshold * delta**2)
            / (1 - np.exp(-leak_coefficient * delta))
            - (external_current * delta**2) / (currents_period * num_neurons)
        ) / (0.5 * small_world_graph_k * (delta - 1))


    w_crit = membrane_threshold / (0.5 * small_world_graph_k) - (
        2 * external_current
    ) / (currents_period * num_neurons * 0.5 * small_world_graph_k)

    w_mean_unique_values = simulation_df["w_mean"].unique()

    grouped_by_w_mean = simulation_df.groupby("w_mean")
    mean_isi_median = grouped_by_w_mean["mean_isi"].median().values
    mean_isi_q1 = grouped_by_w_mean["mean_isi"].quantile(low_quantile).values
    mean_isi_q3 = grouped_by_w_mean["mean_isi"].quantile(high_quantile).values

    log_max_delta = np.log10(np.max(mean_isi_median))
    delta = np.logspace(0.01, log_max_delta, 500)
    w_leak_free_values = w_leak_free(delta)
    w_leak_values = w_leak(delta)

    _, axs = plt.subplots(2, 2, figsize=(24, 16), constrained_layout=True)

    # Top-left plot
    axs[0, 0].plot(
        mean_isi_median,
        w_mean_unique_values,
        label=r"Synthetic SNN",
        color="black",
        linewidth=5,
    )
    axs[0, 0].fill_betweenx(
        w_mean_unique_values, mean_isi_q1, mean_isi_q3, color="black", alpha=0.2
    )
    axs[0, 0].plot(
        delta,
        w_leak_free_values,
        label=r"$\langle W \rangle(\langle \Delta\rangle)_{\text{leak-free}}$ (Eq. 6)",
        color="purple",
        linewidth=5,
    )
    axs[0, 0].axhline(
        w_crit,
        color="purple",
        linestyle="--",
        linewidth=3,
        label=r"$\langle W \rangle_{\text{critical,leak-free}}$ (Eq. 3)",
    )
    axs[0, 0].plot(
        delta,
        w_leak_values,
        label=r"$\langle W \rangle(\langle \Delta\rangle)_{\text{leak}}$ (Eq. 2)",
        color="green",
        linewidth=5,
    )

    axs[0, 0].set_xscale("log")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_ylim(np.min(w_mean_unique_values), np.max(w_mean_unique_values))
    axs[0, 0].set_xlabel(r"Average ISI ($\langle \Delta \rangle$, ms)")
    axs[0, 0].set_ylabel(r"Average Synaptic Weight ($\langle W \rangle$)")
    axs[0, 0].legend(loc="lower left")

    # Top-right plot
    nspike_median = grouped_by_w_mean["total_spikes"].median().values
    nspike_q1 = grouped_by_w_mean["total_spikes"].quantile(low_quantile).values
    nspike_q3 = grouped_by_w_mean["total_spikes"].quantile(high_quantile).values

    axs[0, 1].plot(
        w_mean_unique_values,
        nspike_median,
        color="black",
        linewidth=5,
        label=r"Synthetic SNN",
    )
    axs[0, 1].fill_between(
        w_mean_unique_values, nspike_q1, nspike_q3, color="black", alpha=0.2
    )
    axs[0, 1].axvline(
        x=w_crit,
        color="purple",
        linestyle="--",
        linewidth=5,
        label=r"$\langle W \rangle_{\text{critical,leak-free}}$ (Eq. 3)",
    )

    w_min_spike_lzw_plot = w_crit * 0.5
    w_max_spike_lzw_plot = w_crit * 3
    axs[0, 1].set_xlim(w_min_spike_lzw_plot, w_max_spike_lzw_plot)
    axs[0, 1].set_xlabel(r"Average Synaptic Weight  ($\langle W \rangle$)")
    axs[0, 1].set_ylabel(r"Number of Spikes")

    # Configure y-axis to use scientific notation
    axs[0, 1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axs[0, 1].yaxis.get_major_formatter().set_scientific(True)
    axs[0, 1].yaxis.get_major_formatter().set_powerlimits((-1, 1))

    axs[0, 1].legend(loc="lower right")

    # Bottom-left plot
    median_complexity = grouped_by_w_mean["lzw_complexity"].median().values
    median_complexity_max = median_complexity.max()
    median_complexity /= median_complexity_max
    q1_complexity = (
        grouped_by_w_mean["lzw_complexity"].quantile(low_quantile).values
        / median_complexity_max
    )
    q3_complexity = (
        grouped_by_w_mean["lzw_complexity"].quantile(high_quantile).values
        / median_complexity_max
    )

    axs[1, 0].plot(
        w_mean_unique_values,
        median_complexity,
        color="black",
        linewidth=5,
        label=r"Synthetic SNN",
    )
    axs[1, 0].fill_between(
        w_mean_unique_values, q1_complexity, q3_complexity, color="black", alpha=0.2
    )
    axs[1, 0].axvline(
        x=w_crit,
        color="purple",
        linestyle="--",
        linewidth=5,
        label=r"$\langle W \rangle_{\text{critical,leak-free}}$ (Eq. 3)",
    )
    axs[1, 0].set_xlim(w_min_spike_lzw_plot, w_max_spike_lzw_plot)
    axs[1, 0].set_xlabel(r"Average Synaptic Weight ($\langle W \rangle$)")
    axs[1, 0].set_ylabel(r"Normalized LZW Complexity")
    axs[1, 0].legend(loc="upper right")

    axs[1, 1].axis("off")  # Remove empty axis.
