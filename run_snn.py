"""Simulate a Spiking Neural Network (SNN) with a range of w_mean values."""

import argparse
import logging
import os
import random
import time
from tqdm import tqdm
from typing import Any, Dict, List
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models import lzw_complexity_from_matrix
from src.models import SNN, SimulationParams
from src.snn_plots import plot_all_results
from src.utils import load_config


# Configure logging to include INFO level and above
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_weights_steps_array(w_critical, num_steps):
    w_start = 0.1 * w_critical
    w_mid = 1.5 * w_critical
    w_end = 16 * w_critical
    num_steps_half = num_steps // 2
    w_means_1 = np.linspace(w_start, w_mid, num_steps_half, endpoint=False)
    w_means_2 = np.linspace(w_mid, w_end, num_steps - num_steps_half)
    return np.concatenate([w_means_1, w_means_2])


def compute_critical_weight(simulation_params):
    """Compute the critical weight for the given parameters."""
    return simulation_params.membrane_threshold / (
        0.5 * simulation_params.small_world_graph_k
    ) - (2 * simulation_params.external_current) / (
        simulation_params.currents_period
        * simulation_params.num_neurons
        * 0.5
        * simulation_params.small_world_graph_k
    )


def run_simulation(simulation_params, weights_mean):
    """Run the simulation with the given parameters."""
    network = SNN(weights_mean, simulation_params)
    spike_matrix = network.simulate()
    return {
        "w_mean": weights_mean,
        "total_spikes": network.tot_spikes,
        "lzw_complexity": lzw_complexity_from_matrix(spike_matrix),
        "mean_isi": network.calculate_mean_isi(),
    }


def main():
    """Run simulation for a range of w_mean values, save the results and generate plots."""
    parser = argparse.ArgumentParser(description="Run simulation.")
    parser.add_argument(
        "--config",
        type=str,
        default="snn_config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/snn",
        help="Optional output directory name.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="If enabled, doesn't create a folder with results.",
    )

    args = parser.parse_args()

    if args.dry_run:
        logging.info("Dry run enabled. No output will be saved.")

    logging.info("Loading configuration from %s", args.config)
    config = load_config(args.config)
    simulation_params = SimulationParams(**config["simulation_params"])

    # Set seed for reproducibility
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    logging.info("Running simulations over a range of w_mean values.")
    w_critical = compute_critical_weight(simulation_params)

    w_means = create_weights_steps_array(w_critical, config["w_means_range_num_steps"])

    print(f"Running simulation with parameters: {simulation_params}")
    results: List[Dict[str, Any]] = []
    for w_mean in tqdm(w_means, desc="w_mean values"):
        for _ in range(config["experiment_repetitions"]):
            results.append(run_simulation(simulation_params, float(w_mean)))

    df_results = pd.DataFrame(results)

    if not args.dry_run:
        # Create output directory with a timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(args.output, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Saving results to %s", output_dir)

        # Save results
        df_results.to_csv(
            os.path.join(output_dir, "simulation_results.csv"), index=False
        )

        # Store the SNNParameters as a YAML file
        with open(
            os.path.join(output_dir, "snn_parameters.yaml"), "w", encoding="utf-8"
        ) as f:
            yaml.dump(config, f)

        logging.info("Plotting all results")
        plot_all_results(df_results, simulation_params)
        plt.savefig(os.path.join(output_dir, "simulation_plots.png"), dpi=300)
        logging.info(
            "All results plot saved to %s",
            os.path.join(output_dir, "simulation_all_plot.png"),
        )
        plt.show()

    logging.info("Simulation completed")


if __name__ == "__main__":
    main()
