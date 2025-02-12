"""Spiking Neural Network (SNN) model."""

import logging
import math
import random
from typing import List, Optional

from dataclasses import dataclass
import networkx as nx
import numpy as np


@dataclass
class SimulationParams:
    """Simulation parameters for the Spiking Neural Network (SNN)."""

    num_neurons: int
    membrane_threshold: float
    currents_period: float
    external_current: float
    leak_coefficient: float
    simulation_duration: int
    small_world_graph_p: float
    small_world_graph_k: float

    def __post_init__(self) -> None:
        """Post-initialization checks for the SimulationParams class."""
        self.validate_parameters()

    def validate_parameters(self) -> bool:
        """Validate the parameters based on constraints."""
        logging.info("Validating conditions with parameters: %s", self)

        if (
            not (0.01 <= self.currents_period <= 9.99)
            or not (self.currents_period * 100).is_integer()
        ):
            logging.warning(
                "Condition violated: currents_period must be between 0.01 and 9.99, with at most two decimal places."
            )
        if (
            self.currents_period
            * self.num_neurons
            * self.leak_coefficient
            * self.membrane_threshold
        ) != 0 and self.external_current / (
            self.currents_period
            * self.num_neurons
            * self.leak_coefficient
            * self.membrane_threshold
        ) < 1:
            logging.warning(
                "Condition violated: external_current / (currents_period * num_neurons * leak_coefficient * membrane_threshold) < 1"
            )
            return False
        if (
            self.currents_period * self.num_neurons * self.membrane_threshold
        ) != 0 and 2 * self.external_current / (
            self.currents_period * self.num_neurons * self.membrane_threshold
        ) > 1:
            logging.warning(
                "Condition violated: 2 * external_current / (currents_period * num_neurons * membrane_threshold) > 1"
            )
            return False
        if self.leak_coefficient != 0 and 1 / (2 * self.leak_coefficient) < 1:
            logging.warning("Condition violated: 1 / (2 * leak_coefficient) < 1")
            return False

        logging.info("All conditions validated successfully.")
        return True


class SNN:
    """Spiking Neural Network (SNN) model."""

    def __init__(
        self, weights_mean: float, simulation_params: SimulationParams
    ) -> None:
        """Initialize the spiking neural network (SNN) with parameters."""
        self.REFRACTORY_PERIOD = 10
        self.tot_spikes: int = 0
        self.leak_refractory_ratio: float = (
            simulation_params.leak_coefficient / self.REFRACTORY_PERIOD
        )
        self.num_neurons: int = simulation_params.num_neurons
        self.membrane_threshold: float = simulation_params.membrane_threshold
        self.current_period_times_refractory: float = (
            simulation_params.currents_period * self.REFRACTORY_PERIOD
        )
        self.external_current: float = simulation_params.external_current
        self.simulation_duration: int = simulation_params.simulation_duration
        self.time_step: int = 1
        self.weights_mean: float = weights_mean

        self.membrane_potentials: np.ndarray = np.random.uniform(
            0, self.membrane_threshold, self.num_neurons
        )
        self.spike_times: List[List[int]] = [
            [] for _ in range(simulation_params.num_neurons)
        ]
        self.avg_in_degree: Optional[float] = None
        self.spike_matrix: Optional[np.ndarray] = None

        self.generate_synaptic_weights(
            simulation_params.small_world_graph_p, simulation_params.small_world_graph_k
        )

        self.refractory_timer: np.ndarray = np.zeros(simulation_params.num_neurons)

    def generate_synaptic_weights(
        self,
        small_world_graph_p: float = 0.1,
        small_world_graph_k: int = 10,
        weights_scale_factor: float = 0.1,
    ) -> None:
        """Generate synaptic weights based on a small-world graph."""
        small_world_graph = nx.watts_strogatz_graph(
            n=self.num_neurons, k=small_world_graph_k, p=small_world_graph_p, seed=None
        )
        synaptic_weights = np.zeros((self.num_neurons, self.num_neurons))

        for edge in small_world_graph.edges():
            i, j = edge
            if np.random.rand() < 0.5:
                synaptic_weights[i, j] = np.random.normal(
                    loc=self.weights_mean,
                    scale=abs(self.weights_mean) * weights_scale_factor,
                )
            else:
                synaptic_weights[j, i] = np.random.normal(
                    loc=self.weights_mean,
                    scale=abs(self.weights_mean) * weights_scale_factor,
                )

        np.fill_diagonal(synaptic_weights, 0)
        self.synaptic_weights: np.ndarray = synaptic_weights

        in_degrees = np.count_nonzero(self.synaptic_weights, axis=0)
        self.avg_in_degree = in_degrees.mean()

    def in_degree(self) -> Optional[float]:
        """Return the average in-degree of the network."""
        return self.avg_in_degree

    def stimulate_neuron(self, num_stimulated_neurons: int) -> None:
        """Stimulate a specified number of neurons."""
        target_neurons = random.sample(range(self.num_neurons), num_stimulated_neurons)
        self.membrane_potentials[target_neurons] += self.external_current

    def simulate(self) -> Optional[np.ndarray]:
        """Run the simulation of the SNN."""
        self.tot_spikes = 0
        self.spike_matrix = np.zeros(
            (self.simulation_duration, self.num_neurons), dtype=int
        )
        integer_current_period_times_refractory = int(
            self.current_period_times_refractory
        )
        greatest_common_divisor = math.gcd(
            int(self.current_period_times_refractory * 10), 10
        )
        tau_n = int(self.current_period_times_refractory * 10 / greatest_common_divisor)
        tau_d = int(10 / greatest_common_divisor)
        currents_counter = 0

        for t in range(self.simulation_duration):
            self.refractory_timer = np.maximum(
                0, self.refractory_timer - self.time_step
            )

            if self.current_period_times_refractory < 1:
                if currents_counter % tau_n == 0:
                    self.stimulate_neuron(tau_d)
            if self.current_period_times_refractory >= 1:
                if (
                    integer_current_period_times_refractory > 0
                    and currents_counter % integer_current_period_times_refractory == 0
                ):
                    self.stimulate_neuron(1)
                if currents_counter == self.current_period_times_refractory * 10:
                    currents_counter = 0
            currents_counter += 1

            spiking_neurons = (self.membrane_potentials >= self.membrane_threshold) & (
                self.refractory_timer == 0
            )
            self.spike_matrix[t, :] = spiking_neurons.astype(int)
            self.tot_spikes += np.sum(spiking_neurons)

            for idx in np.where(spiking_neurons)[0]:
                self.spike_times[idx].append(t)

            self.membrane_potentials[spiking_neurons] = 0
            self.refractory_timer[spiking_neurons] = self.REFRACTORY_PERIOD + 1
            self.membrane_potentials = (
                1 - self.leak_refractory_ratio
            ) * self.membrane_potentials + spiking_neurons.astype(
                float
            ) @ self.synaptic_weights
        return self.spike_matrix

    def calculate_mean_isi(self) -> float:
        """Calculate the mean inter-spike interval (ISI)."""
        total_inter_spike_intervals = []
        for spike_times in self.spike_times:
            if len(spike_times) > 1:
                inter_spike_interval = np.diff(spike_times)
                total_inter_spike_intervals.extend(inter_spike_interval)
        if total_inter_spike_intervals:
            mean_inter_spike_interval = np.mean(total_inter_spike_intervals)
        else:
            mean_inter_spike_interval = self.simulation_duration
        return float(mean_inter_spike_interval) / self.REFRACTORY_PERIOD


def lzw_complexity_from_matrix(matrix: np.ndarray) -> int:
    """
    Calculate the Lempel-Ziv-Welch (LZW) complexity of a vector created by
    concatenating the columns of a 2D matrix.
    Args:
        matrix (np.ndarray): A 2D NumPy array representing spike data
                             (rows typically time, columns neurons).
    Returns:
        int: The LZW complexity of the concatenated sequence.
    """

    def lzw(seq: str) -> int:
        """
        Calculate the LZW complexity of a binary (string) sequence.
        Args:
            seq (str): The sequence string (e.g., '101001...').
        Returns:
            int: The size of the generated dictionary, representing
                 the LZW complexity.
        """
        dictionary = {}
        w = ""
        for c in seq:
            wc = w + c
            if wc not in dictionary:
                dictionary[wc] = len(dictionary)
                w = c
            else:
                w = wc
        return len(dictionary)

    # Transpose and then flatten to read column by column
    vector = matrix.T.flatten()
    # Convert the vector into a string
    vector_str = "".join(map(str, vector))
    # Calculate the LZW complexity
    complexity = lzw(vector_str)
    return complexity
