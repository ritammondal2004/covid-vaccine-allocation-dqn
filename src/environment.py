"""
VaxOpt: COVID-19 Vaccine Allocation RL Environment.
"""

import copy
import math
import numpy as np
import pandas as pd


class StatesEnv:
    """   
    Action space:
        Discrete(buckets), where action a in {0, ..., buckets-1}.
        Fraction allocated: A = a / buckets.
        Vials distributed: V = A * total_vials.
               
    Reward:
        R = 100 * exp( - (A - S)^2 / 1e-4 )
        where S is the susceptible ratio (feature index 4).
    """

    def __init__(
        self,
        confirmed_ratio: float,
        dr: float,
        rr: float,
        population_ratio: float,
        susceptible_ratio: float,
        bed_ratio: float,
        icu_ratio: float,
        ventilator_ratio: float,
        old_ratio: float,
        buckets: int = 1000,
        total_vials: float = 1000000.0,
        total_susceptible: float = 1e8,
        episodes: int = 1,
    ):
        self.buckets = buckets
        self.total_vials = float(total_vials)
        self.total_susceptible = float(total_susceptible)
        self.episodes = episodes

        # Store initial context features
        self.confirmed_ratio = float(confirmed_ratio)
        self.dr = float(dr)
        self.rr = float(rr)
        self.population_ratio = float(population_ratio)
        self.susceptible_ratio = float(susceptible_ratio)
        self.bed_ratio = float(bed_ratio)
        self.icu_ratio = float(icu_ratio)
        self.ventilator_ratio = float(ventilator_ratio)
        self.old_ratio = float(old_ratio)

        self.state_dim = 9
        self.action_dim = buckets

        self.curr_step = 1
        self.done = False
        self.state = None
        self.reset()

    def reset(self) -> np.ndarray:
        """Resets the environment state to initial context values."""
        self.curr_step = 1
        self.done = False
        self.state = np.array(
            [
                self.confirmed_ratio,
                self.dr,
                self.rr,
                self.population_ratio,
                self.susceptible_ratio,
                self.bed_ratio,
                self.icu_ratio,
                self.ventilator_ratio,
                self.old_ratio,
            ],
            dtype=np.float32,
        )
        return copy.deepcopy(self.state)

    def calculate_reward(self, action: int) -> float:
        """
        Calculates reward as specified in Paper Equation (10):
        R_i = 100 * exp( - (A_i - S_i)^2 / 10^-4 )
        """
        a_fraction = float(action) / float(self.buckets)
        s_fraction = float(self.state[4])  # Susceptible ratio
        diff = a_fraction - s_fraction
        reward = 100.0 * math.exp(-(diff**2) / 0.0001)
        return float(reward)

    def step(self, action: int):
        """
        Executes one allocation step.
        Returns:
            next_state (np.ndarray): updated context vector
            reward (float): scalar reward from allocation
            done (bool): whether the episode has terminated
            info (dict): diagnostic information
        """
        if self.done:
            raise RuntimeError("Environment step called after episode is done. Call reset().")

        action = int(action)
        reward = self.calculate_reward(action)

        # Fraction and vials allocated
        a_fraction = float(action) / float(self.buckets)
        vaccine_allotted = a_fraction * self.total_vials

        # Update susceptible population count and ratio
        current_state_susc = self.state[4] * self.total_susceptible
        new_state_susc = max(0.0, current_state_susc - vaccine_allotted)
        remaining_total_susc = max(1.0, self.total_susceptible - vaccine_allotted)
        new_susceptible_ratio = new_state_susc / remaining_total_susc

        # Update observation feature 4
        self.state[4] = np.clip(new_susceptible_ratio, 0.0, 1.0)

        if self.curr_step >= self.episodes:
            self.done = True
        else:
            self.curr_step += 1

        info = {
            "action": action,
            "allocation_fraction": a_fraction,
            "vaccine_allotted": vaccine_allotted,
            "initial_susceptible_ratio": self.susceptible_ratio,
            "new_susceptible_ratio": float(self.state[4]),
            "step": self.curr_step,
        }

        return copy.deepcopy(self.state), reward, self.done, info


def extract_context_data(csv_path: str):
    """
    Returns:
        dates (list of str): Unique dates in order
        state_names (list of str): Names of the 5 states
        daily_contexts (dict): Mapping from date to list of dicts (one per state)
    """
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    dates = df["Date"].unique().tolist()
    daily_contexts = {}

    for date in dates:
        day_df = df[df["Date"] == date].copy().reset_index(drop=True)
        states = day_df["State / UT"].tolist()

        arr = day_df.to_numpy()
        # Column mappings based on projections.csv:
        # 0: Date, 1: State / UT, 2: Confirmed, 3: Death Rate, 4: Recovery Rate,
        # 5: Population, 6: Susceptible, 7: Beds, 8: ICU, 9: Ventilators, 10: Age Distribution

        total_confirmed = max(float(np.sum(arr[:, 2])), 1.0)
        total_population = max(float(np.sum(arr[:, 5])), 1.0)
        total_susceptible = max(float(np.sum(arr[:, 6])), 1.0)
        total_beds = max(float(np.sum(arr[:, 7])), 1.0)
        total_icu = max(float(np.sum(arr[:, 8])), 1.0)
        total_venti = max(float(np.sum(arr[:, 9])), 1.0)
        total_old = max(float(np.sum(arr[:, 10])), 1.0)

        state_records = []
        for s_idx, state_name in enumerate(states):
            record = {
                "date": date,
                "state": state_name,
                "confirmed_ratio": float(arr[s_idx, 2]) / total_confirmed,
                "dr": float(arr[s_idx, 3]),
                "rr": float(arr[s_idx, 4]),
                "population_ratio": float(arr[s_idx, 5]) / total_population,
                "susceptible_ratio": float(arr[s_idx, 6]) / total_susceptible,
                "bed_ratio": float(arr[s_idx, 7]) / total_beds,
                "icu_ratio": float(arr[s_idx, 8]) / total_icu,
                "ventilator_ratio": float(arr[s_idx, 9]) / total_venti,
                "old_ratio": float(arr[s_idx, 10]) / total_old,
                "total_susceptible": total_susceptible,
                "raw_confirmed": float(arr[s_idx, 2]),
                "raw_susceptible": float(arr[s_idx, 6]),
                "raw_population": float(arr[s_idx, 5]),
            }
            state_records.append(record)

        daily_contexts[date] = state_records

    # Return ordered state names from first date
    first_date = dates[0]
    state_names = [r["state"] for r in daily_contexts[first_date]]

    return dates, state_names, daily_contexts


def create_env_from_record(record: dict, buckets: int = 1000, total_vials: float = 1000000.0) -> StatesEnv:
    """Instantiates a StatesEnv from a processed state record dict."""
    return StatesEnv(
        confirmed_ratio=record["confirmed_ratio"],
        dr=record["dr"],
        rr=record["rr"],
        population_ratio=record["population_ratio"],
        susceptible_ratio=record["susceptible_ratio"],
        bed_ratio=record["bed_ratio"],
        icu_ratio=record["icu_ratio"],
        ventilator_ratio=record["ventilator_ratio"],
        old_ratio=record["old_ratio"],
        buckets=buckets,
        total_vials=total_vials,
        total_susceptible=record["total_susceptible"],
        episodes=1,
    )
