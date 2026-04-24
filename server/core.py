"""
Internal engine — refactored from RL-TAS/step3_environment.py.
Core astronomy logic and episode management, not exposed via HTTP.
"""
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

OBSERVATORY = EarthLocation(lat=19.82 * u.deg, lon=-155.47 * u.deg, height=4207 * u.m)
SUNSET_BASE = datetime(2025, 3, 15, 18, 30)
SUNRISE_BASE = datetime(2025, 3, 16, 6, 0)

REWARD_SCALE = 475.0  # normalises raw rewards to roughly [-1, 1]

WEATHER_TRANSITION = [
    [0.8, 0.15, 0.05],
    [0.3, 0.5, 0.2],
    [0.1, 0.4, 0.5],
]


# ---------------------------------------------------------------------------
# Pure astronomy helpers (unchanged from step3_environment.py)
# ---------------------------------------------------------------------------

def altitude_from_ra_dec(ra: float, dec: float, time: datetime, location) -> float:
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    altaz = c.transform_to(AltAz(obstime=Time(time), location=location))
    return float(altaz.alt.deg)


def airmass_from_altitude(alt: float) -> float:
    if alt <= 0:
        return 3.0
    airmass = 1 / np.sin(np.radians(alt + 244 / (165 + 47 * alt ** 1.1)))
    return min(float(airmass), 3.0)


def next_weather(prev_state: int, rng: random.Random) -> int:
    """Markov weather transition using a seeded RNG."""
    row = WEATHER_TRANSITION[prev_state]
    roll = rng.random()
    cumulative = 0.0
    for i, p in enumerate(row):
        cumulative += p
        if roll < cumulative:
            return i
    return 2


def load_planet_dataframe() -> pd.DataFrame:
    obs_df = pd.read_csv(os.path.join(DATA_DIR, "phase1_agent_observable.csv"))
    pri_df = pd.read_csv(os.path.join(DATA_DIR, "phase2_priority_durations.csv"))
    merged = pd.merge(obs_df, pri_df, on="pl_name", how="left")
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal core class
# ---------------------------------------------------------------------------

class _TelescopeCore:
    """
    Drives one episode of telescope scheduling.
    Not an OpenEnv Environment — TelescopeSchedulingEnvironment wraps this.
    """

    def __init__(self, df: pd.DataFrame, task_config: dict):
        self.df_base = df.reset_index(drop=True)
        self.task_config = task_config
        self.location = OBSERVATORY
        self.num_planets = len(df)

        # Task-level settings
        self.max_steps: int = task_config.get("max_steps", 44)
        self.weather_locked: bool = task_config.get("weather_locked", False)
        self.start_offset_min: int = task_config.get("start_offset_min", 0)
        self.deadline_step_cutoff = task_config.get("deadline_step_cutoff", None)

        # Initialised by reset()
        self.rng = random.Random(0)
        self.planets: pd.DataFrame = df.copy()
        self.current_time: datetime = SUNSET_BASE
        self.sunrise: datetime = SUNRISE_BASE
        self.weather_state: int = 0
        self.episode_reward: float = 0.0
        self.done: bool = False
        self.step_count: int = 0
        self.deadlines_met_before_cutoff: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.current_time = SUNSET_BASE + timedelta(minutes=self.start_offset_min)
        self.sunrise = SUNRISE_BASE
        self.weather_state = 0 if self.weather_locked else self.rng.choice([0, 1, 2])
        self.planets = self.df_base.copy()
        self.planets["observed_tonight"] = False
        self.planets["times_observed"] = 0
        self.planets["soft_deadline_step"] = None
        self.planets["was_visible_tonight"] = False
        self.episode_reward = 0.0
        self.done = False
        self.step_count = 0
        self.deadlines_met_before_cutoff = 0
        self._set_time_cache: dict = {}   # cache for _estimate_set_time

        if self.deadline_step_cutoff is not None:
            self._inject_soft_deadlines()

    def step(self, action_target: int) -> tuple:
        """
        Returns: (normalised_reward: float, done: bool, info: dict)
        """
        if self.done:
            raise ValueError("Episode finished — call reset().")

        # Clamp to valid range — prevents IndexError on out-of-bounds LLM output
        action_target = max(0, min(self.num_planets, action_target))

        _EPS = 1e-4  # all rewards must be strictly in (0, 1) per validator
        reward = _EPS
        info: dict = {}

        if action_target == self.num_planets:
            # Wait action — small penalty to discourage idle steps.
            reward = _EPS
            info["action_type"] = "wait"
            info["planet_name"] = None
            info["reward_components"] = {}
        else:
            planet = self.planets.iloc[action_target]
            alt = altitude_from_ra_dec(
                planet.ra, planet.dec, self.current_time, self.location
            )
            airmass = airmass_from_altitude(alt)
            visible = alt > 30
            observable = visible and (self.weather_state != 2)

            if visible:
                self.planets.at[action_target, "was_visible_tonight"] = True

            if observable:
                if planet.observed_tonight:
                    # Already observed — no additional scientific value.
                    reward = _EPS
                    info["action_type"] = "observe"
                    info["planet_name"] = str(planet.pl_name)
                    info["reward_components"] = {
                        "base": 0.0,
                        "quality": 0.0,
                        "reason": "already_observed",
                    }
                    self.planets.at[action_target, "times_observed"] += 1
                else:
                    base_reward = float(planet.priority_score) * 10.0
                    quality_factor = 1.0

                    if self.weather_state == 1:
                        quality_factor *= 0.7
                    if alt < 45:
                        quality_factor *= 0.6
                    elif alt > 70:
                        quality_factor *= 1.2
                    if airmass > 2.0:
                        quality_factor *= 0.7
                    elif airmass < 1.3:
                        quality_factor *= 1.1

                    raw = base_reward * quality_factor

                    # Deadline bonus/penalty
                    deadline_mult = 1.0
                    if pd.notna(planet.get("deadline_time")):
                        dl = pd.to_datetime(planet.deadline_time)
                        deadline_mult = 1.3 if self.current_time <= dl else 0.5
                    elif planet.get("soft_deadline_step") is not None:
                        cutoff = planet["soft_deadline_step"]
                        deadline_mult = 1.3 if self.step_count <= cutoff else 0.5

                    raw *= deadline_mult
                    reward = raw / REWARD_SCALE

                    self.planets.at[action_target, "observed_tonight"] = True
                    self.planets.at[action_target, "times_observed"] += 1

                    # Track hard-task deadline satisfaction
                    if (
                        self.deadline_step_cutoff is not None
                        and planet.get("soft_deadline_step") is not None
                        and self.step_count <= self.deadline_step_cutoff
                    ):
                        self.deadlines_met_before_cutoff += 1

                    info["action_type"] = "observe"
                    info["planet_name"] = str(planet.pl_name)
                    info["altitude"] = round(alt, 2)
                    info["priority"] = int(planet.priority_score)
                    info["reward_components"] = {
                        "base": round(base_reward / REWARD_SCALE, 4),
                        "quality": round(quality_factor, 4),
                        "deadline_mult": round(deadline_mult, 4),
                        "normalised": round(reward, 4),
                    }
            else:
                reward = _EPS
                info["action_type"] = "invalid"
                info["planet_name"] = None
                info["reason"] = "below_horizon" if not visible else "bad_weather"
                info["reward_components"] = {}

        # Advance time
        self.current_time += timedelta(minutes=15)
        self.step_count += 1

        if not self.weather_locked:
            self.weather_state = next_weather(self.weather_state, self.rng)

        # Termination
        if self.current_time >= self.sunrise or self.step_count >= self.max_steps:
            self.done = True
            # End-of-night missed-deadline penalty — only for the hard task
            # (deadline_step_cutoff is set) and only for targets that were
            # actually reachable (visible at some point during the episode).
            for _, planet in self.planets.iterrows():
                if (
                    pd.notna(planet.get("deadline_time"))
                    and not planet.observed_tonight
                    and self.deadline_step_cutoff is not None
                    and planet.get("was_visible_tonight", False)
                ):
                    dl = pd.to_datetime(planet.deadline_time)
                    if dl <= self.current_time:
                        reward -= float(planet.priority_score) * 5.0 / REWARD_SCALE

        # Clamp reward to strictly (0, 1) — validator rejects 0.0, negatives, and >= 1.0
        reward = round(max(_EPS, min(reward, 1.0 - _EPS)), 4)

        self.episode_reward += reward
        return reward, self.done, info

    def get_planet_infos(self) -> list:
        """Return per-planet dicts for the current timestep (for building observations)."""
        # Pre-compute all altitudes once to avoid N redundant astropy calls
        alts = {
            idx: altitude_from_ra_dec(row.ra, row.dec, self.current_time, self.location)
            for idx, row in self.planets.iterrows()
        }

        infos = []
        for idx, row in self.planets.iterrows():
            alt = alts[idx]
            airmass = airmass_from_altitude(alt)
            visible = alt > 30
            time_until_set = self._estimate_set_time(row.ra, row.dec)

            has_deadline = pd.notna(row.get("deadline_time")) or (
                row.get("soft_deadline_step") is not None
            )
            deadline_status = "no_deadline"
            if pd.notna(row.get("deadline_time")):
                dl = pd.to_datetime(row.deadline_time)
                deadline_status = (
                    "before_deadline" if self.current_time <= dl else "past_deadline"
                )
            elif row.get("soft_deadline_step") is not None:
                cutoff = row["soft_deadline_step"]
                deadline_status = (
                    "before_deadline" if self.step_count <= cutoff else "past_deadline"
                )

            infos.append(
                {
                    "name": str(row.pl_name),
                    "priority_score": int(row.priority_score),
                    "altitude_deg": round(alt, 2),
                    "airmass": round(airmass, 3),
                    "visible": bool(visible),
                    "time_until_set_hr": round(time_until_set, 2),
                    "observed_tonight": bool(row.observed_tonight),
                    "has_deadline": bool(has_deadline),
                    "deadline_status": deadline_status,
                }
            )
        return infos

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_observed_tonight(self) -> int:
        return int(self.planets["observed_tonight"].sum())

    @property
    def total_priority_observed(self) -> float:
        mask = self.planets["observed_tonight"]
        return float(self.planets.loc[mask, "priority_score"].sum())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _inject_soft_deadlines(self) -> None:
        """Mark the top-3 currently-visible planets with a soft deadline step."""
        visible_alts = []
        for idx, row in self.planets.iterrows():
            alt = altitude_from_ra_dec(row.ra, row.dec, self.current_time, self.location)
            if alt > 30:
                visible_alts.append((idx, float(row.priority_score)))

        # Sort by priority descending, take top 3
        visible_alts.sort(key=lambda x: x[1], reverse=True)
        top3_indices = [idx for idx, _ in visible_alts[:3]]

        for idx in top3_indices:
            self.planets.at[idx, "soft_deadline_step"] = self.deadline_step_cutoff

    def _estimate_set_time(self, ra: float, dec: float, max_hours: int = 12) -> float:
        cache_key = (round(ra, 4), round(dec, 4), self.step_count)
        if cache_key in self._set_time_cache:
            return self._set_time_cache[cache_key]
        for minutes in range(0, max_hours * 60, 15):
            check_time = self.current_time + timedelta(minutes=minutes)
            alt = altitude_from_ra_dec(ra, dec, check_time, self.location)
            if alt < 30:
                result = minutes / 60.0
                self._set_time_cache[cache_key] = result
                return result
        self._set_time_cache[cache_key] = float(max_hours)
        return float(max_hours)
