'''
This module is planned to be a toolbox for data loading and preprocessing
@author: Chengyi Ma

'''

import pandas as pd
import numpy as np

class DataManager:
    '''
    DataManager is a basic class that manages the data loading and preprocessing
    '''
    def __init__(self):
        self.data = None

    def load_dataset_EvEye_raw(self, data_path: str) -> pd.DataFrame:
        '''
        Load dataset in Ev-Eye format. The raw data is in txt format.
        Each line is an event, with the following format:
        timestamp (microseconds), x, y, polarity
        And the txt file is usually very large.
        Args:
            data_path: the path to the txt file
        Returns:
            a pandas dataframe with the following columns:
            x, y, polarity and timestamp
        '''
        self.data = pd.read_csv(
            data_path,
            header=None,
            names=["timestamp", "x", "y", "polarity"],
            dtype={"timestamp": "int64", "x": "int16", "y": "int16", "polarity": "bool"},
            sep=" ",
            engine="c",
            skip_blank_lines=True,
        )
        self.data["timestamp"] = self.data["timestamp"] - self.data["timestamp"].min()
        return self.data

    def events_to_points_window(self, data: pd.DataFrame, time_window: list[int], time_conversion_factor: float) -> pd.DataFrame:
        '''
        Convert events to points in a given time window
        Args:
            data: the dataframe containing the events
            time_window: the time window in microseconds [start, end]
            time_conversion_factor: the factor to convert the timestamp to 3D space-time coordinates
        Returns:
            a dataframe containing the points with columns: x, y, t, polarity
        '''
        mask = (data["timestamp"] >= time_window[0]) & (data["timestamp"] <= time_window[1])
        points = data.loc[mask, ["x", "y", "polarity"]].copy()
        points["t"] = ((data.loc[mask, "timestamp"] - time_window[0]) * time_conversion_factor).astype("float32")
        return points[["x", "y", "t", "polarity"]]

    def evaluate_hotpixels_event_rate(self, data: pd.DataFrame, z_thresh: float = 3.5) -> list[tuple[int, int, float]]:
        '''
        Evaluate the event rate of every pixel and identify hot-pixel outliers.

        Per-pixel event rates span many orders of magnitude, so the Modified Z-Score
        is computed on log10-transformed rates. This compresses the heavy right skew
        into an approximately symmetric distribution and gives a clean separation
        between true hardware hot pixels and legitimately active scene pixels.

            log_rate     = log10(event_rate_hz)
            Modified Z   = 0.6745 * (log_rate - median(log_rate)) / MAD(log_rate)

        Pixels whose Modified Z-Score exceeds z_thresh (default 3.5, Iglewicz &
        Hoaglin 1993) are flagged as hot pixels.

        Args:
            data: dataframe with columns timestamp (µs), x, y, polarity
            z_thresh: Modified Z-Score threshold on log10 scale (default 3.5)
        Returns:
            list of (x, y, event_rate_hz) tuples for hot pixels, sorted by rate descending
        '''
        duration_s = (data["timestamp"].max() - data["timestamp"].min()) / 1e6

        # Event rate (Hz) per pixel
        rates = (data.groupby(["x", "y"], observed=True)
                     .size()
                     .rename("rate")
                     .astype("float64") / duration_s)

        log_rates = np.log10(rates)
        median_log = log_rates.median()
        mad_log    = (log_rates - median_log).abs().median()

        if mad_log == 0:
            # Degenerate: all pixels have the same rate — no hot pixels possible
            hot_mask = pd.Series(False, index=rates.index)
        else:
            modified_z = 0.6745 * (log_rates - median_log) / mad_log
            hot_mask   = modified_z > z_thresh

        hot_pixels   = rates[hot_mask].sort_values(ascending=False)
        median_rate  = rates.median()
        thresh_hz    = 10 ** (median_log + z_thresh * mad_log / 0.6745)

        # --- report ---
        print(f"{'='*57}")
        print(f"  Hot Pixel Report")
        print(f"{'='*57}")
        print(f"  Recording duration  : {duration_s:.2f} s")
        print(f"  Active pixels       : {len(rates):,}")
        print(f"  Median event rate   : {median_rate:.4f} Hz")
        print(f"  MAD (log10 scale)   : {mad_log:.4f}")
        print(f"  Threshold (Z>{z_thresh})  : {thresh_hz:.2f} Hz")
        print(f"  Hot pixels found    : {len(hot_pixels)}")
        print(f"{'='*57}")
        if len(hot_pixels):
            print(f"  {'x':>5}  {'y':>5}  {'rate (Hz)':>12}  {'× median':>10}")
            print(f"  {'-'*5}  {'-'*5}  {'-'*12}  {'-'*10}")
            for (x, y), rate in hot_pixels.items():
                print(f"  {x:>5}  {y:>5}  {rate:>12.1f}  {rate/median_rate:>9.0f}×")
        else:
            print("  No hot pixels detected.")
        print(f"{'='*57}")

        return [(int(x), int(y), float(r)) for (x, y), r in hot_pixels.items()]

    def remove_hot_pixels(self, data: pd.DataFrame, hot_pixels: list[tuple[int, int, float]]) -> pd.DataFrame:
        '''
        Remove hot pixels from the data
        Args:
            data: the dataframe containing the events
            hot_pixels: the list of hot pixels
        Returns:
            a dataframe containing the events without the hot pixels
        '''
        for pixel in hot_pixels:
            data = data[~((data["x"] == pixel[0]) & (data["y"] == pixel[1]))]
        return data