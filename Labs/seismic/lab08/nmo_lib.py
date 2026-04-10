"""
nmo_lib.py
==========
Utility library for seismic NMO (Normal Moveout) analysis.

Provides functions for:
  - Loading multi-shot seismic gathers
  - Midpoint calculation
  - CMP gather construction
  - Semblance (velocity spectrum) computation
  - NMO correction
  - Far-offset muting
  - Stacking

"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_shot_gathers(data_folder: str, pattern: str = "shot_*.npz") -> list[dict]:
    """
    Load all shot gathers matching *pattern* inside *data_folder*.

    Parameters
    ----------
    data_folder : str
        Path to the directory containing .npz shot files.
    pattern : str
        Glob pattern for shot files (default: 'shot_*.npz').

    Returns
    -------
    shots : list of dict
        Each dict contains:
            rx, rz  – receiver x/z positions (m)
            sx, sz  – source x/z position (m)
            dt      – time-step (ms)
            data    – seismic array, shape (n_samples, n_traces)
    """
    fpaths = sorted(glob.glob(os.path.join(data_folder, pattern)))
    if not fpaths:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in '{data_folder}'."
        )

    shots = []
    for fp in fpaths:
        raw = np.load(fp, allow_pickle=True)
        shots.append({
            "rx":   raw["rx"],
            "rz":   raw["rz"],
            "sx":   float(raw["sx"]),
            "sz":   float(raw["sz"]),
            "dt":   float(raw["dt"]),
            "data": raw["data"],          # shape (n_samples, n_traces)
            "file": os.path.basename(fp),
        })

    print(f"Loaded {len(shots)} shot gather(s) from '{data_folder}'.")
    return shots


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Midpoint Calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_midpoints(shots: list[dict]) -> tuple[np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray,
                                                   np.ndarray, list]:
    """
    Compute common midpoint (CMP) x-coordinate for every trace in every shot.

    Parameters
    ----------
    shots : list of dict
        Output of :func:`load_shot_gathers`.

    Returns
    -------
    all_cmps     : (N,)  CMP x-coordinate for each trace
    all_offsets  : (N,)  absolute source-receiver offset (m)
    all_times    : list[np.ndarray]  – kept for convenience (per-shot)
    shot_indices : (N,)  which shot each trace belongs to
    trace_indices: (N,)  local trace index within that shot
    dt_list      : list of dt values, one per shot
    """
    all_cmps, all_offsets, shot_idx, trc_idx = [], [], [], []
    dt_list = []

    for s_i, shot in enumerate(shots):
        rx = shot["rx"]
        sx = shot["sx"]
        cmps = 0.5 * (rx + sx)          # midpoint x
        offsets = np.abs(rx - sx)        # absolute offset

        all_cmps.append(cmps)
        all_offsets.append(offsets)
        shot_idx.append(np.full(len(rx), s_i, dtype=int))
        trc_idx.append(np.arange(len(rx), dtype=int))
        dt_list.append(shot["dt"])

    all_cmps    = np.concatenate(all_cmps)
    all_offsets = np.concatenate(all_offsets)
    shot_idx    = np.concatenate(shot_idx)
    trc_idx     = np.concatenate(trc_idx)

    return all_cmps, all_offsets, shot_idx, trc_idx, dt_list


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CMP Gather Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_cmp_gather(
    shots: list[dict],
    all_cmps: np.ndarray,
    all_offsets: np.ndarray,
    shot_idx: np.ndarray,
    trc_idx: np.ndarray,
    cmp_center: float,
    bin_size: float,
) -> dict:
    """
    Collect all traces whose CMP x-coordinate falls within
    [cmp_center - bin_size/2, cmp_center + bin_size/2].

    Parameters
    ----------
    shots       : list of shot dicts
    all_cmps    : (N,) CMP x-coordinates
    all_offsets : (N,) offsets
    shot_idx    : (N,) shot index per trace
    trc_idx     : (N,) local trace index per trace
    cmp_center  : float  – target CMP location (m)
    bin_size    : float  – full bin width (m)

    Returns
    -------
    gather : dict with keys
        data     – (n_samples, n_traces_in_bin) array, sorted by offset
        offsets  – (n_traces_in_bin,) sorted offsets
        dt       – time step (ms)
        cmp      – cmp_center
        bin_size – bin_size
    """
    half = bin_size / 2.0
    mask = np.abs(all_cmps - cmp_center) <= half

    if not np.any(mask):
        raise ValueError(
            f"No traces found within ±{half} m of CMP={cmp_center} m."
        )

    s_ids  = shot_idx[mask]
    t_ids  = trc_idx[mask]
    offs   = all_offsets[mask]

    # Use dt from first shot (assume consistent)
    dt = shots[s_ids[0]]["dt"]
    n_samples = shots[s_ids[0]]["data"].shape[0]

    # Assemble traces
    traces = np.column_stack([
        shots[si]["data"][:, ti] for si, ti in zip(s_ids, t_ids)
    ])  # shape: (n_samples, n_traces_in_bin)

    # Sort by offset
    order  = np.argsort(offs)
    traces = traces[:, order]
    offs   = offs[order]

    print(f"CMP gather @ {cmp_center} m: {traces.shape[1]} traces, "
          f"offsets {offs.min():.0f}–{offs.max():.0f} m")

    return {
        "data":     traces,
        "offsets":  offs,
        "dt":       dt,
        "cmp":      cmp_center,
        "bin_size": bin_size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Semblance (Velocity Spectrum)
# ─────────────────────────────────────────────────────────────────────────────

def compute_semblance(
    gather: dict,
    v_min: float,
    v_max: float,
    n_v: int,
    t0_min: float = None,
    t0_max: float = None,
    half_win: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the semblance (coherence) panel for a CMP gather.

    For each (t0, v_rms) pair the NMO-shifted traces are summed and
    coherence is measured as::

        S(t0, v) = [ sum_x A(t_x) ]^2  /  [ N * sum_x A(t_x)^2 ]

    where t_x = sqrt(t0^2 + x^2/v^2) is the NMO travel-time and
    the sum is performed over a short time window centred on t0.

    Parameters
    ----------
    gather    : dict from :func:`build_cmp_gather`
    v_min     : float  – minimum velocity to test (m/s)
    v_max     : float  – maximum velocity to test (m/s)
    n_v       : int    – number of velocity samples
    t0_min    : float  – start of t0 axis (ms); default 0
    t0_max    : float  – end   of t0 axis (ms); default full record
    half_win  : int    – half-length of coherence window (samples)

    Returns
    -------
    semblance : (n_t0, n_v)  semblance values in [0, 1]
    t0_axis   : (n_t0,)  zero-offset times (ms)
    v_axis    : (n_v,)   velocities (m/s)
    """
    data    = gather["data"]          # (n_samp, n_trc)
    offsets = gather["offsets"]       # (n_trc,)  in metres
    dt_ms   = gather["dt"]            # ms
    dt_s    = dt_ms * 1e-3            # seconds
    n_samp, n_trc = data.shape

    full_t_ms = np.arange(n_samp) * dt_ms

    t0_min = t0_min if t0_min is not None else 0.0
    t0_max = t0_max if t0_max is not None else full_t_ms[-1]

    t0_axis = full_t_ms[(full_t_ms >= t0_min) & (full_t_ms <= t0_max)]
    v_axis  = np.linspace(v_min, v_max, n_v)

    semblance = np.zeros((len(t0_axis), n_v), dtype=np.float32)

    for iv, v in enumerate(v_axis):
        for it0, t0_ms in enumerate(t0_axis):
            t0_s = t0_ms * 1e-3
            # NMO times for each offset (seconds)
            t_nmo_s = np.sqrt(t0_s**2 + (offsets / v)**2)
            t_nmo_ms = t_nmo_s * 1e3

            # Accumulate windowed semblance
            num = np.zeros(2 * half_win + 1)
            den = np.zeros(2 * half_win + 1)

            for dw in range(-half_win, half_win + 1):
                amp = np.zeros(n_trc)
                for ix in range(n_trc):
                    t_shifted = t_nmo_ms[ix] + dw * dt_ms
                    isamp = t_shifted / dt_ms
                    i0 = int(isamp)
                    frac = isamp - i0
                    if 0 <= i0 < n_samp - 1:
                        amp[ix] = ((1 - frac) * data[i0, ix]
                                   + frac * data[i0 + 1, ix])
                    elif 0 <= i0 < n_samp:
                        amp[ix] = data[i0, ix]
                    # else: 0 (out of bounds)

                idx = dw + half_win
                num[idx] = amp.sum() ** 2
                den[idx] = n_trc * (amp ** 2).sum()

            total_den = den.sum()
            if total_den > 0:
                semblance[it0, iv] = num.sum() / total_den

    return semblance, t0_axis, v_axis


# ─────────────────────────────────────────────────────────────────────────────
# 5.  NMO Correction
# ─────────────────────────────────────────────────────────────────────────────

def nmo_correction(
    gather: dict,
    t0_picks: np.ndarray,
    vrms_picks: np.ndarray,
) -> np.ndarray:
    """
    Apply NMO correction to a CMP gather using picked t0–Vrms pairs.

    Vrms is interpolated linearly between picks; constant extrapolation
    is used outside the pick range.

    Parameters
    ----------
    gather     : dict from :func:`build_cmp_gather`
    t0_picks   : (n_picks,)  picked zero-offset times (ms)
    vrms_picks : (n_picks,)  corresponding Vrms values (m/s)

    Returns
    -------
    corrected : (n_samples, n_traces)  NMO-corrected gather
    """
    data    = gather["data"]
    offsets = gather["offsets"]
    dt_ms   = gather["dt"]
    n_samp, n_trc = data.shape
    t_axis_ms = np.arange(n_samp) * dt_ms

    # Build continuous Vrms profile by linear interpolation
    vrms_interp = interp1d(
        t0_picks, vrms_picks,
        kind="linear", bounds_error=False,
        fill_value=(vrms_picks[0], vrms_picks[-1]),
    )
    vrms_profile = vrms_interp(t_axis_ms)   # (n_samp,)

    corrected = np.zeros_like(data)

    for ix, x in enumerate(offsets):
        for it0 in range(n_samp):
            t0_s  = t_axis_ms[it0] * 1e-3
            v     = vrms_profile[it0]
            if v <= 0:
                continue
            t_nmo_s  = np.sqrt(t0_s**2 + (x / v)**2)
            t_nmo_ms = t_nmo_s * 1e3

            # Linear interpolation in original trace
            isamp = t_nmo_ms / dt_ms
            i0    = int(isamp)
            frac  = isamp - i0
            if 0 <= i0 < n_samp - 1:
                corrected[it0, ix] = ((1 - frac) * data[i0, ix]
                                      + frac * data[i0 + 1, ix])
            elif i0 == n_samp - 1:
                corrected[it0, ix] = data[i0, ix]

    return corrected


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Far-Offset Mute
# ─────────────────────────────────────────────────────────────────────────────

def apply_far_offset_mute(
    corrected: np.ndarray,
    offsets: np.ndarray,
    max_offset: float,
) -> np.ndarray:
    """
    Zero out traces whose offset exceeds *max_offset*.

    Parameters
    ----------
    corrected  : (n_samples, n_traces)  NMO-corrected gather
    offsets    : (n_traces,)  offset per trace (m)
    max_offset : float  – mute threshold (m)

    Returns
    -------
    muted : (n_samples, n_traces)  gather after mute
    """
    muted = corrected.copy()
    muted[:, offsets > max_offset] = 0.0
    n_kept = int(np.sum(offsets <= max_offset))
    print(f"Far-offset mute: kept {n_kept}/{len(offsets)} traces "
          f"(max offset = {max_offset:.0f} m)")
    return muted


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Stacking
# ─────────────────────────────────────────────────────────────────────────────

def stack_gather(muted: np.ndarray, offsets: np.ndarray,
                 max_offset: float) -> np.ndarray:
    """
    Simple mean-stack of non-muted traces.

    Parameters
    ----------
    muted      : (n_samples, n_traces)  muted NMO gather
    offsets    : (n_traces,)
    max_offset : float

    Returns
    -------
    stack : (n_samples,)  stacked trace
    """
    live = offsets <= max_offset
    if not np.any(live):
        raise ValueError("No live traces to stack — check max_offset.")
    stack = muted[:, live].mean(axis=1)
    return stack


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Plotting Helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_shot_gather(shot: dict, clip_pct: float = 98, ax=None):
    """Wiggle/image plot of a single shot gather."""
    data = shot["data"]
    dt   = shot["dt"]
    rx   = shot["rx"]
    sx   = shot["sx"]
    n_samp, n_trc = data.shape
    t_axis = np.arange(n_samp) * dt

    vmax = np.percentile(np.abs(data), clip_pct)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(
        data, aspect="auto", cmap="seismic",
        vmin=-vmax, vmax=vmax,
        extent=[0, n_trc, t_axis[-1], 0],
    )
    ax.set_xlabel("Trace index")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Shot gather  |  sx = {sx:.0f} m")
    return ax


def plot_cmp_gather(gather: dict, title: str = "", clip_pct: float = 98, ax=None):
    """Image plot of a CMP gather sorted by offset."""
    data    = gather["data"]
    offsets = gather["offsets"]
    dt      = gather["dt"]
    n_samp  = data.shape[0]
    t_axis  = np.arange(n_samp) * dt
    vmax = np.percentile(np.abs(data), clip_pct)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(
        data, aspect="auto", cmap="seismic",
        vmin=-vmax, vmax=vmax,
        extent=[offsets[0], offsets[-1], t_axis[-1], 0],
    )
    ax.set_xlabel("Offset (m)")
    ax.set_ylabel("Two-way time (ms)")
    ax.set_title(title or f"CMP gather @ {gather['cmp']:.0f} m")
    return ax


def plot_semblance(
    semblance: np.ndarray,
    t0_axis: np.ndarray,
    v_axis: np.ndarray,
    picks_t0: np.ndarray = None,
    picks_v:  np.ndarray = None,
    ax=None,
):
    """
    Display semblance panel with optional velocity picks overlaid.

    Parameters
    ----------
    semblance : (n_t0, n_v)
    t0_axis   : (n_t0,) ms
    v_axis    : (n_v,)  m/s
    picks_t0  : optional picked t0 values (ms)
    picks_v   : optional picked Vrms values (m/s)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 8))

    im = ax.imshow(
        semblance, aspect="auto", cmap="viridis",
        vmin=0, vmax=semblance.max(),
        extent=[v_axis[0], v_axis[-1], t0_axis[-1], t0_axis[0]],
        origin="upper",
    )
    plt.colorbar(im, ax=ax, label="Semblance")

    if picks_t0 is not None and picks_v is not None:
        ax.plot(picks_v, picks_t0, "c^-", ms=8, lw=1.5,
                label="Vrms picks")
        ax.legend(loc="upper right")

    ax.set_xlabel("Vrms (m/s)")
    ax.set_ylabel("t₀ (ms)")
    ax.set_title("Semblance panel")
    return ax


def plot_vrms_profile(
    t0_picks: np.ndarray,
    vrms_picks: np.ndarray,
    t0_full: np.ndarray = None,
    vrms_full: np.ndarray = None,
    ax=None,
):
    """Plot Vrms–t0 interval profile."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 8))

    if t0_full is not None and vrms_full is not None:
        ax.plot(vrms_full, t0_full, "gray", lw=1.2, label="Interpolated")

    ax.plot(vrms_picks, t0_picks, "rs-", ms=8, lw=1.5, label="Picks")
    ax.invert_yaxis()
    ax.set_xlabel("Vrms (m/s)")
    ax.set_ylabel("t₀ (ms)")
    ax.set_title("Vrms – t₀ Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_stack_trace(stack: np.ndarray, dt: float, ax=None):
    """Plot a single stacked trace as a wiggle."""
    n_samp = len(stack)
    t_axis = np.arange(n_samp) * dt

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 8))

    ax.plot(stack, t_axis, "k", lw=0.8)
    ax.fill_betweenx(t_axis, 0, stack, where=(stack > 0),
                     color="black", alpha=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Two-way time (ms)")
    ax.set_title("Stacked Trace")
    ax.grid(True, alpha=0.3)
    return ax
