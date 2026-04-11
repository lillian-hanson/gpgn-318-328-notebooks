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
  - 2-D seismic section assembly (NEW)
  - Time-to-depth conversion           (NEW)
  - Interval velocity from Dix equation (NEW)
  - Plotting helpers for 2-D section and depth image (NEW)
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
# 8.  2-D Section Assembly  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def get_cmp_bin_centers(
    all_cmps: np.ndarray,
    bin_size: float,
    min_fold: int = 3,
) -> np.ndarray:
    """
    Return the centre x-coordinate of every CMP bin that has at least
    *min_fold* contributing traces.

    Bins are defined on a regular grid with spacing *bin_size*, anchored
    at the rounded minimum CMP location.

    Parameters
    ----------
    all_cmps : (N,)  CMP x-coordinate for every trace in the survey
    bin_size : float  full bin width (m)
    min_fold : int    minimum number of traces required to include a bin

    Returns
    -------
    centers : (M,)  sorted array of qualifying bin-centre x-coordinates
    """
    x_min = np.floor(all_cmps.min() / bin_size) * bin_size
    x_max = np.ceil(all_cmps.max()  / bin_size) * bin_size
    edges = np.arange(x_min, x_max + bin_size, bin_size)
    centers = 0.5 * (edges[:-1] + edges[1:])

    qualifying = []
    for c in centers:
        fold = int(np.sum(np.abs(all_cmps - c) <= bin_size / 2.0))
        if fold >= min_fold:
            qualifying.append(c)

    print(f"Found {len(qualifying)} CMP bins with fold ≥ {min_fold} "
          f"(bin size = {bin_size} m, total bins scanned = {len(centers)})")
    return np.array(qualifying)


def build_2d_section(
    shots: list[dict],
    all_cmps: np.ndarray,
    all_offsets: np.ndarray,
    shot_idx: np.ndarray,
    trc_idx: np.ndarray,
    cmp_centers: np.ndarray,
    bin_size: float,
    t0_picks: np.ndarray,
    vrms_picks: np.ndarray,
    max_offset_frac: float = 0.65,
    abs_max_offset: float = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loop over every qualifying CMP bin, apply NMO + mute + stack, and
    assemble the results into a 2-D time-domain seismic section.

    Parameters
    ----------
    shots, all_cmps, all_offsets, shot_idx, trc_idx
        Outputs of :func:`load_shot_gathers` / :func:`compute_midpoints`.
    cmp_centers     : (M,)  qualifying bin centres from :func:`get_cmp_bin_centers`
    bin_size        : float  full bin width (m)
    t0_picks        : (n_picks,)  picked t0 values (ms)  – applied uniformly
    vrms_picks      : (n_picks,)  corresponding Vrms (m/s)
    max_offset_frac : float  far-offset mute as fraction of each gather's max
                             offset. Ignored if abs_max_offset is provided.
    abs_max_offset  : float  if provided, use this absolute mute threshold (m)
                             for every gather instead of the per-gather fraction.
                             Recommended when bin offsets vary across the survey.

    Returns
    -------
    section  : (n_samples, M)  2-D stacked seismic section
    t_axis   : (n_samples,)   two-way time axis (ms)
    x_axis   : (M,)           CMP x-positions (m)
    """
    n_samp = shots[0]["data"].shape[0]
    dt     = shots[0]["dt"]
    t_axis = np.arange(n_samp) * dt
    section = np.zeros((n_samp, len(cmp_centers)), dtype=np.float32)

    for i, cx in enumerate(cmp_centers):
        try:
            gather = build_cmp_gather(
                shots, all_cmps, all_offsets,
                shot_idx, trc_idx,
                cmp_center=cx, bin_size=bin_size,
            )
        except ValueError:
            # No traces found in this bin — leave column as zeros
            continue

        # Determine mute threshold: absolute takes priority over fractional
        if abs_max_offset is not None:
            max_off = abs_max_offset
        else:
            max_off = max_offset_frac * gather["offsets"].max()

        # Skip bin if mute threshold would eliminate all traces
        if not np.any(gather["offsets"] <= max_off):
            print(f"  Skipping CMP @ {cx:.0f} m — no traces survive mute threshold")
            continue

        corrected = nmo_correction(gather, t0_picks, vrms_picks)
        muted     = apply_far_offset_mute(corrected, gather["offsets"], max_off)
        stack     = stack_gather(muted, gather["offsets"], max_off)
        section[:, i] = stack

        if (i + 1) % 10 == 0 or i == len(cmp_centers) - 1:
            print(f"  Stacked {i + 1}/{len(cmp_centers)} CMP bins …")

    return section, t_axis, cmp_centers


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Interval Velocity via Dix Equation  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def dix_interval_velocities(
    t0_picks: np.ndarray,
    vrms_picks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute interval velocities from RMS velocities using the Dix equation::

        V_int(n)^2 = ( Vrms(n)^2 * t(n)  –  Vrms(n-1)^2 * t(n-1) )
                     / ( t(n) – t(n-1) )

    Parameters
    ----------
    t0_picks   : (n_picks,)  two-way zero-offset times (ms), sorted ascending
    vrms_picks : (n_picks,)  corresponding RMS velocities (m/s)

    Returns
    -------
    t0_int  : (n_picks - 1,)  mid-point times between picks (ms)
    v_int   : (n_picks - 1,)  interval velocities (m/s)
    """
    t  = t0_picks   * 1e-3   # convert ms → s for the Dix formula
    v  = vrms_picks

    v_int = np.sqrt(
        np.diff(v**2 * t) / np.diff(t)
    )
    t0_int = 0.5 * (t0_picks[:-1] + t0_picks[1:])   # midpoint times (ms)

    return t0_int, v_int


# ─────────────────────────────────────────────────────────────────────────────
# 10. Time-to-Depth Conversion  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def time_to_depth(
    section: np.ndarray,
    t_axis: np.ndarray,
    t0_picks: np.ndarray,
    vrms_picks: np.ndarray,
    dz: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2-D time-domain seismic section to depth using a 1-D
    velocity model derived from the Vrms picks via the Dix equation.

    The conversion maps two-way time *t* to depth *z* via::

        z(t) = ∫₀^{t/2} V_int(τ) dτ

    which is evaluated numerically on the original time samples.

    Parameters
    ----------
    section    : (n_t, n_x)  time-domain stacked section
    t_axis     : (n_t,)      two-way time (ms)
    t0_picks   : (n_picks,)  picked t0 values (ms)
    vrms_picks : (n_picks,)  corresponding Vrms (m/s)
    dz         : float  depth sample interval (m); default derived from data

    Returns
    -------
    depth_section : (n_z, n_x)  depth-domain section
    z_axis        : (n_z,)      depth axis (m)
    """
    n_t, n_x = section.shape
    dt_ms = t_axis[1] - t_axis[0]
    dt_s  = dt_ms * 1e-3

    # ── Step 1: build a continuous Vrms(t) profile ──────────────────────────
    vrms_interp = interp1d(
        t0_picks, vrms_picks,
        kind="linear", bounds_error=False,
        fill_value=(vrms_picks[0], vrms_picks[-1]),
    )
    vrms_cont = vrms_interp(t_axis)   # (n_t,)  Vrms at each time sample

    # ── Step 2: compute instantaneous interval velocity via Dix ─────────────
    # dV_int^2/dt ≈ d(Vrms^2 * t)/dt  (continuous form)
    # For numerical stability we work sample-by-sample.
    t_s = t_axis * 1e-3               # two-way time in seconds
    vrms2_t = vrms_cont**2 * t_s      # Vrms^2 * t

    # Derivative (central differences, forward/backward at edges)
    d_vrms2t = np.gradient(vrms2_t, t_s)
    # Instantaneous interval velocity (one-way)
    v_inst = np.sqrt(np.maximum(d_vrms2t, 100.0))   # clamp to avoid sqrt(<0)

    # ── Step 3: integrate one-way travel time to get depth ──────────────────
    # one-way dt = dt_s / 2
    depth_at_t = np.cumsum(v_inst * (dt_s / 2.0))   # (n_t,)  depth in metres
    depth_at_t = np.insert(depth_at_t[:-1], 0, 0.0) # shift so depth[0] = 0

    # ── Step 4: define regular depth axis ───────────────────────────────────
    z_max = depth_at_t[-1]
    if dz is None:
        dz = z_max / n_t              # same number of samples as time domain
    z_axis = np.arange(0, z_max + dz, dz)
    n_z    = len(z_axis)

    # ── Step 5: resample each CMP column from time to depth ─────────────────
    depth_section = np.zeros((n_z, n_x), dtype=np.float32)

    for ix in range(n_x):
        # Interpolate amplitude as a function of depth
        col_interp = interp1d(
            depth_at_t, section[:, ix],
            kind="linear", bounds_error=False, fill_value=0.0,
        )
        depth_section[:, ix] = col_interp(z_axis)

    return depth_section, z_axis


# ─────────────────────────────────────────────────────────────────────────────
# 11. Plotting Helpers
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
        ax.plot(picks_v, picks_t0, "r^-", ms=8, lw=1.5,
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


def plot_2d_section(
    section: np.ndarray,
    t_axis: np.ndarray,
    x_axis: np.ndarray,
    title: str = "2-D Stacked Seismic Section",
    clip_pct: float = 98,
    ax=None,
):
    """
    Display a 2-D time-domain stacked seismic section.

    Parameters
    ----------
    section  : (n_t, n_x)  stacked amplitudes
    t_axis   : (n_t,)      two-way time (ms)
    x_axis   : (n_x,)      CMP x-positions (m)
    title    : str
    clip_pct : float        amplitude clip percentile (default 98)
    ax       : matplotlib Axes or None
    """
    vmax = np.percentile(np.abs(section), clip_pct)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))

    ax.imshow(
        section, aspect="auto", cmap="seismic",
        vmin=-vmax, vmax=vmax,
        extent=[x_axis[0], x_axis[-1], t_axis[-1], t_axis[0]],
        origin="upper",
    )
    ax.set_xlabel("CMP x (m)")
    ax.set_ylabel("Two-way time (ms)")
    ax.set_title(title)
    return ax


def plot_depth_section(
    depth_section: np.ndarray,
    z_axis: np.ndarray,
    x_axis: np.ndarray,
    title: str = "2-D Depth-Converted Seismic Section",
    clip_pct: float = 98,
    ax=None,
):
    """
    Display a 2-D depth-converted seismic section.

    Parameters
    ----------
    depth_section : (n_z, n_x)  amplitudes in depth domain
    z_axis        : (n_z,)      depth axis (m)
    x_axis        : (n_x,)      CMP x-positions (m)
    title         : str
    clip_pct      : float
    ax            : matplotlib Axes or None
    """
    vmax = np.percentile(np.abs(depth_section), clip_pct)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))

    ax.imshow(
        depth_section, aspect="auto", cmap="seismic",
        vmin=-vmax, vmax=vmax,
        extent=[x_axis[0], x_axis[-1], z_axis[-1], z_axis[0]],
        origin="upper",
    )
    ax.set_xlabel("CMP x (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    return ax


def plot_fold_map(
    all_cmps: np.ndarray,
    bin_size: float,
    cmp_centers: np.ndarray,
    min_fold: int = 3,
    ax=None,
):
    """
    Bar chart of CMP fold across the survey, highlighting bins used
    for imaging (fold ≥ min_fold).

    Parameters
    ----------
    all_cmps    : (N,)  all CMP x-coordinates in the survey
    bin_size    : float  bin width (m)
    cmp_centers : (M,)  qualifying bin centres
    min_fold    : int   threshold used during selection
    ax          : matplotlib Axes or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    x_min = np.floor(all_cmps.min() / bin_size) * bin_size
    x_max = np.ceil(all_cmps.max()  / bin_size) * bin_size
    edges = np.arange(x_min, x_max + bin_size, bin_size)
    all_centers = 0.5 * (edges[:-1] + edges[1:])

    folds = np.array([
        int(np.sum(np.abs(all_cmps - c) <= bin_size / 2.0))
        for c in all_centers
    ])

    colors = ["steelblue" if c in cmp_centers else "lightgray"
              for c in all_centers]
    ax.bar(all_centers, folds, width=bin_size * 0.85,
           color=colors, edgecolor="none")
    ax.axhline(min_fold, color="red", ls="--", lw=1.2,
               label=f"Min fold = {min_fold}")
    ax.set_xlabel("CMP x (m)")
    ax.set_ylabel("Fold (traces per bin)")
    ax.set_title("CMP Fold Map  (blue = used for imaging)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return ax
