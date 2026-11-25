"""
Spectral power for the alcohol-project EEG: baseline (LB), IGT1, IGT2.

Input: one cleaned file per participant (subject_<ID>_clean_raw.fif, whole recording, from
eeg_preprocessing_alcohol.py) and the segment boundaries the pipeline writes to
qc/subject_<ID>_segments.json (markers after the documented repairs).

Windows: LB = last 300 s of its marked block; IGT and IGT2 = the whole segment. Stretches
annotated BAD_* are left out. Channels: the 30 EEG channels (average reference); TP9 (EOG) is
not used.

Output (one row per participant x segment x region): absolute and relative band power (delta,
theta, alpha, beta, gamma), individual alpha frequency (IAF) from LB, specparam periodic alpha
and aperiodic offset/exponent, and the number of clean 4-s pieces in each spectrum.

Run --check on one participant first, then --table/--qc for the whole sample.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks

# ----------------------------------------------------------------- CONFIG
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
TOTAL_RANGE = (1, 40)          # relative power = band power / power in this range
EPOCH_SEC = 4.0                # spectrum = average over 4-s pieces (0.25 Hz resolution), 50% overlap
EPOCH_OVERLAP_SEC = 2.0
IAF_SEARCH = (6, 15)           # Kopcanova et al. (2024)
IAF_HALF_WIDTH = 1.0           # Giorgi et al. (2025): IAF +/- 1 Hz
SPECPARAM_RANGE = (3, 40)      # Kopcanova et al. (2024)
SPECPARAM_SETTINGS = dict(peak_width_limits=(2, 15), max_n_peaks=4, peak_threshold=2.0, aperiodic_mode="fixed")

# Same numbers as in eeg_preprocessing_alcohol3.py
LB_MAX_SEC = 300.0
ANALYSIS_WINDOW_SEC = None     # None = whole segment (as in the article); 200.0 = central 200-s window.
SEGMENT_KEY = {"LB": "LB", "IGT1": "IGT", "IGT2": "IGT2"}   # name in our list -> name in the json

# Regions from the 30 EEG channels of this montage.
REGIONS = {
    "frontal":   ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central":   ["C3", "C4", "Cz", "FC1", "FC2", "CP1", "CP2"],
    "temporal":  ["T7", "T8", "FT9", "FT10", "FC5", "FC6", "CP5", "CP6"],
    "parietal":  ["Pz", "P3", "P4", "P7", "P8"],
    "occipital": ["O1", "Oz", "O2"],
}
POSTERIOR = ["parietal", "occipital"]   # used for the baseline IAF

# ----------------------------------------------------------------- SEGMENT WINDOW (same rule as the pipeline)
def analysis_window(seg_type, start, end):
    """LB: last 300 s of the marked block; IGT/IGT2 whole (central window only if ANALYSIS_WINDOW_SEC is set)."""
    if seg_type == "LB" and end - start > LB_MAX_SEC:
        start = end - LB_MAX_SEC
    dur = end - start
    if not ANALYSIS_WINDOW_SEC or dur <= ANALYSIS_WINDOW_SEC:
        return start, end
    mid = (start + end) / 2.0
    return mid - ANALYSIS_WINDOW_SEC / 2.0, mid + ANALYSIS_WINDOW_SEC / 2.0

def read_segment_times(qc_dir, subject, segment):
    path = Path(qc_dir) / f"subject_{subject}_segments.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    ivs = [iv for iv in data.get(SEGMENT_KEY[segment], []) if iv.get("end") is not None]
    if not ivs:
        raise ValueError(f"{subject} {segment}: no complete segment in {path.name}")
    if len(ivs) > 1:
        print(f"  note: {subject} {segment} has {len(ivs)} intervals, using the first")
    return float(ivs[0]["start"]), float(ivs[0]["end"])

# ----------------------------------------------------------------- SPECTRUM
def psd_from_epochs(epoch_data, sfreq):
    """epoch_data: (n_epochs, n_channels, n_samples) in volts -> freqs, psd (n_channels, n_freqs) in uV^2/Hz.
    One Hann-windowed spectrum per 4-s piece, averaged over pieces."""
    n = epoch_data.shape[-1]
    f, p = welch(epoch_data * 1e6, fs=sfreq, window="hann", nperseg=n, noverlap=0, axis=-1)
    return f, p.mean(axis=0)

def band_power(freqs, psd, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return trap(psd[..., m], freqs[m], axis=-1)

def estimate_iaf(freqs, psd):
    """Alpha peak after removing a 1/f line fitted outside 6-15 Hz; NaN if there is no clear peak."""
    lo, hi = IAF_SEARCH
    fit_m = (freqs >= 2) & (freqs <= 40) & ~((freqs >= 6) & (freqs <= 15))
    slope, icpt = np.polyfit(np.log10(freqs[fit_m]), np.log10(psd[fit_m]), 1)
    m = (freqs >= lo) & (freqs <= hi)
    resid = np.log10(psd[m]) - (icpt + slope * np.log10(freqs[m]))
    peaks, props = find_peaks(resid, prominence=0.1)  # prominence threshold
    if len(peaks) == 0:
        return np.nan
    return float(freqs[m][peaks[np.argmax(props["prominences"])]])

def specparam_alpha(freqs, psd):
    """NOT TESTED - needs the 'specparam' package. Returns NaNs if it is missing."""
    empty = dict(sp_cf=np.nan, sp_pw=np.nan, sp_bw=np.nan, sp_offset=np.nan, sp_exponent=np.nan, sp_r2=np.nan)
    try:
        from specparam import SpectralModel
    except ImportError:
        return empty
    try:
        fm = SpectralModel(**SPECPARAM_SETTINGS)
        fm.fit(freqs, psd, list(SPECPARAM_RANGE))
        ap = fm.get_params("aperiodic")
        peaks = np.atleast_2d(fm.get_params("peak"))
        out = dict(empty, sp_offset=float(ap[0]), sp_exponent=float(ap[-1]),
                   sp_r2=float(fm.get_metrics("gof", "rsquared")))
        if peaks.size:
            sel = peaks[(peaks[:, 0] >= IAF_SEARCH[0]) & (peaks[:, 0] <= IAF_SEARCH[1])]
            if len(sel):
                top = sel[np.argmax(sel[:, 1])]
                out.update(sp_cf=float(top[0]), sp_pw=float(top[1]), sp_bw=float(top[2]))
        return out
    except Exception as e:                      # one failed fit must not stop the run
        print("  specparam failed:", e)
        return empty

def region_psd(psd, ch_names):
    out = {}
    for name, chans in REGIONS.items():
        idx = [ch_names.index(c) for c in chans if c in ch_names]
        if idx:
            out[name] = psd[idx].mean(axis=0)
    return out

def segment_metrics(freqs, psd, ch_names, iaf=None):
    rows = []
    for region, rp in region_psd(psd, ch_names).items():
        total = band_power(freqs, rp, *TOTAL_RANGE)
        row = dict(region=region)
        for b, (lo, hi) in BANDS.items():
            bp = band_power(freqs, rp, lo, hi)
            row[f"abs_{b}"], row[f"rel_{b}"] = bp, bp / total
        if iaf is not None and not np.isnan(iaf):
            bp = band_power(freqs, rp, iaf - IAF_HALF_WIDTH, iaf + IAF_HALF_WIDTH)
            row["abs_iafband"], row["rel_iafband"] = bp, bp / total
        else:
            row["abs_iafband"] = row["rel_iafband"] = np.nan
        row.update(specparam_alpha(freqs, rp))
        rows.append(row)
    return pd.DataFrame(rows)

def baseline_iaf(freqs, psd, ch_names):
    rp = region_psd(psd, ch_names)
    post = [rp[r] for r in POSTERIOR if r in rp]
    return estimate_iaf(freqs, np.mean(post, axis=0)) if post else np.nan

# ----------------------------------------------------------------- READING THE FILES (UNTESTED - needs MNE)
def segment_epochs(raw, t0, t1):
    """Crop to the window, cut into 4-s pieces, drop pieces that touch BAD_* annotations.
    Returns (n_epochs, n_channels, n_samples), sfreq, channel names."""
    import mne
    seg = raw.copy().crop(tmin=t0, tmax=min(t1, raw.times[-1]))
    seg.pick("eeg")                                   # TP9 is typed EOG in your files -> left out
    ep = mne.make_fixed_length_epochs(seg, duration=EPOCH_SEC, overlap=EPOCH_OVERLAP_SEC,
                                      preload=True, reject_by_annotation=True, verbose="ERROR")
    return ep.get_data(), seg.info["sfreq"], seg.ch_names

def process_subject(rows, qc_dir):
    """rows: the table rows of ONE person (LB, IGT1, IGT2). Returns a DataFrame."""
    import mne
    subj = rows.iloc[0]["subject"]
    raw = mne.io.read_raw_fif(rows.iloc[0]["path"], preload=True, verbose="ERROR")
    spectra = {}
    for _, r in rows.iterrows():
        t0, t1 = analysis_window(SEGMENT_KEY[r.segment], *read_segment_times(qc_dir, subj, r.segment))
        data, sf, chs = segment_epochs(raw, t0, t1)
        if len(data) == 0:
            print(f"  {subj} {r.segment}: no clean 4-s piece left - skipped")
            continue
        f, p = psd_from_epochs(data, sf)
        spectra[r.segment] = (f, p, chs, len(data), t0, t1)
    iaf = np.nan
    if "LB" in spectra:
        f, p, chs, *_ = spectra["LB"]
        iaf = baseline_iaf(f, p, chs)
    out = []
    for seg, (f, p, chs, n_ep, t0, t1) in spectra.items():
        m = segment_metrics(f, p, chs, iaf)
        m.insert(0, "iaf_baseline", iaf)
        m.insert(0, "n_clean_pieces", n_ep)
        m.insert(0, "window_s", f"{t0:.1f}-{t1:.1f}")
        m.insert(0, "segment", seg)
        for c in ("subject", "id", "group", "sex", "borderline", "group_uncertain"):
            if c in rows.columns:
                m.insert(0, c, rows.iloc[0][c])
        out.append(m)
    return pd.concat(out) if out else pd.DataFrame()

def run(table_csv, qc_dir, out_csv, only=None):
    meta = pd.read_csv(table_csv, dtype=str).fillna("")
    parts = []
    for subj, g in meta.groupby("subject", sort=False):
        if only and subj != only:
            continue
        print("processing", subj)
        parts.append(process_subject(g, qc_dir))
    pd.concat(parts).to_csv(out_csv, index=False)
    print("wrote", out_csv)

def check(path, qc_dir, subject):
    """Prints what a person's file looks like, so we can confirm everything before the real run."""
    import mne
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    print("channels:", raw.ch_names)
    print("types   :", {t: [c for c, k in zip(raw.ch_names, raw.get_channel_types()) if k == t] for t in set(raw.get_channel_types())})
    print("sfreq   :", raw.info["sfreq"], " length (s):", round(raw.times[-1], 1))
    bad = [(round(a["onset"], 1), round(a["duration"], 1), a["description"]) for a in raw.annotations
           if a["description"].upper().startswith("BAD")]
    print("BAD annotations:", len(bad), " first 5:", bad[:5])
    for seg in ("LB", "IGT1", "IGT2"):
        s, e = read_segment_times(qc_dir, subject, seg)
        t0, t1 = analysis_window(SEGMENT_KEY[seg], s, e)
        data, sf, chs = segment_epochs(raw, t0, t1)
        print(f"{seg}: marked {s:.1f}-{e:.1f} s  -> window {t0:.1f}-{t1:.1f} s  -> {len(data)} clean 4-s pieces "
              f"(about {len(data) * 2 + 2} s of clean data)")
    print("regions found:", {r: [c for c in cs if c in chs] for r, cs in REGIONS.items()})

# ----------------------------------------------------------------- SELF-TEST (simulated, no MNE needed)
def _simulate_epochs(sfreq, n_epochs, ch_names, iaf, alpha_amp, rng):
    n = int(sfreq * EPOCH_SEC); t = np.arange(n) / sfreq
    X = np.zeros((n_epochs, len(ch_names), n))
    post = REGIONS["occipital"] + REGIONS["parietal"]
    for e in range(n_epochs):
        for i, ch in enumerate(ch_names):
            fr = np.fft.rfftfreq(n, 1 / sfreq); fr[0] = 1
            spec = (rng.standard_normal(len(fr)) + 1j * rng.standard_normal(len(fr))) / fr
            noise = np.fft.irfft(spec, n); noise = noise / noise.std() * 8e-6
            amp = alpha_amp * (1.0 if ch in post else 0.3) * 1e-6
            X[e, i] = noise + amp * np.sin(2 * np.pi * iaf * t + rng.uniform(0, 6.28))
    return X

def selftest():
    # 1) window rule: LB 0-305 -> last 300 s = 5-305 -> central 200 s = 55-255
    global ANALYSIS_WINDOW_SEC
    assert analysis_window("LB", 0, 605) == (305.0, 605.0)     # LB: last 300 s
    assert analysis_window("IGT", 100, 400) == (100, 400)      # IGT whole
    ANALYSIS_WINDOW_SEC = 200.0                                 # (optional v3 window, only tested here)
    assert analysis_window("LB", 0, 305) == (55.0, 255.0)
    assert analysis_window("IGT", 100, 400) == (150.0, 350.0)
    ANALYSIS_WINDOW_SEC = None
    # 2) spectrum, band power, IAF on simulated data
    rng = np.random.default_rng(1)
    chs = sorted({c for v in REGIONS.values() for c in v})
    sf = 500.0
    X = _simulate_epochs(sf, 40, chs, 10.5, 6.0, rng)
    f, p = psd_from_epochs(X, sf)
    iaf = baseline_iaf(f, p, chs)
    assert abs(iaf - 10.5) <= 0.5, iaf
    m = segment_metrics(f, p, chs, iaf)
    occ = m[m.region == "occipital"].iloc[0]; fro = m[m.region == "frontal"].iloc[0]
    assert occ.rel_alpha > fro.rel_alpha, (occ.rel_alpha, fro.rel_alpha)
    assert 0 < occ.rel_alpha < 1
    print("selftest OK  (IAF found: %.2f Hz, occipital rel alpha %.2f, frontal %.2f)" % (iaf, occ.rel_alpha, fro.rel_alpha))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", metavar="FIF", help="inspect ONE person's file (needs --qc and --subject)")
    ap.add_argument("--subject")
    ap.add_argument("--table", help="file list csv (columns subject, group, sex, segment, path)")
    ap.add_argument("--qc", help="folder output_manual/qc with subject_<ID>_segments.json")
    ap.add_argument("--out", default="spectral_features.csv")
    ap.add_argument("--only", help="run just one subject")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.check:
        check(a.check, a.qc, a.subject)
    elif a.table:
        run(a.table, a.qc, a.out, a.only)
    else:
        ap.print_help()
