# -*- coding: utf-8 -*-
"""
EEG Individual Pre-processing Pipeline — Alcohol project (BrainVision, 31 ch)
=============================================================================
Study   : Acute psychosocial stress and decision making (alcohol project).
          Stress arm = TSST-VR (files ending in "E"); control arm = nature
          video (files ending in "C").
Segments: LB    baseline — last 5 min of the habituation
          IGT   Iowa Gambling Task, run 1 (before the manipulation)
          IGT2  Iowa Gambling Task, run 2 (after the manipulation; marked
                "IGT_ALC" in the files — no alcohol is given in the session)
          TSST, the heartbeat task (INTERO) and the recovery are NOT
          preprocessed — see the note at SEG_TYPE_ORDER in Cell 1.
Output  : cleaned EEG .fif + per-subject QC folder + preprocessing_summary.csv
          in output_manual/ (the earlier automatic triage lives in output/)

WORKFLOW (one subject at a time — set SUJETO in Cell 2):
─────────────────────────────────────────────────────────
  Cell 1   Imports · config · all function definitions  (run once, never edit)
  Cell 2   Load raw → markers → documented repairs → segments   ← BREAKPOINT
  Cell 3a  Inspect RAW: LB    → mark BAD channels / BAD sections
  Cell 3b  Inspect RAW: IGT   → mark BAD channels / BAD sections
  Cell 3c  Inspect RAW: IGT2  → mark BAD channels / BAD sections
  Cell 3d  Segment exclusions (per subject, with the reason)
  Cell 3e  Summary of manual edits (also saved to manual_marks.json)
  Cell 4   Preprocessing Steps 1-8  (filter → bad-ch → artifacts → ICA → interp → ref)
  Optional ICA source viewer — review components, edit ica_exclude.json
  Cell 5a-c Inspect CLEANED signal per segment
  Cell 6   Preprocessing summary + save cleaned .fif + progress log

PIPELINE ORDER (Kim et al. 2023 / Klug & Gramann 2021 / Makoto's pipeline):
  Step 1  Channel selection + channel types (TP9 = sub-ocular EOG proxy)
  Step 2  Montage (standard_1020)
  Step 3  Two filtered copies: ICA-fit (1-40 Hz) and final (0.1-40 Hz) + notch
  Step 4  Bad channels per segment: automatic (LOF + flatline) UNION manual
  Step 5  Gross artifact annotation (PTP + gamma-band muscle) on ICA-fit copy
  Step 6  ICA fit on concatenated QC-passing segments → apply to final dataset
  Step 7  Interpolate bad channels per segment
  Step 8  Average re-reference per segment

WHAT DIFFERS FROM THE BREATH-HOLD PIPELINE THIS WAS BUILT FROM
  Corrections to the manual-cleaning path (each checked in verification):
  * Manual bad-channel marks take effect. Previously a channel clicked in the
    viewer was reported and then discarded: Step 4 overwrote raw.info["bads"]
    with the automatic result. They are now kept per segment and unioned with
    the automatic detection of that segment.
  * A channel bad in one segment is no longer re-interpolated in the others.
    Step 8 used to inherit the union of every segment's bad channels, so the
    average skipped them and they were interpolated again everywhere.
  * BAD time is counted once where marks overlap. A manual mark and an
    automatic one on the same stretch used to be summed, inflating bad_ratio.
  * The interpolation ratio uses all 30 EEG channels as denominator (it used
    the good channels only, so 3 bad channels read as 3/27 = 11% > 10%).
  * An edited ica_exclude.json is honoured only for the decomposition it was
    written for. If the input changes, ICA is refitted and the components are
    renumbered; the old list would have removed the wrong ones.
  * The saved file's metadata is correct: info["bads"] is cleared after
    interpolation and the average-reference flag is set.
  * Eye activity no longer makes frontal channels "bad" (version 2026-09-16).
    When the blink mask leaves too few clean windows — the rule during the IGT —
    the variance check is skipped instead of run on unmasked data, which had
    flagged F3/F4/F7/F8/Fz and excluded whole IGT/IGT2 segments. TP9 is now
    filtered with the EEG before it feeds the blink mask. The automatic
    peak-to-peak artifact check ignores Fp1/Fp2, where ordinary blinks exceeded
    250 uV and turned up to 73 % of a segment into BAD_artifact.
  Channel handling specific to this study:
  * Cz, Fp1 and Fp2 stay real data. Every alcohol recording used TP10 as the
    online reference, which is not one of the 31 data channels, so no channel
    is reference-linked. Fp1/Fp2 remain excluded from the bad-channel
    DETECTOR (blinks are biological signal and they are the ICA ocular
    proxies) but are no longer interpolated away at the average reference.
  Set LEGACY_BH_CHANNEL_EXCLUSIONS = True to restore the breath-hold channel
  handling (Cz excluded from detection; Cz/Fp1/Fp2 interpolated at Step 8;
  union bad channels carried into Step 8; TP9 unfiltered; variance check
  falling back to unmasked data; Fp1/Fp2 in the peak-to-peak artifact check).
  Disk:
  * Only the step-5 intermediate .fif is written (the ICA viewer needs it) and
    Cell 6 deletes it. The old default wrote eleven copies (~2.5 GB/subject).

ICA MANUAL OVERRIDE:
  After Cell 4, look at output_manual/qc/subject_X/ica_topomap_*.png and, if
  useful, run the optional ICA viewer cell. To change which components are
  removed, edit "exclude" in output_manual/qc/subject_X/ica_exclude.json and
  re-run Cell 4.
"""

#%%
# =============================================================================
# CELL 1 — IMPORTS · CONFIGURATION · FUNCTION DEFINITIONS
# =============================================================================
# Nothing is *executed* here other than defining constants, functions, and
# performing the minimal side-effect-free imports.
# =============================================================================

# ---------------------------------------------------------------------------
# 1.1  IMPORTS
# ---------------------------------------------------------------------------
import os
import re
import sys
import json
import copy
import logging
import warnings
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import mne
from mne.preprocessing import ICA

# ---------------------------------------------------------------------------
# 1.2  CONFIGURATION CONSTANTS
# ---------------------------------------------------------------------------

# Printed in Cell 2 and written to progress_log.csv, so every cleaned subject
# records the version it was processed with.
PIPELINE_VERSION: str = "2026-09-16"

# --- Paths ----------------------------------------------------------------
def _resolve_base_dir() -> Path:
    """Project root: the folder that holds scripts/ and raw_data/.

    Order: environment variable EEG_ALCOHOL_BASE_DIR -> the script's own
    location -> the working directory or its parent. Interactive windows may
    run cells without __file__ and with the working directory set to either
    the project folder or its scripts/ folder; both are handled.
    """
    env_override = os.environ.get("EEG_ALCOHOL_BASE_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    try:
        return Path(__file__).resolve().parent.parent  # type: ignore[name-defined]
    except NameError:
        pass
    here = Path.cwd()
    for cand in (here, here.parent):
        if (cand / "raw_data").is_dir() and (cand / "scripts").is_dir():
            return cand
    return here


BASE_DIR   = _resolve_base_dir()
INPUT_DIR  = BASE_DIR / "raw_data"
OUTPUT_DIR = BASE_DIR / "output_manual"   # manual pass; the automatic triage is in output/
FIF_DIR    = OUTPUT_DIR / "fif_steps"
QC_DIR     = OUTPUT_DIR / "qc"
MPLCONFIGDIR = BASE_DIR / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


def _project_python() -> str:
    """Interpreter for GUI subprocesses (prefer project .venv on any OS)."""
    if sys.platform == "win32":
        venv_py = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    else:
        venv_py = BASE_DIR / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def _configure_matplotlib_for_popup() -> None:
    """Switch away from inline/Agg so overview plots open in real windows."""
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if "inline" not in backend and backend not in ("agg",):
        return
    for name in ("macosx", "Qt5Agg", "TkAgg"):
        try:
            matplotlib.use(name, force=True)
            plt.ion()
            return
        except Exception:
            continue

# --- Channel configuration -------------------------------------------------
EXPECTED_CHANNELS: List[str] = [
    "Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "TP9",
    "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8",
    "CP6", "CP2", "Cz", "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8",
    "Fp2",
]
EOG_CHANNEL: str = "TP9"            # relocated below left eye, used as EOG proxy
ACQUISITION_REFERENCE: str = "TP10"  # every alcohol .vhdr: Reference Channel Name = TP10

# --- Filtering parameters --------------------------------------------------
NOTCH_FREQ: float         = 50.0    # Hz  (line noise)
NOTCH_HARMONICS: bool     = True    # also notch at 100 Hz
ICA_FIT_HIGH_PASS: float  = 1.0     # Hz  (temporary dataset used only to fit ICA)
FINAL_HIGH_PASS: float    = 0.1     # Hz  (final cleaned dataset keeps low frequencies)
FILTER_LOW_PASS: float    = 40.0    # Hz  (shared low-pass for ICA-fit and final dataset)

# --- ICA parameters --------------------------------------------------------
ICA_METHOD: str                      = "fastica"
ICA_N_COMPONENTS: Optional[float]    = 0.999999  # variance-based (avoids unstable mixing matrix)
ICA_RANDOM_STATE: int                = 42
ICA_MAX_ITER: int                    = 1000
ICA_EOG_CHANNEL_CANDIDATES: Tuple[str, ...] = (EOG_CHANNEL, "Fp1", "Fp2")
ICA_EOG_FALLBACK_MIN_ABS_SCORE: float = 0.60
ICA_MAX_EOG_COMPONENTS: int = 2  # cap: keep only top-N by max|score|; prevents removing brain signal via low-variance components (e.g. indices 20-23)
# High-index guard: components with index > this threshold are late/low-variance FastICA components.
# They require a minimum TP9 (sub-ocular electrode) score to confirm they are genuine eye artifacts.
# Without this, a random low-variance component can survive the 2-component cap via weak Fp1/Fp2 correlation.
# Set to None to disable. Motivated by S10 (component 23/27, TP9 score=0.13).
ICA_HIGH_INDEX_THRESHOLD: Optional[int] = 20        # components above this index need TP9 confirmation
ICA_HIGH_INDEX_MIN_TP9_SCORE: float     = 0.30      # minimum |TP9 score| to keep a high-index component
# Manual ICA override: after Cell 4, inspect the topomaps and the Optional ICA
# viewer, then edit output/qc/subject_X/ica_exclude.json if needed. Re-running
# Cell 4 will respect the edited exclusion list because this flag is True.
# Set to False only if you want to force automatic detection and ignore any JSON.
ICA_RESPECT_MANUAL_OVERRIDE_JSON: bool = True

# --- Bad-channel detection --------------------------------------------------
# Primary detector: Local Outlier Factor (LOF), Kumaravel et al. 2022, adopted
# in EEGLAB and exposed by MNE as ``mne.preprocessing.find_bad_channels_lof``.
# It catches channels whose neighborhood-density profile is anomalous compared
# to the rest of the montage, and outperforms variance / correlation / RANSAC
# in the LOF reference paper. We use LOF + a strict flatline check as defaults,
# with the legacy persistent-variance detector available as a QC cross-check.
BAD_CHAN_USE_LOF: bool = True
BAD_CHAN_LOF_N_NEIGHBORS: int = 20      # MNE default; sufficient for ~30 EEG channels
BAD_CHAN_LOF_THRESHOLD: float = 1.6   # raised from 1.5: avoids flagging channels bad for only ~2/112 windows (transient bursts)
BAD_CHAN_LOF_METRIC: str = "euclidean"  # passed straight to scikit-learn
# Strict flatline detector (channel with practically zero signal).
# Always applied alongside LOF; LOF density-based scoring can miss dead channels.
HARD_FLATLINE_EPS_UV: float = 1e-6
# Low-activity threshold is stored for QC only (not auto-rejected by itself).
FLATLINE_THRESH_UV: float  = 0.5
# Persistent-variance detector: previous primary criterion, now opt-in.
# Even when off, the diagnostics are still computed and stored for cross-check.
BAD_CHAN_USE_PERSISTENT_VARIANCE: bool = True
BAD_CHAN_ZSCORE_THRESH: float = 4.0    # robust z-score on windowed variance; higher = less aggressive
BAD_CHAN_WINDOW_SEC: float = 2.0       # variance is evaluated in short windows
BAD_CHAN_PERSIST_RATIO: float = 0.30   # channel must be outlier in >=30% of windows
BAD_CHAN_MAX_INTERP_RATIO: float = 0.10  # >10% bad channels → warn, likely reject segment
BAD_CHAN_MAX_PER_UNIT: int = 3  # if a unit has >3 bad channels, exclude it from preprocessing
# Hard QC rules used before statistical analysis (segment-level).
ANALYSIS_MAX_BAD_TIME_RATIO: float = 0.35
ANALYSIS_MAX_INTERP_RATIO: float = 0.10
# Exclusions for bad-channel DETECTION (Step 4).
# Fp1/Fp2: their high variance is dominated by blinks (biological signal), not
# electrode failure — LOF flags them as outliers for the wrong reason. They are
# also the ICA ocular proxies; interpolating them before ICA would remove the
# ocular signal ICA needs to separate eye components (Luck 2014; Makoto; Kim 2023).
# ACQUISITION_REFERENCE (TP10) is listed for completeness only: it is not one of
# the 31 data channels, so the entry is inert.
# Cz is NOT excluded: in the breath-hold study Cz was flat in the subjects
# recorded with Cz as online reference, but every alcohol recording used TP10,
# so Cz is ordinary data and is checked like any other channel.
LEGACY_BH_CHANNEL_EXCLUSIONS: bool = False
BAD_CHAN_EXCLUDE_CHANNELS: Tuple[str, ...] = (
    (ACQUISITION_REFERENCE, "Cz", "Fp1", "Fp2") if LEGACY_BH_CHANNEL_EXCLUSIONS
    else (ACQUISITION_REFERENCE, "Fp1", "Fp2")
)
# Channels left out of the AVERAGE REFERENCE (Step 8) and re-created by
# interpolation afterwards. Only a channel that is flat because it WAS the
# online reference belongs here. None of the 31 alcohol data channels is, so
# the list is empty and Cz/Fp1/Fp2 enter the average as real data. (The
# breath-hold pipeline reused the detector list here, which interpolated real
# Cz/Fp1/Fp2 signal away; LEGACY_BH_CHANNEL_EXCLUSIONS = True restores that.)
AVG_REF_EXCLUDE_CHANNELS: Tuple[str, ...] = (
    ("Cz", "Fp1", "Fp2") if LEGACY_BH_CHANNEL_EXCLUSIONS else ()
)
# Ocular masking for the variance detector (VEOG bipolar proxy: TP9 - Fp1).
# Two methods supported:
#   "step" — Luck-style step function (Luck 2014 / Applied ERP analysis ch.8):
#           slides a sub-window across the variance window and flags a blink
#           when |mean(second-half) - mean(first-half)| exceeds STEP_THRESH_UV.
#           More specific to blink-shaped transients than within-window PTP.
#   "ptp"  — within-window peak-to-peak threshold (legacy fallback).
BAD_CHAN_EXCLUDE_BLINK_WINDOWS: bool = True
BAD_CHAN_VEOG_MASK_METHOD: str = "step"   # "step" (Luck) | "ptp" (legacy)
BAD_CHAN_VEOG_CH_POS: str = EOG_CHANNEL  # TP9 in this montage
BAD_CHAN_VEOG_CH_NEG: str = "Fp1"
BAD_CHAN_VEOG_BLINK_THRESH_UV: float = 100.0  # used when VEOG_MASK_METHOD == "ptp"
BAD_CHAN_VEOG_STEP_WINDOW_SEC: float = 0.4    # Luck recommendation
BAD_CHAN_VEOG_STEP_THRESH_UV: float = 50.0    # Luck recommendation
# Stride of the sliding step kernel as a fraction of its length. Needs to be
# small enough that at least one kernel position aligns its half-boundary with
# the blink's rising edge — Luck recommends ≤ rise-duration / 2 ≈ 25 ms, i.e.
# ≈5–10% of the 400 ms kernel. Cumulative-sum implementation keeps it cheap.
BAD_CHAN_VEOG_STEP_STRIDE_FRAC: float = 0.05
BAD_CHAN_MIN_CLEAN_WINDOWS: int = 20
# If the VEOG mask leaves fewer than BAD_CHAN_MIN_CLEAN_WINDOWS blink-free
# windows, the variance check is SKIPPED for that segment; flatline and LOF still
# run, and channels marked by hand still count. The breath-hold version fell back
# to unmasked variance instead. During the IGT nearly every 2-s window holds a
# blink or an eye movement (median 3-4 % survive the mask), so the fallback
# measured eye activity: F3/F4/F7/F8/Fz became outliers and 35-49 % of IGT/IGT2
# segments passed BAD_CHAN_MAX_PER_UNIT in the August triage with intact
# electrodes (7E IGT2: excess power only below 4 Hz, and no channel flagged once
# ICA had removed the eye components). Adding those channels to
# BAD_CHAN_EXCLUDE_CHANNELS was tested and only moved the flags to FC2/FC5/FC6.
BAD_CHAN_FALLBACK_TO_UNMASKED_VARIANCE: bool = LEGACY_BH_CHANNEL_EXCLUSIONS

# --- Artifact annotation ---------------------------------------------------
# Peak-to-peak threshold per 1-s window (uV). Increase if too many segments
# are flagged as artifacts (200-250 is moderate; 300-400 is more lenient).
# Set to 0 or very high (e.g., 9999) to disable automatic artifact detection.
ARTIFACT_PTP_THRESH_UV: float = 250.0   # less aggressive; avoids flagging too many blinks
AUTO_ARTIFACT_DETECTION: bool = True    # Set to False to skip automatic detection
# Channels left out of the peak-to-peak check (the muscle check still uses all).
# Ordinary blinks pass 250 uV on Fp1/Fp2. With them included, 24-73 % of LB/IGT/
# IGT2 became BAD_artifact in the first five subjects cleaned (7E IGT2: 205 s, every
# second from Fp1/Fp2, none above 250 uV once ICA had run), 7 of 15 segments
# failed the 35 % rule on automatic marks alone, and the blinks were kept out of
# the ICA fit that has to learn them. Without them: 0-19 %, mostly T7/T8, P8 and
# the largest F7/F8 eye movements.
ARTIFACT_PTP_EXCLUDE_CHANNELS: Tuple[str, ...] = (
    () if LEGACY_BH_CHANNEL_EXCLUSIONS else ("Fp1", "Fp2")
)
# Channels clicked as bad in a RAW viewer are kept per segment (MANUAL_BAD_CHANNELS)
# and added to that segment's automatic detection in Step 4. True = apply every
# manual channel mark to all segments instead.
APPLY_MANUAL_BAD_CHANNELS_GLOBALLY: bool = False
# Default interactive viewer scales (initial setting, still user-adjustable in the UI)
RAW_INTERACTIVE_SCALE_UV: float = 100.0
CLEAN_INTERACTIVE_SCALE_UV: float = 40.0
# Muscle artifact detection: gamma-band (30-40 Hz) power threshold per 1-s window.
# Windows where ANY channel exceeds this are flagged BAD_muscle.
# Units: uV^2/Hz. ~5.0 is a reasonable starting threshold; lower = more aggressive.
MUSCLE_GAMMA_THRESH_UV2: float = 7.5   # uV^2/Hz; slightly less aggressive than before

# --- Inspection / artifact helper parameters -------------------------------
MUSCLE_GAMMA_BAND: Tuple[float, float] = (30.0, 40.0)
INSPECT_PSD_N_FFT: int = 2048
INSPECT_PSD_N_OVERLAP: int = 1024
INSPECT_PSD_FMIN: float = 1.0
INSPECT_PSD_FMAX: float = 40.0

# --- Segments --------------------------------------------------------------
# Every marker label is parsed (so Cell 2 can show them all), but only these
# three segments are preprocessed and enter the ICA fit:
#   LB    baseline — the standardised 5-minute window used for the physiology
#   IGT   first Iowa Gambling Task run, before the manipulation
#   IGT2  second run, after it
# Deliberately NOT preprocessed:
#   TSST    the markers bracket the manipulation in BOTH arms (TSST-VR for the
#           stress arm, the three 5-min nature-video blocks for controls).
#           Gamma-band EMG covers 73-96% of windows in the stress arm and 1-6%
#           in controls, so artifact level is confounded with condition.
#   INTERO  six short heartbeat-tracking trials (25-45 s) separated by verbal
#           reports: not a continuous recording.
#   Recov   not needed for the planned analysis (LB, IGT, IGT2).
SEG_TYPE_ORDER: Tuple[str, ...] = ("LB", "IGT", "IGT2")
PREPROCESS_SEG_TYPES: Tuple[str, ...] = SEG_TYPE_ORDER
_EXPECTED_N_BY_TYPE: Dict[str, int] = {"LB": 1, "IGT": 1, "IGT2": 1}

# LB keeps the LAST 300 s of its marked block — the "last 5 minutes of
# habituation" convention used for this project's peripheral physiology. Most
# subjects' LB markers already bracket ~300 s; four (10E, 12C, 13E, 14E)
# bracket the whole 10-min habituation, where the last 300 s is the baseline.
# IGT and IGT2 are self-paced and kept whole.
LB_MAX_SEC: float = 300.0

# Plausible durations, used ONLY to raise warnings in Cell 2. Set from the
# observed distribution over the analysed sample, not the nominal protocol:
#   LB    median 300 s, 5th percentile 290 s (a few seconds short is normal)
#   IGT   median 281 s; up to 621 s (3E — confirmed by the BIOPAC record)
#   IGT2  median 262 s; the shortest sit on the 4-minute floor (235-240 s)
SEG_EXPECTED_SEC: Dict[str, Tuple[float, float]] = {
    "LB": (240.0, 305.0),
    "IGT": (210.0, 660.0),
    "IGT2": (210.0, 660.0),
}

# Marker counts per label in a complete recording (soft; for warnings only).
EXPECTED_MARKER_COUNTS: Dict[str, int] = {
    "LB": 2, "INTERO": 2, "IGT": 2, "TSST": 6, "IGT2": 2, "FIN": 1,
}

# --- Per-subject state (Cell 2 resets all three) ---------------------------
# A segment enters preprocessing only once its Cell 3 viewer has been run
# (SEGMENTS_INSPECTED) and it has not been excluded in Cell 3d.
SEGMENTS_INSPECTED: Dict[str, bool] = {s: True for s in PREPROCESS_SEG_TYPES}
SEGMENT_EXCLUSIONS: Dict[str, str] = {}
# Channels marked bad by hand in each segment's RAW viewer.
MANUAL_BAD_CHANNELS: Dict[str, List[str]] = {}

# --- File saving flags ------------------------------------------------------
# Set to False to reduce disk usage (only final cleaned data will be saved)
SAVE_INTERMEDIATE_FIF: bool = True    # needed for the optional ICA sources viewer
# Only these intermediate steps are written: the ICA viewer reads step5 alone,
# and saving every step costs ~2.5 GB per subject. None = save all steps.
SAVE_INTERMEDIATE_STEPS: Optional[Tuple[str, ...]] = ("step5_annotated",)
# Delete this subject's intermediate .fif once Cell 6 has saved the cleaned
# data (they are only needed while reviewing that subject's ICA).
DELETE_INTERMEDIATE_AFTER_SAVE: bool = True
OVERWRITE_FIF: bool         = True
# Controls QC/diagnostic files: ICA solution .fif, ICA topomap PNGs,
# segment overview PNGs, and preprocessing_summary.csv.
# The clean data .fif is always saved regardless.
# ica_suggest.json is also always saved (needed for manual ICA override).
SAVE_QC_FILES: bool         = True    # Set False to keep only cleaned data

# --- Label normalisation map ------------------------------------------------
# All protocol labels are mapped so Cell 2 can show and repair every marker.
# "IGT_ALC" is the second IGT run: no alcohol is administered in the session.
_LABEL_NORMALIZE: Dict[str, str] = {
    "lb": "LB", "baseline": "LB",
    "intero": "INTERO", "hbtt": "INTERO",
    "igt": "IGT", "igt1": "IGT",
    "tsst": "TSST",
    "igt_alc": "IGT2", "igtalc": "IGT2", "igt2": "IGT2",
    "fin": "FIN",
}


# ---------------------------------------------------------------------------
# 1.3  LOGGER (object creation only - handlers added at runtime in Cell 2)
# ---------------------------------------------------------------------------
logger = logging.getLogger("eeg_bh_pipeline")
logger.propagate = False  # prevent duplicate messages from root logger


def setup_logging(log_dir: Path = QC_DIR, console_level: int = logging.WARNING) -> None:
    """
    Configure file (DEBUG) + optional console logging.

    By default console only shows WARNING+, so cell output stays clean.
    Detailed logs go to the file for later review.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "preprocess_log.txt"

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_fmt = logging.Formatter("%(levelname)s | %(message)s")

    # Console: only warnings by default (keep output clean)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    # File: full debug
    fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.debug(f"Logging initialised  ->  {log_file}")


# ---------------------------------------------------------------------------
# 1.4  DIRECTORY HELPERS
# ---------------------------------------------------------------------------

def ensure_output_dirs() -> None:
    for d in (OUTPUT_DIR, FIF_DIR, QC_DIR):
        d.mkdir(parents=True, exist_ok=True)


def subject_fif_dir(subj_id: Union[int, str]) -> Path:
    d = FIF_DIR / f"subject_{subj_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def subject_qc_dir(subj_id: Union[int, str]) -> Path:
    d = QC_DIR / f"subject_{subj_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 1.5  SUBJECT DISCOVERY
# ---------------------------------------------------------------------------

def discover_subjects(input_dir: Path = INPUT_DIR) -> List[str]:
    """Subject IDs ("15C", "24E", ...) found as *.vhdr, in numeric order."""
    ids = [f.stem for f in Path(input_dir).glob("*.vhdr")
           if not f.stem.endswith("_fixed")]

    def _key(s: str) -> Tuple[int, str]:
        digits = "".join(c for c in s if c.isdigit())
        return (int(digits) if digits else 10 ** 9, s)

    return sorted(ids, key=_key)


def subject_vhdr(subj_id: Union[int, str], input_dir: Path = INPUT_DIR) -> Path:
    """Path to a subject's .vhdr. The subject ID is the file name without its
    extension; the last letter is the condition (C = control, E = stress)."""
    return Path(input_dir) / f"{subj_id}.vhdr"


# ---------------------------------------------------------------------------
# 1.6  VHDR REFERENCE FIXER
# ---------------------------------------------------------------------------

def fix_vhdr_references(vhdr_path: Path, subj_id: Union[int, str, None] = None) -> Path:
    """Repair a .vhdr whose DataFile= / MarkerFile= point at the wrong names.

    Six alcohol recordings were renamed after acquisition (6C, 18C, 32C, 57E,
    58C, 62E) without updating the header, so a BrainVision reader cannot find
    the data. The corrected copy ``*_fixed.vhdr`` is written beside the
    original and returned. In all six cases the FILE NAME is the correct ID
    (checked against the condition column of the master database).
    """
    vhdr_path = Path(vhdr_path)
    vhdr_dir = vhdr_path.parent
    stem = vhdr_path.stem
    text = vhdr_path.read_text(encoding="utf-8", errors="replace")

    needs_fix = False
    new_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        for key, ext in (("datafile=", ".eeg"), ("markerfile=", ".vmrk")):
            if low.startswith(key):
                ref = stripped.split("=", 1)[1].strip()
                if not (vhdr_dir / ref).exists():
                    target = f"{stem}{ext}"
                    if (vhdr_dir / target).exists():
                        new_lines.append(f"{stripped.split('=', 1)[0]}={target}")
                        needs_fix = True
                        logger.warning(
                            f"S{subj_id or stem} vhdr: {ref} not found -> {target} "
                            "(file renamed after recording; repaired automatically)"
                        )
                        break
                new_lines.append(line)
                break
        else:
            new_lines.append(line)

    if not needs_fix:
        return vhdr_path
    fixed = vhdr_dir / f"{stem}_fixed.vhdr"
    fixed.write_text("\n".join(new_lines), encoding="utf-8")
    logger.info(f"S{subj_id or stem}: corrected vhdr -> {fixed.name}")
    return fixed


# ---------------------------------------------------------------------------
# 1.7  PARSE RECORDING REFERENCE FROM .VHDR
# ---------------------------------------------------------------------------

def parse_vhdr_reference(vhdr_path: Path) -> Optional[str]:
    """Extract 'Reference Channel Name = ...' from a BrainVision header.

    Skips comment lines (starting with ';') and lines containing angle
    brackets '<' / '>' which belong to the channel-info description header.
    """
    try:
        for line in vhdr_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            # Skip comments and description-header lines
            if stripped.startswith(";") or "<" in stripped or ">" in stripped:
                continue
            low = stripped.lower()
            if "reference channel name" in low and "=" in low:
                ref = stripped.split("=", 1)[1].strip()
                if ref:
                    return ref
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 1.8  ANNOTATION / MARKER PARSING
# ---------------------------------------------------------------------------

def list_raw_annotations(raw: mne.io.BaseRaw, subj_id: Union[int, str]) -> str:
    """Print and return every marker in chronological order, in MINUTES, with
    the gap to the next marker.

    The gap column is what makes a mislabelled press obvious: a 4-5 min gap
    after an orphan marker is a segment that was closed under the wrong name.
    """
    anns = list(raw.annotations)
    lines: List[str] = [
        "",
        "=" * 78,
        f"  MARKERS  (SUBJECT {subj_id})  --  chronological",
        "=" * 78,
        f"  {'#':<4}{'minute':>9}{'sec':>10}{'gap->next':>12}   label",
        "-" * 78,
    ]
    for i, ann in enumerate(anns):
        onset = float(ann["onset"])
        desc = str(ann["description"])
        if "/" in desc:
            desc = desc.split("/", 1)[-1].strip()
        gap = (float(anns[i + 1]["onset"]) - onset) if i + 1 < len(anns) else None
        gap_s = f"{gap / 60:8.1f}m" if gap is not None else "        -"
        lines.append(
            f"  {i + 1:<4}{onset / 60:>8.1f}m{onset:>10.1f}{gap_s:>12}   "
            f"{desc or '(new segment)'}"
        )
    lines.append("=" * 78)
    report = "\n".join(lines)
    print(report)
    return report


def parse_annotations(raw: mne.io.BaseRaw) -> Dict[str, List[float]]:
    """
    Return ``{canonical_label: [onset_sec, ...]}`` from raw.annotations.

    Handles descriptions such as ``"Comment/LB"``, ``"Comment/Recovery"``,
    or plain ``"LB"``.  Unknown labels are silently skipped.
    """
    label_onsets: Dict[str, List[float]] = defaultdict(list)
    for ann in raw.annotations:
        desc = ann["description"].strip()
        # Strip BrainVision Type/ prefix (e.g. "Comment/CPT" -> "CPT")
        if "/" in desc:
            desc = desc.split("/", 1)[-1].strip()
        # Normalize key: collapse whitespace, lower case (handles "CPT ", "  cpt", etc.)
        key = " ".join(desc.lower().split()) if desc else ""
        canonical = _LABEL_NORMALIZE.get(key)
        # Fallback: try without spaces (e.g. "C P T" from some export)
        if canonical is None and key:
            canonical = _LABEL_NORMALIZE.get(key.replace(" ", ""))
        if canonical is not None:
            label_onsets[canonical].append(float(ann["onset"]))

    for lab in label_onsets:
        label_onsets[lab].sort()

    return dict(label_onsets)


def reconstruct_segments(
    label_onsets: Dict[str, List[float]],
) -> Dict[str, List[Tuple[float, Optional[float]]]]:
    """
    Pair consecutive same-label markers into ``(start, end)`` intervals.

    If a label has an odd number of markers the final unpaired marker is
    stored as ``(onset, None)`` so that inspection flags it instead of the
    segment silently disappearing. TSST and INTERO are reconstructed too, for
    the record, although they are not preprocessed.
    """
    segments: Dict[str, List[Tuple[float, Optional[float]]]] = {}
    for label in ("LB", "INTERO", "IGT", "TSST", "IGT2"):
        onsets = label_onsets.get(label, [])
        pairs: List[Tuple[float, Optional[float]]] = []
        i = 0
        while i + 1 < len(onsets):
            pairs.append((onsets[i], onsets[i + 1]))
            i += 2
        if i < len(onsets):
            pairs.append((onsets[i], None))
        segments[label] = pairs
    return segments


# ---------------------------------------------------------------------------
# 1.9  SEGMENT INSPECTION  (human-readable console report)
# ---------------------------------------------------------------------------

def inspect_segments(
    subj_id: Union[int, str],
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
    sfreq: float,
) -> str:
    """Print and return the duration table for LB, IGT and IGT2.

    Durations are shown AFTER apply_segment_rules() — i.e. exactly what will
    be analysed — with the marked duration in brackets when the two differ.
    """
    lines: List[str] = [
        "",
        "=" * 84,
        f"  SEGMENT DURATION INSPECTION  (SUBJECT {subj_id})",
        "=" * 84,
    ]
    warns: List[str] = []
    for label in SEG_TYPE_ORDER:
        intervals = segments.get(label, [])
        if not intervals:
            lines.append(f"  {label:<5}:  NOT FOUND (no markers)")
            warns.append(f"{label}: no markers found")
            continue
        for start, end in intervals:
            if end is None:
                lines.append(
                    f"  {label:<5}:  UNPAIRED marker at {start / 60:.1f} min "
                    f"({start:.1f} s) — no end"
                )
                warns.append(f"{label}: unpaired marker — segment cannot be built")
                continue
            raw_dur = end - start
            t0, t1 = apply_segment_rules(label, start, end)
            dur = t1 - t0
            note = (f"  [marked {raw_dur:.0f} s, last {dur:.0f} s used]"
                    if abs(dur - raw_dur) > 1.0 else "")
            lo, hi = SEG_EXPECTED_SEC.get(label, (0.0, 1e9))
            flag = ""
            if raw_dur <= 0:
                flag = "  *** ZERO/NEGATIVE ***"
                warns.append(f"{label}: duration {raw_dur:.1f} s — duplicate press?")
            elif not lo <= dur <= hi:
                flag = f"  *** OUT OF RANGE (expected {lo:.0f}-{hi:.0f} s) ***"
                warns.append(f"{label}: {dur:.0f} s, expected {lo:.0f}-{hi:.0f} s")
            lines.append(
                f"  {label:<5}:  {dur:6.1f} s ({dur / 60:4.1f} min)  |  "
                f"{t0 / 60:6.1f} -> {t1 / 60:6.1f} min  |  "
                f"samples {int(t0 * sfreq)}-{int(t1 * sfreq)}{note}{flag}"
            )
    if warns:
        lines.append("-" * 84)
        lines.append("  WARNINGS:")
        lines.extend(f"    >> {w}" for w in warns)
    lines.append("=" * 84)
    report = "\n".join(lines)
    print(report)
    return report


def save_segments_json(
    subj_id: Union[int, str],
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
    out_dir: Path = QC_DIR,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"subject_{subj_id}_segments.json"
    payload = {
        lab: [{"start": s, "end": e} for s, e in ivs]
        for lab, ivs in segments.items()
        if not lab.startswith("_")
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_segments_json(
    subj_id: Union[int, str],
    out_dir: Path = QC_DIR,
) -> Dict[str, List[Tuple[float, Optional[float]]]]:
    path = out_dir / f"subject_{subj_id}_segments.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        lab: [(iv["start"], iv["end"]) for iv in ivs]
        for lab, ivs in data.items()
    }


# ---------------------------------------------------------------------------
# 1.10  PREPROCESSING  STEP FUNCTIONS  (Steps 0-7)
# ---------------------------------------------------------------------------

def save_step(
    raw: mne.io.Raw, subj_id: Union[int, str], step_name: str,
) -> Optional[Path]:
    """Save intermediate .fif if the flag is enabled and write manifests."""
    if not SAVE_INTERMEDIATE_FIF:
        return None
    if SAVE_INTERMEDIATE_STEPS is not None and step_name not in SAVE_INTERMEDIATE_STEPS:
        return None
    fdir = subject_fif_dir(subj_id)
    fname = f"{step_name}_raw.fif"
    path = fdir / fname
    raw.save(str(path), overwrite=OVERWRITE_FIF, verbose=False)
    logger.debug(f"S{subj_id}: saved {path.name}")

    # --- per-subject manifest (JSON) -------------------------------------
    manifest_path = fdir / "manifest.json"
    try:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {}
    except Exception:
        manifest = {}

    manifest[step_name] = {
        "filename": fname,
        "path": str(path),
        "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_channels": len(raw.ch_names),
        "sfreq": float(raw.info.get("sfreq", np.nan)),
    }
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning(f"S{subj_id}: failed to write manifest.json: {exc}")

    # --- global CSV manifest in QC folder --------------------------------
    try:
        QC_DIR.mkdir(parents=True, exist_ok=True)
        csvp = QC_DIR / "fif_manifest.csv"
        header = not csvp.exists()
        row = {
            "subject": subj_id,
            "step": step_name,
            "filename": fname,
            "path": str(path),
            "saved_at": manifest[step_name]["saved_at"],
            "n_channels": manifest[step_name]["n_channels"],
            "sfreq": manifest[step_name]["sfreq"],
        }
        pd.DataFrame([row]).to_csv(csvp, mode="a", header=header, index=False)
    except Exception as exc:
        logger.debug(f"S{subj_id}: could not append global manifest CSV: {exc}")

    return path


# ---- Step 0: Load --------------------------------------------------------

def step0_load_raw(
    subj_id: Union[int, str], input_dir: Path = INPUT_DIR,
) -> mne.io.Raw:
    """Load a subject's BrainVision recording (repairing renamed headers)."""
    vhdr = subject_vhdr(subj_id, input_dir)
    if not vhdr.exists():
        raise FileNotFoundError(f"VHDR not found: {vhdr}")
    vhdr = fix_vhdr_references(vhdr, subj_id)
    logger.info(f"S{subj_id} | Step 0  Load  {vhdr.name}")
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
    logger.info(
        f"S{subj_id} | {len(raw.ch_names)} ch, "
        f"sfreq={raw.info['sfreq']:.0f} Hz, dur={raw.times[-1]:.1f} s"
    )
    return raw


# ---- Step 1: Channel setup -----------------------------------------------

def step1_setup_channels(
    raw: mne.io.Raw, subj_id: Union[int, str],
) -> mne.io.Raw:
    """
    Select the expected 31 channels, set TP9 -> eog, rest -> eeg.

    If a channel from the expected list is missing because it served as the
    on-line recording reference, it is re-added as a flat (zero) channel via
    ``mne.add_reference_channels`` so that it can be reconstructed when the
    average reference is applied later.
    """
    logger.info(f"S{subj_id} | Step 1  Channel setup")

    # Detect actual recording reference from the header
    orig_vhdr = subject_vhdr(subj_id)
    ref_name = parse_vhdr_reference(orig_vhdr)
    logger.info(f"S{subj_id} | Recording reference (header): {ref_name}")

    # Re-add the reference channel if it is in our expected list
    for ch_name in EXPECTED_CHANNELS:
        if ch_name not in raw.ch_names:
            if ref_name and ch_name.lower() == ref_name.lower():
                logger.info(
                    f"S{subj_id} | Adding reference '{ch_name}' back as flat channel"
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    raw = mne.add_reference_channels(raw, ch_name)
            else:
                logger.warning(f"S{subj_id} | Channel '{ch_name}' not in data")

    # Keep only expected channels that exist
    pick = [ch for ch in EXPECTED_CHANNELS if ch in raw.ch_names]
    raw.pick(pick)

    # Assign types
    mapping = {ch: ("eog" if ch == EOG_CHANNEL else "eeg") for ch in raw.ch_names}
    raw.set_channel_types(mapping)

    n_eeg = sum(1 for v in mapping.values() if v == "eeg")
    n_eog = sum(1 for v in mapping.values() if v == "eog")
    logger.info(f"S{subj_id} | Step 1 done  {n_eeg} EEG + {n_eog} EOG")
    return raw


# ---- Step 2: Montage -----------------------------------------------------

def step2_set_montage(
    raw: mne.io.Raw, subj_id: Union[int, str],
) -> mne.io.Raw:
    logger.info(f"S{subj_id} | Step 2  Montage (standard_1020)")
    montage = mne.channels.make_standard_montage("standard_1020")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw.set_montage(montage, on_missing="warn", verbose=False)
    logger.info(f"S{subj_id} | Step 2 done")
    return raw


# ---- Step 3: Filtering ---------------------------------------------------

def step3_filter(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    high_pass: float,
    low_pass: float,
    dataset_label: str,
) -> mne.io.Raw:
    logger.info(f"S{subj_id} | Step 3  Filtering ({dataset_label})")
    freqs = [NOTCH_FREQ]
    if NOTCH_HARMONICS:
        freqs.append(NOTCH_FREQ * 2)
    # TP9 (typed "eog") is filtered with the EEG. MNE filters only data channels
    # by default, so the breath-hold version left TP9 with its DC offset (-13.6 mV
    # in 7E) and line noise, and the VEOG blink mask (TP9 - Fp1) subtracted a
    # filtered Fp1 from an unfiltered TP9. Filtering keeps blinks and eye movements.
    picks = None if LEGACY_BH_CHANNEL_EXCLUSIONS else ["eeg", "eog"]
    raw.notch_filter(freqs, picks=picks, verbose=False)
    raw.filter(high_pass, low_pass, picks=picks, verbose=False)
    logger.info(
        f"S{subj_id} | {dataset_label}: Notch {freqs} Hz, BP {high_pass}-{low_pass} Hz"
    )
    return raw


# ---- Step 5: Bad channels ------------------------------------------------

def _robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mad = max(mad, 1e-12)
    return (x - med) / (mad * 1.4826)


def _build_veog_blink_mask(
    veog_data_uv: np.ndarray,
    sfreq: float,
    n_windows: int,
    win_samp: int,
    method: str,
) -> np.ndarray:
    """Mark variance windows that contain a blink in the bipolar VEOG channel.

    Returns a boolean array of length ``n_windows`` where True = clean (keep
    for variance computation), False = blink (exclude).

    Two methods:

    * ``"step"`` — Luck step function (Luck 2014 / Applied ERP Data Analysis
      ch.8). A step kernel of ``BAD_CHAN_VEOG_STEP_WINDOW_SEC`` seconds slides
      across the entire VEOG signal; a blink is detected wherever the absolute
      difference between the means of the kernel's two halves exceeds
      ``BAD_CHAN_VEOG_STEP_THRESH_UV``. Any variance window that overlaps the
      ``step_samp``-wide neighborhood of a detected step is excluded. Detecting
      across the whole signal (rather than within each variance window) is
      essential: blink rising edges that straddle a window boundary would
      otherwise be missed by both step- and PTP-style within-window checks.
    * ``"ptp"`` — within-window peak-to-peak threshold; legacy fallback.
    """
    valid = np.ones(n_windows, dtype=bool)
    if n_windows == 0:
        return valid

    if method == "ptp":
        thresh = BAD_CHAN_VEOG_BLINK_THRESH_UV
        for win_idx in range(n_windows):
            start = win_idx * win_samp
            stop = start + win_samp
            if np.ptp(veog_data_uv[start:stop]) > thresh:
                valid[win_idx] = False
        return valid

    # "step" — Luck step function across the full signal, then map onto windows.
    step_samp = max(2, int(round(BAD_CHAN_VEOG_STEP_WINDOW_SEC * sfreq)))
    if step_samp % 2:
        step_samp += 1  # keep two equal halves
    half = step_samp // 2
    stride = max(1, int(round(step_samp * BAD_CHAN_VEOG_STEP_STRIDE_FRAC)))
    thresh = BAD_CHAN_VEOG_STEP_THRESH_UV
    n_sig = veog_data_uv.size
    if n_sig < step_samp:
        # signal shorter than one step kernel — fall back to a single PTP check
        if np.ptp(veog_data_uv) > thresh * 2:
            valid[:] = False
        return valid

    # Vectorized sliding-window means via cumulative sums (Bottleneck-free).
    cumsum = np.concatenate(([0.0], np.cumsum(veog_data_uv, dtype=np.float64)))
    starts = np.arange(0, n_sig - step_samp + 1, stride)
    first_half_sum = cumsum[starts + half] - cumsum[starts]
    second_half_sum = cumsum[starts + step_samp] - cumsum[starts + half]
    diff = np.abs(second_half_sum - first_half_sum) / half  # mean difference
    step_starts = starts[diff > thresh]
    # Each detected step contaminates [start, start + step_samp)
    for s in step_starts:
        blink_start = max(0, s)
        blink_end = min(n_sig, s + step_samp)
        w_first = blink_start // win_samp
        w_last = (blink_end - 1) // win_samp
        valid[w_first : min(n_windows, w_last + 1)] = False
    return valid


def _run_lof_detection(
    raw: mne.io.Raw,
    eeg_names: List[str],
) -> Tuple[List[str], Dict[str, float], bool, str]:
    """Wrap ``mne.preprocessing.find_bad_channels_lof`` with graceful fallback.

    Picks are passed as channel names so ordering of returned scores is
    unambiguous: ``scores[i]`` corresponds to ``eeg_names[i]``.

    Returns ``(bads, scores_by_name, available, reason)`` where:
      * ``bads``           — channel names flagged by LOF (subset of eeg_names).
      * ``scores_by_name`` — per-channel ``abs(LOF score)``; higher = more
        anomalous. ``abs`` matches what the threshold compares against
        internally, so values are directly interpretable next to the
        configured threshold (default 1.5).
      * ``available``      — True if LOF ran; False on import/runtime failure.
      * ``reason``         — short explanation when ``available`` is False.
    """
    lof_func = getattr(mne.preprocessing, "find_bad_channels_lof", None)
    if lof_func is None:
        return [], {}, False, "mne.preprocessing.find_bad_channels_lof unavailable"
    try:
        result = lof_func(
            raw,
            n_neighbors=BAD_CHAN_LOF_N_NEIGHBORS,
            picks=eeg_names,
            metric=BAD_CHAN_LOF_METRIC,
            threshold=BAD_CHAN_LOF_THRESHOLD,
            return_scores=True,
            verbose=False,
        )
    except TypeError:
        # Older MNE without return_scores
        try:
            bads = lof_func(
                raw,
                n_neighbors=BAD_CHAN_LOF_N_NEIGHBORS,
                picks=eeg_names,
                metric=BAD_CHAN_LOF_METRIC,
                threshold=BAD_CHAN_LOF_THRESHOLD,
                verbose=False,
            )
            return [str(b) for b in bads], {}, True, ""
        except Exception as exc:
            return [], {}, False, f"LOF failed: {exc}"
    except Exception as exc:
        return [], {}, False, f"LOF failed: {exc}"

    bads, scores = result
    scores_by_name: Dict[str, float] = {}
    if scores is not None:
        for name, score in zip(eeg_names, np.abs(np.asarray(scores))):
            try:
                scores_by_name[name] = float(score)
            except (TypeError, ValueError):
                continue
    return [str(b) for b in bads], scores_by_name, True, ""


def _compute_bad_channel_diagnostics(
    raw: mne.io.Raw,
    detect_idx: List[int],
    eeg_names_detect: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Compute per-channel metrics for bad-channel detection.

    Always reports: strict flatline (std<=eps), LOF score (if enabled), and
    persistent-variance windows. The final decision is made in
    :func:`step5_bad_channels` based on the configured detectors.
    """
    data = raw.get_data(picks=detect_idx)
    ch_std_uv = np.std(data, axis=1) * 1e6
    hard_flatline_mask = ch_std_uv <= HARD_FLATLINE_EPS_UV
    low_activity_mask = (ch_std_uv > HARD_FLATLINE_EPS_UV) & (ch_std_uv < FLATLINE_THRESH_UV)

    # --- LOF (primary detector) ---------------------------------------------
    lof_bads: List[str] = []
    lof_scores: Dict[str, float] = {}
    lof_available = False
    lof_reason = ""
    if BAD_CHAN_USE_LOF:
        lof_bads, lof_scores, lof_available, lof_reason = _run_lof_detection(
            raw, eeg_names_detect
        )
        if not lof_available:
            logger.warning(f"LOF detection unavailable: {lof_reason}")

    # --- Persistent variance (legacy / QC cross-check) ----------------------
    win_samp = max(1, int(BAD_CHAN_WINDOW_SEC * raw.info["sfreq"]))
    n_windows = data.shape[1] // win_samp
    bad_counts = np.zeros(len(eeg_names_detect), dtype=int)
    max_window_z = np.full(len(eeg_names_detect), np.nan)
    valid_window_mask = np.ones(n_windows, dtype=bool)

    veog_available = BAD_CHAN_VEOG_CH_POS in raw.ch_names and BAD_CHAN_VEOG_CH_NEG in raw.ch_names
    veog_blink_ratio = 0.0
    veog_windows_excluded = 0
    veog_method_used = BAD_CHAN_VEOG_MASK_METHOD
    if BAD_CHAN_EXCLUDE_BLINK_WINDOWS and n_windows > 0 and veog_available:
        veog_data_uv = (
            raw.get_data(picks=[BAD_CHAN_VEOG_CH_POS])[0]
            - raw.get_data(picks=[BAD_CHAN_VEOG_CH_NEG])[0]
        ) * 1e6
        valid_window_mask = _build_veog_blink_mask(
            veog_data_uv,
            sfreq=raw.info["sfreq"],
            n_windows=n_windows,
            win_samp=win_samp,
            method=veog_method_used,
        )
        veog_windows_excluded = int((~valid_window_mask).sum())
        veog_blink_ratio = (veog_windows_excluded / n_windows) if n_windows > 0 else 0.0

    n_clean_windows_initial = int(valid_window_mask.sum())
    fallback_to_unmasked = bool(
        BAD_CHAN_FALLBACK_TO_UNMASKED_VARIANCE
        and n_windows > 0
        and n_clean_windows_initial < BAD_CHAN_MIN_CLEAN_WINDOWS
    )
    if fallback_to_unmasked:
        valid_window_mask = np.ones(n_windows, dtype=bool)
        veog_windows_excluded = 0
        veog_blink_ratio = 0.0

    n_clean_windows = int(valid_window_mask.sum())
    variance_eligible = n_clean_windows >= BAD_CHAN_MIN_CLEAN_WINDOWS

    if n_windows > 0:
        for win_idx in range(n_windows):
            if not valid_window_mask[win_idx]:
                continue
            start = win_idx * win_samp
            stop = start + win_samp
            win_var = np.var(data[:, start:stop], axis=1)
            zsc = _robust_zscore(win_var)
            bad_counts += (zsc > BAD_CHAN_ZSCORE_THRESH).astype(int)
            max_window_z = np.fmax(max_window_z, zsc)

    bad_ratio = (bad_counts / n_clean_windows) if n_clean_windows > 0 else np.zeros(len(eeg_names_detect))

    diagnostics: Dict[str, Dict[str, Any]] = {}
    for i, name in enumerate(eeg_names_detect):
        flagged_hard_flatline = bool(hard_flatline_mask[i])
        flagged_variance = bool(variance_eligible and bad_ratio[i] >= BAD_CHAN_PERSIST_RATIO)
        flagged_lof = bool(BAD_CHAN_USE_LOF and lof_available and name in lof_bads)
        lof_score = lof_scores.get(name)
        reasons: List[str] = []
        if flagged_hard_flatline:
            reasons.append("hard_flatline")
        if flagged_lof:
            reasons.append("lof")
        if flagged_variance:
            reasons.append("persistent_variance")
        diagnostics[name] = {
            "std_uv": float(ch_std_uv[i]),
            "hard_flatline_eps_uv": float(HARD_FLATLINE_EPS_UV),
            "flagged_hard_flatline": flagged_hard_flatline,
            # Backward-compatible alias used by previous QC consumers.
            "flagged_flatline": flagged_hard_flatline,
            "low_activity_threshold_uv": float(FLATLINE_THRESH_UV),
            "flagged_low_activity": bool(low_activity_mask[i]),
            "lof_enabled": bool(BAD_CHAN_USE_LOF),
            "lof_available": bool(lof_available),
            "lof_unavailable_reason": lof_reason,
            "lof_score": (float(lof_score) if lof_score is not None else None),
            "lof_threshold": float(BAD_CHAN_LOF_THRESHOLD),
            "lof_n_neighbors": int(BAD_CHAN_LOF_N_NEIGHBORS),
            "lof_metric": str(BAD_CHAN_LOF_METRIC),
            "flagged_lof": flagged_lof,
            "persistent_variance_enabled": bool(BAD_CHAN_USE_PERSISTENT_VARIANCE),
            "variance_windows_above_threshold": int(bad_counts[i]),
            "variance_windows_total": int(n_windows),
            "variance_windows_used_initial": int(n_clean_windows_initial),
            "variance_windows_used": int(n_clean_windows),
            "variance_windows_excluded_blink": int(veog_windows_excluded),
            "variance_eligible_for_decision": bool(variance_eligible),
            "variance_min_clean_windows": int(BAD_CHAN_MIN_CLEAN_WINDOWS),
            "variance_fallback_to_unmasked": bool(fallback_to_unmasked),
            "variance_bad_ratio": float(bad_ratio[i]) if n_windows > 0 else 0.0,
            "variance_zscore_threshold": float(BAD_CHAN_ZSCORE_THRESH),
            "variance_persist_ratio_threshold": float(BAD_CHAN_PERSIST_RATIO),
            "max_window_variance_zscore": (
                float(max_window_z[i]) if n_clean_windows > 0 and np.isfinite(max_window_z[i]) else None
            ),
            "flagged_variance": flagged_variance,
            "veog_mask_enabled": bool(BAD_CHAN_EXCLUDE_BLINK_WINDOWS),
            "veog_mask_method": veog_method_used,
            "veog_channels_available": bool(veog_available),
            "veog_channel_pos": BAD_CHAN_VEOG_CH_POS,
            "veog_channel_neg": BAD_CHAN_VEOG_CH_NEG,
            "veog_blink_threshold_uv": float(BAD_CHAN_VEOG_BLINK_THRESH_UV),
            "veog_step_window_sec": float(BAD_CHAN_VEOG_STEP_WINDOW_SEC),
            "veog_step_threshold_uv": float(BAD_CHAN_VEOG_STEP_THRESH_UV),
            "veog_blink_window_ratio": float(veog_blink_ratio),
            "heuristic_reasons": reasons,
        }
    return diagnostics


def step5_bad_channels(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    context: Optional[str] = None,
) -> Tuple[mne.io.Raw, List[str], Dict[str, Dict[str, Any]]]:
    """Detect bad EEG channels.

    Defaults (configurable above):
      * Strict flatline (std<=eps) — always applied as safety net.
      * LOF (``BAD_CHAN_USE_LOF``) — primary detector, density-based outlier
        score from ``mne.preprocessing.find_bad_channels_lof`` (Kumaravel 2022).
      * Persistent variance (``BAD_CHAN_USE_PERSISTENT_VARIANCE``) — computed
        on blink-free windows only; skipped for a segment with fewer than
        ``BAD_CHAN_MIN_CLEAN_WINDOWS`` of them (see Cell 1).

    Only marks channels in raw.info["bads"] — does NOT interpolate. Interpolation
    happens after ICA in :func:`step_interpolate_bad_channels`. Channels in
    BAD_CHAN_EXCLUDE_CHANNELS (Fp1/Fp2 — see Cell 1) are not tested. Manual
    marks are added to this result by run_preprocessing_steps.
    """
    scope = f" [{context}]" if context else ""
    method_summary = "flatline"
    if BAD_CHAN_USE_LOF:
        method_summary += "+lof"
    if BAD_CHAN_USE_PERSISTENT_VARIANCE:
        method_summary += "+variance"
    logger.info(
        f"S{subj_id} | Step 4  Bad-channel detection{scope} ({method_summary})"
    )

    eeg_idx = mne.pick_types(raw.info, eeg=True, eog=False)
    eeg_names = [raw.ch_names[i] for i in eeg_idx]

    excluded_chs = set(BAD_CHAN_EXCLUDE_CHANNELS)
    eeg_names_detect = [n for n in eeg_names if n not in excluded_chs]
    detect_idx = [eeg_idx[eeg_names.index(n)] for n in eeg_names_detect]

    diagnostics = _compute_bad_channel_diagnostics(raw, detect_idx, eeg_names_detect)
    bads: List[str] = []

    for name, info in diagnostics.items():
        if info["flagged_hard_flatline"]:
            bads.append(name)
            logger.info(f"S{subj_id} | Hard flatline: {name}  std={info['std_uv']:.6f} uV")

    if BAD_CHAN_USE_LOF:
        for name, info in diagnostics.items():
            if name in bads:
                continue
            if info.get("flagged_lof", False):
                bads.append(name)
                score_str = (
                    f"{info['lof_score']:.2f}" if info.get("lof_score") is not None else "n/a"
                )
                logger.info(
                    f"S{subj_id} | LOF outlier: {name}  "
                    f"score={score_str}  threshold={info['lof_threshold']}"
                )

    if BAD_CHAN_USE_PERSISTENT_VARIANCE:
        _first = next(iter(diagnostics.values()), {})
        if _first and not _first["variance_eligible_for_decision"]:
            logger.info(
                f"S{subj_id} | Variance check skipped{scope}: "
                f"{_first['variance_windows_used']}/{_first['variance_windows_total']} "
                f"blink-free windows (< {BAD_CHAN_MIN_CLEAN_WINDOWS})"
            )
        for name, info in diagnostics.items():
            if name in bads:
                continue
            if info["flagged_variance"]:
                bads.append(name)
                logger.info(
                    f"S{subj_id} | Persistent variance outlier: {name}  "
                    f"{info['variance_windows_above_threshold']}/"
                    f"{info['variance_windows_used']} clean windows "
                    f"({info['variance_bad_ratio']*100:.0f}%)"
                )

    for name, info in diagnostics.items():
        active_methods: List[str] = ["flatline"]
        if BAD_CHAN_USE_LOF:
            active_methods.append("lof")
        if BAD_CHAN_USE_PERSISTENT_VARIANCE:
            active_methods.append("persistent_variance")
        info["detection_method"] = "+".join(active_methods)
        info["detected_by_pipeline"] = name in bads
        info["pipeline_reasons"] = (
            list(info["heuristic_reasons"]) if name in bads else []
        )

    raw.info["bads"] = bads
    logger.info(f"S{subj_id} | Bad channels marked{scope}: {bads if bads else 'none'}")
    logger.info(f"S{subj_id} | Step 4 done")
    return raw, bads, diagnostics


# ---- Step 6: Artifact annotation -----------------------------------------

def step6_annotate_artifacts(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    context: Optional[str] = None,
) -> Tuple[mne.io.Raw, mne.Annotations]:
    """Mark 1-s windows as BAD based on two automatic checks:

    1. **BAD_artifact**: peak-to-peak amplitude exceeds ARTIFACT_PTP_THRESH_UV on any
       channel outside ARTIFACT_PTP_EXCLUDE_CHANNELS (movement, electrode pops;
       blinks on Fp1/Fp2 are left to ICA).
    2. **BAD_muscle**: gamma-band (30-40 Hz) power exceeds MUSCLE_GAMMA_THRESH_UV2 on
       any channel (high-frequency bursts from muscle activity).

    Windows already flagged as BAD_artifact are not double-flagged as BAD_muscle.
    If AUTO_ARTIFACT_DETECTION is False, this step is skipped entirely.
    """
    scope = f" [{context}]" if context else ""
    logger.info(f"S{subj_id} | Step 5  Artifact annotation{scope}")

    if not AUTO_ARTIFACT_DETECTION:
        logger.info(f"S{subj_id} | Auto artifact detection disabled, skipping{scope}")
        return raw, mne.Annotations([], [], [], orig_time=raw.annotations.orig_time)

    eeg_idx = mne.pick_types(raw.info, eeg=True, eog=False)
    sfreq = raw.info["sfreq"]
    win_samp = int(1.0 * sfreq)
    data = raw.get_data(picks=eeg_idx)
    n_samp = data.shape[1]
    thresh_v = ARTIFACT_PTP_THRESH_UV * 1e-6
    ptp_rows = np.array([raw.ch_names[i] not in ARTIFACT_PTP_EXCLUDE_CHANNELS for i in eeg_idx])

    # Gamma band limits for muscle detection
    gamma_low, gamma_high = MUSCLE_GAMMA_BAND
    muscle_thresh_v2hz = MUSCLE_GAMMA_THRESH_UV2 * 1e-12  # uV^2/Hz -> V^2/Hz

    ptp_onsets, ptp_durs = [], []
    muscle_onsets, muscle_durs = [], []

    for s in range(0, n_samp - win_samp, win_samp):
        seg = data[:, s : s + win_samp]
        t_onset = s / sfreq

        # --- PTP check ---
        ptp = seg[ptp_rows].max(axis=1) - seg[ptp_rows].min(axis=1)
        if ptp.max() > thresh_v:
            ptp_onsets.append(t_onset)
            ptp_durs.append(1.0)
            continue  # don't also flag as muscle if already flagged

        # --- Gamma-band muscle check ---
        try:
            eff_nfft = min(INSPECT_PSD_N_FFT, win_samp)
            eff_noverlap = min(INSPECT_PSD_N_OVERLAP, eff_nfft - 1)
            psd_win, freqs_win = mne.time_frequency.psd_array_welch(
                seg, sfreq=sfreq,
                fmin=gamma_low, fmax=gamma_high,
                n_fft=eff_nfft, n_overlap=eff_noverlap,
                verbose=False,
            )
            # psd_win: (n_ch, n_freqs) in V^2/Hz; average across gamma band
            gamma_mean = psd_win.mean(axis=1)  # per channel
            if gamma_mean.max() > muscle_thresh_v2hz:
                muscle_onsets.append(t_onset)
                muscle_durs.append(1.0)
        except Exception:
            pass  # too short or degenerate window — skip silently

    auto_annotations = mne.Annotations([], [], [], orig_time=raw.annotations.orig_time)
    new_annotations = raw.annotations
    if ptp_onsets:
        ptp_annotations = mne.Annotations(
            onset=ptp_onsets, duration=ptp_durs,
            description=["BAD_artifact"] * len(ptp_onsets),
            orig_time=raw.annotations.orig_time,
        )
        auto_annotations = auto_annotations + ptp_annotations
        new_annotations = new_annotations + ptp_annotations
        logger.info(f"S{subj_id} | {len(ptp_onsets)} s flagged as BAD_artifact{scope}")
    else:
        logger.info(f"S{subj_id} | No gross artefacts detected (PTP){scope}")

    if muscle_onsets:
        muscle_annotations = mne.Annotations(
            onset=muscle_onsets, duration=muscle_durs,
            description=["BAD_muscle"] * len(muscle_onsets),
            orig_time=raw.annotations.orig_time,
        )
        auto_annotations = auto_annotations + muscle_annotations
        new_annotations = new_annotations + muscle_annotations
        logger.info(f"S{subj_id} | {len(muscle_onsets)} s flagged as BAD_muscle{scope}")
    else:
        logger.info(f"S{subj_id} | No muscle artefacts detected (gamma){scope}")

    safe_set_annotations(raw, new_annotations)
    logger.info(f"S{subj_id} | Step 5 done")
    return raw, auto_annotations


# ---- Step 6: ICA ---------------------------------------------------------

def step7_ica(
    raw_ica: mne.io.Raw,
    raw_apply: mne.io.Raw,
    subj_id: Union[int, str],
    fit_segments: Optional[List[str]] = None,
) -> mne.io.Raw:
    """
    Fit ICA on a temporary high-pass-filtered dataset and apply it to the final dataset.

    Decision logic:
    * Save auto-suggestion to ``ica_suggest.json``.
    * If ``ica_exclude.json`` exists (manual override), use it.
    * Otherwise use the suggestion and write a template for future manual edits.
    """
    logger.info(f"S{subj_id} | Step 6  ICA")
    qdir = subject_qc_dir(subj_id)

    rank_dict = mne.compute_rank(raw_ica, rank="info", verbose=False)
    eeg_rank = rank_dict.get("eeg", None)
    n_comp = ICA_N_COMPONENTS
    # If n_comp is None, fall back to variance-based (0.999999) to avoid unstable mixing matrix
    if n_comp is None:
        n_comp = 0.999999
    logger.info(f"S{subj_id} | EEG rank={eeg_rank}, n_components={n_comp}")

    ica = ICA(
        n_components=n_comp, method=ICA_METHOD,
        random_state=ICA_RANDOM_STATE, max_iter=ICA_MAX_ITER,
    )
    ica.fit(raw_ica, picks="eeg", reject_by_annotation=True, verbose=False)
    logger.info(f"S{subj_id} | ICA fitted ({ica.n_components_} components)")

    if SAVE_QC_FILES:
        ica.save(str(qdir / "ica_solution-ica.fif"), overwrite=True, verbose=False)

    # EOG component detection (robust fallback: TP9 + frontal channels).
    per_channel: Dict[str, Dict[str, Any]] = {}
    eog_idx_set: set[int] = set()
    primary_scores = np.array([])
    for ch_name in ICA_EOG_CHANNEL_CANDIDATES:
        if ch_name not in raw_ica.ch_names:
            continue
        try:
            idx, scores = ica.find_bads_eog(raw_ica, ch_name=ch_name, verbose=False)
            idx_list = [int(x) for x in idx]
            max_abs = float(np.max(np.abs(scores))) if scores.size else 0.0

            # If MNE auto-threshold misses obvious frontal ocular component,
            # include strongest component when the absolute score is high.
            if not idx_list and max_abs >= ICA_EOG_FALLBACK_MIN_ABS_SCORE:
                idx_list = [int(np.argmax(np.abs(scores)))]
                logger.info(
                    f"S{subj_id} | ICA EOG fallback on {ch_name}: "
                    f"max|score|={max_abs:.3f} -> component {idx_list[0]}"
                )

            per_channel[ch_name] = {
                "auto_eog_indices": idx_list,
                "max_abs_score": max_abs,
                "scores": scores.tolist() if scores.size else [],
            }
            eog_idx_set.update(idx_list)
            if ch_name == EOG_CHANNEL:
                primary_scores = scores
        except Exception as exc:
            logger.warning(f"S{subj_id} | EOG detection error on {ch_name}: {exc}")

    eog_idx = sorted(eog_idx_set)

    # Cap: keep only top ICA_MAX_EOG_COMPONENTS by max |score| across detection channels.
    # Prevents removing low-variance components (e.g. indices 20-23) that happen to
    # correlate with noisy frontal channels but are not genuine eye components.
    if len(eog_idx) > ICA_MAX_EOG_COMPONENTS:
        n_comp = int(ica.n_components_)
        comp_max_score = np.zeros(n_comp)
        for ch_info in per_channel.values():
            ch_scores = np.array(ch_info.get("scores", []))
            if ch_scores.size == n_comp:
                comp_max_score = np.maximum(comp_max_score, np.abs(ch_scores))
        eog_idx_ranked = sorted(eog_idx, key=lambda i: comp_max_score[i] if i < n_comp else 0.0, reverse=True)
        eog_idx = sorted(eog_idx_ranked[:ICA_MAX_EOG_COMPONENTS])
        logger.info(f"S{subj_id} | ICA EOG cap: kept {eog_idx} (from {sorted(eog_idx_set)})")

    # High-index guard: late FastICA components (likely low-variance) must have a minimum TP9
    # score to be accepted as eye artifacts. Prevents marginal false positives like S10 component 23.
    if ICA_HIGH_INDEX_THRESHOLD is not None and eog_idx:
        tp9_info = per_channel.get(EOG_CHANNEL, {})
        tp9_scores_arr = np.array(tp9_info.get("scores", []))
        filtered: List[int] = []
        for ci in eog_idx:
            if ci > ICA_HIGH_INDEX_THRESHOLD:
                tp9_s = float(abs(tp9_scores_arr[ci])) if tp9_scores_arr.size > ci else 0.0
                if tp9_s < ICA_HIGH_INDEX_MIN_TP9_SCORE:
                    logger.info(
                        f"S{subj_id} | ICA high-index guard: component {ci} "
                        f"(index>{ICA_HIGH_INDEX_THRESHOLD}) rejected — "
                        f"TP9 score={tp9_s:.3f} < {ICA_HIGH_INDEX_MIN_TP9_SCORE}"
                    )
                    continue
            filtered.append(ci)
        if filtered != eog_idx:
            logger.info(f"S{subj_id} | ICA high-index guard: {eog_idx} -> {filtered}")
        eog_idx = filtered

    if eog_idx:
        logger.info(f"S{subj_id} | Auto EOG components (combined): {eog_idx}")
    else:
        logger.warning(
            f"S{subj_id} | Auto EOG found no components using {ICA_EOG_CHANNEL_CANDIDATES}"
        )

    suggestion = {
        "auto_eog_indices": [int(x) for x in eog_idx],
        "eog_scores": primary_scores.tolist() if primary_scores.size else [],
        "eog_detection_channels": list(ICA_EOG_CHANNEL_CANDIDATES),
        "eog_detection_per_channel": per_channel,
        "n_components": int(ica.n_components_),
        "method": ICA_METHOD,
        "fit_segments": fit_segments or [],
    }
    (qdir / "ica_suggest.json").write_text(
        json.dumps(suggestion, indent=2), encoding="utf-8",
    )

    # Save topography plots (non-interactive)
    if SAVE_QC_FILES:
        try:
            prev_backend = plt.get_backend()
            plt.switch_backend("Agg")
            figs = ica.plot_components(show=False)
            if not isinstance(figs, list):
                figs = [figs]
            for i, fig in enumerate(figs):
                fig.savefig(str(qdir / f"ica_topomap_{i}.png"), dpi=120)
                plt.close(fig)
            plt.switch_backend(prev_backend)
        except Exception as exc:
            logger.warning(f"S{subj_id} | ICA plot failed: {exc}")

    # Decide which components to exclude (ocular auto-suggestions only).
    auto_exclude = sorted(set([int(x) for x in eog_idx]))
    # Fingerprint of THIS decomposition. Component numbers only mean something
    # for the ICA they came from: if the input changes (new BAD marks, other
    # bad channels, a segment excluded) the refit renumbers the components and
    # an old "exclude" list would remove the wrong ones. An edited list is
    # therefore honoured only when it was written for an identical fit.
    import hashlib as _hashlib
    fingerprint = _hashlib.sha1(
        np.round(ica.mixing_matrix_, 6).tobytes() + ",".join(ica.ch_names).encode()
    ).hexdigest()[:16]
    override_path = qdir / "ica_exclude.json"
    exclude = auto_exclude
    manual_override = False
    if override_path.exists() and ICA_RESPECT_MANUAL_OVERRIDE_JSON:
        try:
            override = json.loads(override_path.read_text(encoding="utf-8"))
            if override.get("ica_fingerprint") == fingerprint:
                exclude = [int(x) for x in override.get("exclude", auto_exclude)]
                manual_override = sorted(exclude) != auto_exclude
                if manual_override:
                    logger.info(f"S{subj_id} | Manual ICA override -> {exclude}")
                    print(f"           ICA: using your edited list {exclude} "
                          f"(automatic suggestion: {auto_exclude})")
            elif override.get("exclude") != override.get("auto_eog_indices"):
                logger.warning(
                    f"S{subj_id} | ica_exclude.json was edited for a different "
                    f"decomposition -> ignored, using {auto_exclude}"
                )
                print("           !! ICA was refitted because the input changed, so the "
                      "components were renumbered and your earlier edit of "
                      f"ica_exclude.json no longer applies. Using {auto_exclude}; "
                      "check the new topomaps and edit the file again if needed.")
        except Exception:
            exclude = auto_exclude
    (qdir / "ica_exclude.json").write_text(
        json.dumps({
            "exclude": exclude,
            "_comment": (
                "Edit 'exclude' to override the automatic choice, then re-run "
                "Cell 4. The edit applies only while 'ica_fingerprint' matches the "
                "current decomposition: if the input changes, ICA is refitted, the "
                "components are renumbered and the edit is discarded with a warning."
            ),
            "auto_eog_indices": [int(x) for x in eog_idx],
            "ica_fingerprint": fingerprint,
            "manual_override": manual_override,
        }, indent=2),
        encoding="utf-8",
    )

    ica.exclude = exclude
    # ICA was fit on all EEG channels (none marked bad at fit time).
    # raw_apply may already have bads set from step 4; temporarily clear them
    # so that ica.apply() sees the same channel set that was used for fitting.
    # The bads list is restored immediately after so that step 7 (interpolation)
    # can still act on the right channels.
    _saved_bads = list(raw_apply.info["bads"])
    raw_apply.info["bads"] = []
    raw_apply = ica.apply(raw_apply, verbose=False)
    raw_apply.info["bads"] = _saved_bads
    logger.info(
        f"S{subj_id} | Step 6 done - removed {len(exclude)} component(s)"
    )
    return raw_apply


# ---- Step 7: Interpolate bad channels (after ICA) -------------------------

def step_interpolate_bad_channels(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    bads: List[str],
    context: Optional[str] = None,
) -> mne.io.Raw:
    """Interpolate bad channels detected in step5_bad_channels.

    Called AFTER ICA so that bad channels do not corrupt the decomposition.
    If more than BAD_CHAN_MAX_INTERP_RATIO (10%) of EEG channels are bad,
    a warning is printed — the segment may need to be rejected downstream.
    """
    scope = f" [{context}]" if context else ""
    logger.info(f"S{subj_id} | Step 7  Interpolating bad channels{scope}")

    # Re-set bads in case ICA modified raw.info["bads"]
    raw.info["bads"] = bads

    if not bads:
        logger.info(f"S{subj_id} | No bad channels to interpolate{scope}")
        logger.info(f"S{subj_id} | Step 7 done")
        return raw

    # All EEG channels, not just the good ones (bad/total, not bad/good).
    n_eeg = len(mne.pick_types(raw.info, eeg=True, eog=False, exclude=[]))
    ratio = len(bads) / n_eeg if n_eeg > 0 else 0.0
    if ratio > BAD_CHAN_MAX_INTERP_RATIO:
        msg = (
            f"S{subj_id} | {context or 'Segment'}: {len(bads)}/{n_eeg} EEG channels bad "
            f"({ratio*100:.0f}% > {BAD_CHAN_MAX_INTERP_RATIO*100:.0f}%). "
            f"Consider excluding it from analysis."
        )
        logger.warning(msg)

    logger.info(f"S{subj_id} | Interpolating {len(bads)} channel(s){scope}: {bads}")
    raw.interpolate_bads(reset_bads=True, verbose=False)
    logger.info(f"S{subj_id} | Step 7 done")
    return raw


# ---- Step 8: Average re-reference (after ICA + interpolation) -------------

def step_rereference(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    context: Optional[str] = None,
) -> mne.io.Raw:
    """Re-reference all EEG channels to the common average.

    Applied AFTER ICA and bad-channel interpolation to avoid rank deficiency
    during ICA fitting (Kim et al. 2023; Bigdely-Shamlo et al. 2015). TP9
    (typed eog) is left out of the average by MNE but stays in the data.

    Only channels in AVG_REF_EXCLUDE_CHANNELS — a channel flat because it was
    the online reference — are left out of the average and re-created by
    interpolation afterwards. For the alcohol recordings the list is empty
    (reference TP10 is not a data channel), so all 30 EEG channels, Cz, Fp1
    and Fp2 included, enter the average as real data.
    """
    scope = f" [{context}]" if context else ""
    logger.info(f"S{subj_id} | Step 8  Average reference{scope}")

    excluded_ref_chs: List[str] = []
    for ref_ch in AVG_REF_EXCLUDE_CHANNELS:
        if ref_ch in raw.ch_names:
            if ref_ch not in raw.info["bads"]:
                raw.info["bads"].append(ref_ch)
            excluded_ref_chs.append(ref_ch)
    if excluded_ref_chs:
        logger.info(
            f"S{subj_id} | Excluding reference-linked channel(s) from "
            f"average reference: {excluded_ref_chs}"
        )

    raw, _ = mne.set_eeg_reference(raw, ref_channels="average", verbose=False)

    if excluded_ref_chs:
        logger.info(
            f"S{subj_id} | Interpolating reference-linked channel(s) after "
            f"average reference{scope}: {excluded_ref_chs}"
        )
        raw.interpolate_bads(reset_bads=True, verbose=False)

    logger.info(f"S{subj_id} | Step 8 done")
    return raw


# ---------------------------------------------------------------------------
# 1.11  PREPROCESSING DRIVER  (Steps 1-8, takes already-loaded raw)
# ---------------------------------------------------------------------------

def run_preprocessing_steps(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
) -> Tuple[mne.io.Raw, List[str], Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Run Steps 1-8 on an already-loaded raw object.

    Pipeline order (Kim 2023; Klug 2024; Makoto):
      1. Channel setup
      2. Montage
      3. Create two filtered copies after montage:
         - ICA-fit dataset: 1-40 Hz + notch
         - Final dataset: 0.1-40 Hz + notch
      4. Detect bad channels by preprocessing unit (LB / IGT / IGT2): automatic + manual
      5. Annotate gross artifacts by segment on the ICA-fit dataset
      6. Fit ICA on concatenated analysis segments and apply it to the final dataset
      7. Interpolate bad channels per preprocessing unit
      8. Average re-reference per preprocessing unit

    Returns (raw_clean, bad_channels_union, ica_info, segment_qc).
    Raises on failure (caller should handle).
    """
    # Step 1 - channels
    print(f"  [Step 1] Setting up channels...")
    raw = step1_setup_channels(raw, subj_id)
    save_step(raw, subj_id, "step1_channels")

    # Step 2 - montage
    print(f"\n  [Step 2] Applying montage (standard_1020)...")
    raw = step2_set_montage(raw, subj_id)
    save_step(raw, subj_id, "step2_montage")

    # Step 3 - filtering
    print(
        f"\n  [Step 3a] Creating ICA-fit filtered copy "
        f"({ICA_FIT_HIGH_PASS:.1f}-{FILTER_LOW_PASS:.1f} Hz + notch)..."
    )
    raw_ica = step3_filter(
        raw.copy(), subj_id,
        high_pass=ICA_FIT_HIGH_PASS,
        low_pass=FILTER_LOW_PASS,
        dataset_label="ICA fit",
    )
    save_step(raw_ica, subj_id, "step3_ica_filtered")

    print(
        f"\n  [Step 3b] Creating final filtered copy "
        f"({FINAL_HIGH_PASS:.1f}-{FILTER_LOW_PASS:.1f} Hz + notch)..."
    )
    raw_final = step3_filter(
        raw.copy(), subj_id,
        high_pass=FINAL_HIGH_PASS,
        low_pass=FILTER_LOW_PASS,
        dataset_label="final output",
    )
    save_step(raw_final, subj_id, "step3_final_filtered")
    save_step(raw_final, subj_id, "step3_filtered")

    units = build_preprocessing_units(segments)
    if not units:
        raise RuntimeError(
            "No LB/IGT/IGT2 segment available for preprocessing "
            "(inspected in Cell 3 and not excluded in Cell 3d)."
        )

    # Channels marked bad for every segment at once (only when
    # APPLY_MANUAL_BAD_CHANNELS_GLOBALLY is True).
    _global_manual_bads = list(raw.info["bads"])

    # Step 4 - bad-channel detection by segment/unit
    print(f"\n  [Step 4] Detecting bad channels by segment...")
    segment_qc: Dict[str, Dict[str, Any]] = {}
    all_bads_set = set()
    unit_ica_raws: List[mne.io.Raw] = []
    included_units: List[Dict[str, Any]] = []
    auto_annotations_abs = mne.Annotations([], [], [], orig_time=raw_ica.annotations.orig_time)

    for unit in units:
        unit_raw_for_bads, _ = crop_and_concat_segments(raw_ica, unit["segments"])
        _, unit_bads, unit_bad_details = step5_bad_channels(
            unit_raw_for_bads, subj_id, context=unit["display_name"]
        )
        # Add the channels marked bad by hand in this segment's RAW viewer.
        # (In the breath-hold version these were reported and then discarded.)
        _auto_bads = sorted(unit_bads)
        _eeg_names = set(
            unit_raw_for_bads.ch_names[i]
            for i in mne.pick_types(unit_raw_for_bads.info, eeg=True, eog=False, exclude=[])
        )
        _manual_bads = sorted(
            ch for ch in set(MANUAL_BAD_CHANNELS.get(unit["label"], [])) | set(_global_manual_bads)
            if ch in _eeg_names and ch not in unit_bads
        )
        unit_bads = sorted(set(unit_bads) | set(_manual_bads))

        excluded_from_preprocessing = len(unit_bads) > BAD_CHAN_MAX_PER_UNIT
        exclusion_reason = (
            f">{BAD_CHAN_MAX_PER_UNIT} bad channels detected ({len(unit_bads)})"
            if excluded_from_preprocessing else ""
        )

        artifact_seconds = 0.0
        annotated_parts: List[mne.io.Raw] = []
        if not excluded_from_preprocessing:
            for seg in unit["segments"]:
                max_t = float(raw_ica.times[-1])
                tmin = max(0.0, float(seg["start"]))
                tmax = min(float(seg["end"]), max_t)
                seg_raw = raw_ica.copy().crop(tmin=tmin, tmax=tmax)
                seg_raw, auto_ann = step6_annotate_artifacts(seg_raw, subj_id, context=seg["name"])
                annotated_parts.append(seg_raw)
                artifact_seconds += float(np.sum(auto_ann.duration)) if len(auto_ann) else 0.0
                auto_annotations_abs = auto_annotations_abs + shift_annotations(
                    auto_ann, seg["start"], raw_ica.annotations.orig_time
                )

            unit_raw_ica = annotated_parts[0] if len(annotated_parts) == 1 else mne.concatenate_raws(
                annotated_parts, preload=True, verbose=False
            )
            unit_ica_raws.append(unit_raw_ica)
            included_units.append(unit)

        segment_qc[unit["label"]] = {
            "display_name": unit["display_name"],
            "segments": unit["segments"],
            "segment_names": [seg["name"] for seg in unit["segments"]],
            "bads": sorted(unit_bads),
            "bads_auto": _auto_bads,
            "bads_manual": _manual_bads,
            "bad_channel_diagnostics": unit_bad_details,
            "artifact_seconds": artifact_seconds,
            "excluded_from_preprocessing": bool(excluded_from_preprocessing),
            "exclusion_reason": exclusion_reason,
        }
        if not excluded_from_preprocessing:
            all_bads_set.update(unit_bads)

        if unit_bads:
            warn_suffix = " WARNING" if excluded_from_preprocessing else ""
            _man = f"  (by hand: {_manual_bads})" if _manual_bads else ""
            print(f"           {unit['display_name']}: {sorted(unit_bads)}{_man}{warn_suffix}")
        else:
            print(f"           {unit['display_name']}: no bad channels detected")

    bads = sorted(all_bads_set)
    raw_ica.info["bads"] = list(bads)
    raw_final.info["bads"] = list(bads)
    save_bad_channel_report(subj_id, bads, segment_qc)
    save_step(raw_ica, subj_id, "step4_bads_detected")

    if not included_units:
        raise RuntimeError(
            f"All preprocessing units were excluded (> {BAD_CHAN_MAX_PER_UNIT} bad channels)."
        )

    # Step 5 - artifact annotation by segment
    print(f"\n  [Step 5] Annotating gross artifacts by segment...")
    safe_set_annotations(raw_ica, raw_ica.annotations + auto_annotations_abs)
    safe_set_annotations(raw_final, raw_final.annotations + auto_annotations_abs)
    for unit in units:
        qc = segment_qc[unit["label"]]
        if qc.get("excluded_from_preprocessing", False):
            print(f"           {qc['display_name']}: excluded (skipped)")
        else:
            print(
                f"           {qc['display_name']}: "
                f"{qc['artifact_seconds']:.1f} s flagged as BAD_artifact/BAD_muscle"
            )
    save_step(raw_ica, subj_id, "step5_annotated")

    # Step 6 - ICA on concatenated analysis segments
    print(f"\n  [Step 6] Running ICA...")
    raw_ica_fit = unit_ica_raws[0] if len(unit_ica_raws) == 1 else mne.concatenate_raws(
        unit_ica_raws, preload=True, verbose=False
    )
    fit_segment_names = [unit["display_name"] for unit in included_units]
    raw_final = step7_ica(raw_ica_fit, raw_final, subj_id, fit_segments=fit_segment_names)
    print(f"           Fit on concatenated segments: {', '.join(fit_segment_names)}")
    save_step(raw_final, subj_id, "step6_ica")

    # Step 7 + Step 8 - clean each processing unit independently
    print(f"\n  [Step 7] Interpolating bad channels by segment...")
    for unit in included_units:
        qc = segment_qc[unit["label"]]
        unit_raw_final, unit_lengths = crop_and_concat_segments(raw_final, unit["segments"])
        unit_raw_final = step_interpolate_bad_channels(
            unit_raw_final, subj_id, qc["bads"], context=qc["display_name"]
        )
        write_unit_data_back(raw_final, unit_raw_final, unit["segments"], unit_lengths)
        if qc["bads"]:
            print(f"           {qc['display_name']}: interpolated {qc['bads']}")
        else:
            print(f"           {qc['display_name']}: no interpolation needed")
    save_step(raw_final, subj_id, "step7_interpolated")

    print(f"\n  [Step 8] Applying average reference by segment...")
    for unit in included_units:
        qc = segment_qc[unit["label"]]
        unit_raw_final, unit_lengths = crop_and_concat_segments(raw_final, unit["segments"])
        if not LEGACY_BH_CHANNEL_EXCLUSIONS:
            # raw_final still lists the union of EVERY segment's bad channels
            # (set in Step 4). On this segment, its own bad channels were
            # interpolated in Step 7 and the others are real data, so none is
            # bad here. Inheriting the union made the average skip good
            # channels, and the breath-hold version then re-interpolated them.
            unit_raw_final.info["bads"] = []
        unit_raw_final = step_rereference(unit_raw_final, subj_id, context=qc["display_name"])
        write_unit_data_back(raw_final, unit_raw_final, unit["segments"], unit_lengths)
        print(f"           {qc['display_name']}: average reference applied")
    # write_unit_data_back copies samples only; bring the metadata in line with
    # what was done to the analysed segments.
    raw_final.info["bads"] = []
    with raw_final.info._unlock():
        raw_final.info["custom_ref_applied"] = mne.io.constants.FIFF.FIFFV_MNE_CUSTOM_REF_ON
    save_step(raw_final, subj_id, "step8_refavg")

    # Gather ICA QC info
    ica_info: Dict[str, Any] = {"method": ICA_METHOD, "excluded": []}
    suggest_p = subject_qc_dir(subj_id) / "ica_suggest.json"
    if suggest_p.exists():
        ica_info.update(json.loads(suggest_p.read_text(encoding="utf-8")))
    excl_p = subject_qc_dir(subj_id) / "ica_exclude.json"
    if excl_p.exists():
        ica_info["excluded"] = json.loads(
            excl_p.read_text(encoding="utf-8")
        ).get("exclude", [])

    print(f"\n  Preprocessing complete.")
    print(f"  ICA components excluded: {ica_info.get('excluded', [])}")
    return raw_final, bads, ica_info, segment_qc


# ---------------------------------------------------------------------------
# 1.12  VISUALIZATION FUNCTIONS
# ---------------------------------------------------------------------------

def _iter_valid_segments(
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
) -> List[Tuple[str, int, float, float]]:
    """(seg_name, seg_idx_1based, start, end) for every valid interval."""
    result: List[Tuple[str, int, float, float]] = []
    for label in SEG_TYPE_ORDER:
        result.extend(get_segments_by_type(segments, label))
    return result


def segment_run_used_in_preproc(seg_type: str, seg_idx_1based: int = 1) -> bool:
    """A segment enters Cell 4 only if its Cell 3 viewer was run and it was
    not excluded in Cell 3d."""
    return bool(SEGMENTS_INSPECTED.get(seg_type, False)) and seg_type not in SEGMENT_EXCLUSIONS


def print_segment_preproc_status(
    seg_type: str,
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
) -> None:
    """One line: will this segment enter preprocessing?"""
    sched = get_segments_by_type(segments, seg_type, apply_time_rules=True)
    if not sched:
        print(f"  {seg_type:<5} -> no segment found")
        return
    _, _, start, end = sched[0]
    if seg_type in SEGMENT_EXCLUSIONS:
        status = f"EXCLUDED — {SEGMENT_EXCLUSIONS[seg_type]}"
    elif not SEGMENTS_INSPECTED.get(seg_type, False):
        status = "NOT INSPECTED — run its Cell 3 first"
    else:
        status = "included"
    print(f"  {seg_type:<5} -> {status}  ({(end - start) / 60:.1f} min)")


def print_manual_edits_summary(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
) -> pd.DataFrame:
    """Cell 3e: channels marked by hand and BAD time per segment."""
    print("=" * 70)
    print(f"  Subject {subj_id} -- Summary of manual edits")
    print("=" * 70)
    rows: List[Dict[str, Any]] = []
    total_bad = total_dur = 0.0
    for seg_type in SEG_TYPE_ORDER:
        for seg_name, seg_idx, start, end in get_segments_by_type(
            segments, seg_type, apply_time_rules=True
        ):
            seg_dur = end - start
            bad_time = _merged_bad_seconds(raw, start, end)
            n_marks = sum(
                1 for a in raw.annotations
                if str(a["description"]).upper().startswith("BAD")
                and a["onset"] < end and a["onset"] + a["duration"] > start
            )
            used = segment_run_used_in_preproc(seg_type, seg_idx)
            if seg_type in SEGMENT_EXCLUSIONS:
                status = f"EXCLUDED ({SEGMENT_EXCLUSIONS[seg_type]})"
            elif not SEGMENTS_INSPECTED.get(seg_type, False):
                status = "NOT INSPECTED"
            else:
                status = "included"
            rows.append({
                "Segment": seg_name,
                "Duration (s)": round(seg_dur, 1),
                "BAD (s)": round(bad_time, 1),
                "BAD (%)": round(bad_time / seg_dur * 100, 1) if seg_dur > 0 else 0.0,
                "# marks": n_marks,
                "Bad channels (by hand)": ", ".join(MANUAL_BAD_CHANNELS.get(seg_type, [])) or "-",
                "Status": status,
            })
            if used:
                total_bad += bad_time
                total_dur += seg_dur
    df = pd.DataFrame(rows)
    print(f"\n{df.to_string(index=False)}")
    pct = (total_bad / total_dur * 100) if total_dur > 0 else 0.0
    print(f"\n  Included segments: {total_bad:.1f} s BAD out of {total_dur:.1f} s ({pct:.1f}%)")
    not_insp = [r["Segment"] for r in rows if r["Status"] == "NOT INSPECTED"]
    if not_insp:
        print(f"\n  !! Not inspected yet: {', '.join(not_insp)} — Cell 4 will SKIP "
              "them. Run their Cell 3 first.")
    print("\n>>> Proceed to Cell 4 for preprocessing.\n")
    return df


def get_segments_by_type(
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
    seg_type: str,
    apply_time_rules: bool = False,
) -> List[Tuple[str, int, float, float]]:
    """(seg_name, seg_idx_1based, start, end) for one segment type.

    seg_type : "LB", "IGT" or "IGT2"
    apply_time_rules : if True, keep the last 300 s of LB (IGT/IGT2 whole)
    """
    expected_n = _EXPECTED_N_BY_TYPE.get(seg_type, 1)
    result: List[Tuple[str, int, float, float]] = []
    for idx, (start, end) in enumerate(segments.get(seg_type, [])):
        if end is None or end <= start:
            continue
        if apply_time_rules:
            start, end = apply_segment_rules(seg_type, start, end)
        seg_name = f"{seg_type} {idx + 1}" if expected_n > 1 else seg_type
        result.append((seg_name, idx + 1, start, end))
    return result


def build_preprocessing_units(
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
) -> List[Dict[str, Any]]:
    """One processing unit per segment (LB, IGT, IGT2) that was inspected in
    Cell 3 and not excluded in Cell 3d."""
    units: List[Dict[str, Any]] = []
    for seg_type in PREPROCESS_SEG_TYPES:
        for seg_name, seg_idx, start, end in get_segments_by_type(
            segments, seg_type, apply_time_rules=True
        ):
            if not segment_run_used_in_preproc(seg_type, seg_idx):
                continue
            units.append({
                "label": seg_name,
                "display_name": seg_name,
                "segments": [{"name": seg_name, "idx": seg_idx,
                              "start": start, "end": end}],
            })
    return units


def crop_and_concat_segments(
    raw: mne.io.Raw,
    segments: List[Dict[str, Any]],
) -> Tuple[mne.io.Raw, List[int]]:
    """Crop one or more segments and concatenate them into a single Raw."""
    parts: List[mne.io.Raw] = []
    lengths: List[int] = []
    max_t = float(raw.times[-1])
    for seg in segments:
        tmin = max(0.0, float(seg["start"]))
        tmax = min(float(seg["end"]), max_t)
        if tmax <= tmin:
            raise ValueError(
                f"Invalid crop for {seg.get('name', seg)}: "
                f"tmin={tmin:.2f}, tmax={tmax:.2f}, recording ends at {max_t:.2f} s"
            )
        part = raw.copy().crop(tmin=tmin, tmax=tmax)
        parts.append(part)
        lengths.append(part.n_times)
    if not parts:
        raise ValueError("No valid segments to crop.")
    if len(parts) == 1:
        return parts[0], lengths
    merged = mne.concatenate_raws(parts, preload=True, verbose=False)
    return merged, lengths


def shift_annotations(
    annotations: mne.Annotations,
    onset_offset: float,
    orig_time: Optional[datetime],
) -> mne.Annotations:
    """Shift annotations by a fixed number of seconds."""
    if len(annotations) == 0:
        return mne.Annotations([], [], [], orig_time=orig_time)
    return mne.Annotations(
        onset=[float(onset + onset_offset) for onset in annotations.onset],
        duration=[float(duration) for duration in annotations.duration],
        description=[str(desc) for desc in annotations.description],
        orig_time=orig_time,
    )


def safe_set_annotations(raw: mne.io.Raw, annotations: mne.Annotations) -> None:
    """Apply annotations while hiding benign crop-range RuntimeWarnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw.set_annotations(annotations)


def save_bad_channel_report(
    subj_id: Union[int, str],
    all_bads: List[str],
    segment_qc: Dict[str, Dict[str, Any]],
) -> None:
    """Persist union bad channels plus per-unit details for downstream QC."""
    payload = {
        "bad_channels": all_bads,
        "by_unit": {
            label: {
                "display_name": info["display_name"],
                "segments": list(info["segment_names"]),
                "bad_channels": list(info["bads"]),
                "bad_channel_diagnostics": info.get("bad_channel_diagnostics", {}),
                "artifact_seconds": float(info["artifact_seconds"]),
                "excluded_from_preprocessing": bool(info.get("excluded_from_preprocessing", False)),
                "exclusion_reason": str(info.get("exclusion_reason", "")),
            }
            for label, info in segment_qc.items()
        },
    }
    (subject_qc_dir(subj_id) / "bad_channels.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )


def write_unit_data_back(
    raw_target: mne.io.Raw,
    cleaned_unit: mne.io.Raw,
    segments: List[Dict[str, Any]],
    segment_lengths: List[int],
) -> None:
    """Write concatenated unit data back into the corresponding time spans."""
    sfreq = raw_target.info["sfreq"]
    src_start = 0
    for seg, seg_len in zip(segments, segment_lengths):
        tgt_start = int(round(seg["start"] * sfreq))
        tgt_stop = tgt_start + seg_len
        src_stop = src_start + seg_len
        raw_target._data[:, tgt_start:tgt_stop] = cleaned_unit.get_data(start=src_start, stop=src_stop)
        src_start = src_stop


# ---------------------------------------------------------------------------
# 1.12b  SAVE FINAL CLEANED DATA
# ---------------------------------------------------------------------------

CLEAN_DATA_DIR: Path = OUTPUT_DIR / "clean_data"


def save_clean_data(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    overwrite: bool = True,
) -> Path:
    """
    Save the final cleaned (post-ICA) Raw object to a dedicated folder.

    This allows reusing the cleaned data for future analyses (e.g., connectivity,
    time-frequency) without re-running the preprocessing pipeline.

    Returns the path to the saved .fif file.
    """
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLEAN_DATA_DIR / f"subject_{subj_id}_clean_raw.fif"
    raw.save(str(out_path), overwrite=overwrite, verbose=False)
    logger.info(f"S{subj_id}: Saved cleaned data -> {out_path}")
    return out_path


def load_clean_data(subj_id: Union[int, str]) -> Optional[mne.io.Raw]:
    """
    Load previously saved cleaned data for a subject.

    Returns None if the file does not exist.
    """
    fif_path = CLEAN_DATA_DIR / f"subject_{subj_id}_clean_raw.fif"
    if not fif_path.exists():
        logger.warning(f"S{subj_id}: Clean data not found at {fif_path}")
        return None
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    logger.info(f"S{subj_id}: Loaded cleaned data from {fif_path}")
    return raw


# ---------------------------------------------------------------------------
# 1.12c  MANUAL BAD SECTION MARKING
# ---------------------------------------------------------------------------

def mark_bad_sections(
    raw: mne.io.Raw,
    bad_intervals: List[Tuple[float, float]],
    description: str = "BAD_manual",
) -> None:
    """
    Mark time intervals as BAD in the Raw object's annotations.

    Parameters
    ----------
    raw : mne.io.Raw
        The Raw object to modify (in-place).
    bad_intervals : list of (start, end) tuples
        Time intervals in seconds to mark as BAD.
        Example: [(100, 130), (200, 210)] marks 100-130s and 200-210s as BAD.
    description : str
        Annotation description (default "BAD_manual").

    Example usage in a cell:
        mark_bad_sections(RAW, [(100, 130), (200, 210)])
    """
    if not bad_intervals:
        return

    onsets = []
    durations = []
    for start, end in bad_intervals:
        if end <= start:
            print(f"  Warning: Invalid interval ({start}, {end}) - skipped")
            continue
        onsets.append(float(start))
        durations.append(float(end - start))

    if onsets:
        new_annot = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=[description] * len(onsets),
            orig_time=raw.annotations.orig_time,
        )
        raw.set_annotations(raw.annotations + new_annot)
        print(f"  Marked {len(onsets)} BAD sections:")
        for s, d in zip(onsets, durations):
            print(f"    - {s:.1f}s to {s+d:.1f}s ({d:.1f}s)")


def show_bad_annotations(
    raw: mne.io.Raw,
    seg_start: Optional[float] = None,
    seg_end: Optional[float] = None,
    label: str = "",
) -> List[Dict[str, Any]]:
    """Show BAD annotations filtered to a segment window, with an optional label header."""
    if raw is None:
        print("!! Raw object is None.")
        return []

    bad_anns = []
    for ann in raw.annotations:
        if not ann["description"].upper().startswith("BAD"):
            continue
        onset    = float(ann["onset"])
        duration = float(ann["duration"])
        end      = onset + duration
        if seg_start is not None and seg_end is not None:
            if end < seg_start or onset > seg_end:
                continue
        bad_anns.append({"onset": onset, "duration": duration,
                         "end": end, "description": str(ann["description"])})

    if label:
        print(f"  BAD annotations for {label}:")
    if bad_anns:
        for ann in bad_anns:
            print(f"  - {ann['onset']:.1f}s – {ann['end']:.1f}s  "
                  f"({ann['duration']:.1f}s)  [{ann['description']}]")
    else:
        print("  No BAD annotations.")
    
    return bad_anns


def inspect_and_mark_segment(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    seg_type: str,
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
    *,
    tag: str = "raw",
    interactive: bool = True,
    apply_time_rules: bool = False,
    segment_qc: Optional[Dict[str, Dict[str, Any]]] = None,
    only_preprocessed: bool = False,
) -> None:
    """
    Inspect a segment type and allow marking BAD sections in the interactive viewer.
    
    This is the main function for segment inspection. It:
    1. Shows an overview plot
    2. Opens the interactive viewer (if interactive=True)
    3. Any BAD annotations made in the viewer are automatically applied to raw
    4. Shows a summary of BAD annotations after closing the viewer
    
    Parameters
    ----------
    raw : mne.io.Raw
        The Raw object to inspect and modify.
    subj_id : subject identifier
    seg_type : "LB", "IGT" or "IGT2"
    segments : dict of segment intervals
    tag : "raw" or "clean"
    interactive : if True, open the interactive viewer
    apply_time_rules : if True, keep the last 300 s of LB (IGT/IGT2 whole)
    """
    if raw is None:
        print(f"!! Raw object is None.")
        return
    
    qc_dir = subject_qc_dir(subj_id)
    segs = get_segments_by_type(segments, seg_type, apply_time_rules=apply_time_rules)
    
    if not segs:
        print(f"  No {seg_type} segments found.")
        return

    if only_preprocessed:
        filtered_segs: List[Tuple[str, int, float, float]] = []
        skipped_names: List[str] = []
        for seg_name, seg_idx, start, end in segs:
            if not segment_run_used_in_preproc(seg_type, seg_idx):
                skipped_names.append(seg_name)
                continue
            qc_info = (segment_qc or {}).get(seg_name, {})
            if bool(qc_info.get("excluded_from_preprocessing", False)):
                skipped_names.append(seg_name)
                continue
            filtered_segs.append((seg_name, seg_idx, start, end))
        segs = filtered_segs
        if not segs:
            skipped_txt = ", ".join(skipped_names) if skipped_names else seg_type
            print(
                f"  {seg_type}: clean inspection skipped (all runs omitted/excluded): "
                f"{skipped_txt}"
            )
            return
    
    max_sec = ""
    if seg_type == "LB" and apply_time_rules:
        max_sec = f"  [last {LB_MAX_SEC:.0f} s of the marked block]"
    
    print("=" * 70)
    print(f"  Subject {subj_id} -- {tag.upper()}: {seg_type}{max_sec}")
    print("=" * 70)

    _configure_matplotlib_for_popup()
    
    for seg_name, seg_idx, start, end in segs:
        print(f"\n--- {seg_name} ---  ({end - start:.1f} s)  [absolute: {start:.1f}s - {end:.1f}s]")
        plot_segment_overview(raw, subj_id, f"{tag}_{seg_name}", start, end,
                             save_dir=qc_dir if SAVE_QC_FILES else None)
        
        if interactive:
            launch_interactive_viewer(raw, subj_id, seg_name, start, end, tag=tag)
    
    # Show BAD annotations summary — one header per segment
    print()
    for seg_name, seg_idx, start, end in segs:
        show_bad_annotations(raw, start, end, label=seg_name)

    print(f"\n>>> {seg_type} inspection done.")


def print_preprocessing_summary(
    raw_clean: mne.io.Raw,
    subj_id: Union[int, str],
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
    bad_channels: List[str],
    ica_info: Dict[str, Any],
    segment_qc: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Print and return a summary of preprocessing results.
    
    Returns a DataFrame with segment-level summary.
    """
    print("=" * 70)
    print(f"  Subject {subj_id} -- PREPROCESSING SUMMARY")
    print("=" * 70)

    summary_rows = []
    n_eeg_channels = int(len(mne.pick_types(raw_clean.info, eeg=True, eog=False, exclude=[])))
    for seg_type in SEG_TYPE_ORDER:
        segs = get_segments_by_type(segments, seg_type)
        for seg_name, seg_idx, start, end in segs:
            orig_dur = end - start
            trunc_start, trunc_end = apply_segment_rules(seg_type, start, end)
            final_dur = trunc_end - trunc_start
            qc_key = seg_name
            qc_map = (segment_qc or {})
            qc_info = qc_map.get(qc_key, {})
            # Report whether this segment was actually processed in the current run.
            # If segment_qc is available, it is the source of truth (robust to later
            # changes to the Cell 3d exclusions after Cell 4 finished).
            if segment_qc is not None:
                use_pre = (qc_key in qc_map) and not bool(qc_info.get("excluded_from_preprocessing", False))
            else:
                use_pre = segment_run_used_in_preproc(seg_type, seg_idx)
            if not use_pre:
                continue
            qc_bads_list = list(qc_info.get("bads", []))
            qc_bads = ", ".join(qc_bads_list) if qc_bads_list else "-"

            # BAD time in this segment, overlapping marks counted ONCE. (A manual
            # mark and an automatic one on the same stretch used to be summed.)
            bad_time = _merged_bad_seconds(raw_clean, trunc_start, trunc_end)
            manual_bad_time = _merged_bad_seconds(raw_clean, trunc_start, trunc_end, kind="manual")

            bad_ratio = (bad_time / final_dur) if final_dur > 0 else 1.0
            interp_ratio = (len(qc_bads_list) / n_eeg_channels) if n_eeg_channels > 0 else 1.0
            exclude_bad_ratio = bool(bad_ratio > ANALYSIS_MAX_BAD_TIME_RATIO)
            exclude_interp_ratio = bool(interp_ratio > ANALYSIS_MAX_INTERP_RATIO)
            include_prelim = bool(use_pre) and not (exclude_bad_ratio or exclude_interp_ratio)

            summary_rows.append({
                "Segment": seg_name,
                "Original (s)": round(orig_dur, 1),
                "Final (s)": round(final_dur, 1),
                "BAD time (s)": round(bad_time, 1),
                "Manual BAD (s)": round(manual_bad_time, 1),
                "Usable (s)": round(max(0, final_dur - bad_time), 1),
                "BAD ratio": round(bad_ratio, 3),
                "Bad channels": qc_bads,
                "N bad channels": int(len(qc_bads_list)),
                "Bad channels (by hand)": ", ".join(qc_info.get("bads_manual", [])) or "-",
                "Interp ratio": round(interp_ratio, 3),
                "Exclude bad_ratio": exclude_bad_ratio,
                "Exclude interp_ratio": exclude_interp_ratio,
                "Include (prelim)": include_prelim,
                "Inspected": "Yes" if SEGMENTS_INSPECTED.get(seg_type, False) else "No",
                "Preprocess": "yes" if use_pre else "no",
            })

    df_summary = pd.DataFrame(summary_rows)
    if df_summary.empty:
        print("\n  No segments were preprocessed.")
        return df_summary

    print("\n  Segments preprocessed:")
    for _, row in df_summary.iterrows():
        seg = str(row["Segment"])
        final_s = float(row["Final (s)"])
        usable_s = float(row["Usable (s)"])
        bad_ratio = float(row["BAD ratio"])
        interp_ratio = float(row["Interp ratio"])
        bad_ch = str(row["Bad channels"]) if str(row["Bad channels"]).strip() else "-"
        print(
            f"    - {seg}: final={final_s:.1f}s, usable={usable_s:.1f}s, "
            f"bad_ratio={bad_ratio:.3f}, interp_ratio={interp_ratio:.3f}, "
            f"bad_channels={bad_ch}"
        )

    print("\n  Bad channels by segment:")
    for _, row in df_summary.iterrows():
        seg = str(row["Segment"])
        bad_ch = str(row["Bad channels"]) if str(row["Bad channels"]).strip() else "-"
        print(f"    - {seg}: {bad_ch}")

    print(f"\n  Bad channels union (auto + manual): {bad_channels}")
    print(
        "  ICA components excluded (global for this subject): "
        f"{ica_info.get('excluded', [])}"
    )

    return df_summary


def _prelim_decisions(df_summary: pd.DataFrame) -> pd.DataFrame:
    """Add the preliminary include/exclude columns to the summary table."""
    df_dec = df_summary.copy()
    df_dec["Segment"] = df_dec["Segment"].astype(str)
    df_dec["Exclude retention"] = "pending_spectral_qc"
    df_dec["Exclude any (prelim)"] = (
        df_dec["Exclude bad_ratio"].astype(bool) | df_dec["Exclude interp_ratio"].astype(bool)
    )
    df_dec["Include main (prelim)"] = (
        (df_dec["Preprocess"].astype(str).str.lower() == "yes")
        & (~df_dec["Exclude any (prelim)"])
    )
    return df_dec


def save_analysis_qc_decisions(
    subj_id: Union[int, str],
    df_summary: pd.DataFrame,
) -> None:
    """Persist the preliminary include/exclude decisions for the analysis."""
    if df_summary.empty:
        return
    df_dec = _prelim_decisions(df_summary)
    avail = {s: bool(((df_dec["Segment"] == s) & df_dec["Include main (prelim)"]).any())
             for s in PREPROCESS_SEG_TYPES}
    payload = {
        "subject": subj_id,
        "group": subject_condition(subj_id)["group"],
        "rules": {
            "max_bad_time_ratio": ANALYSIS_MAX_BAD_TIME_RATIO,
            "max_interp_ratio": ANALYSIS_MAX_INTERP_RATIO,
            "min_retention_pct": "pending_spectral_qc",
        },
        "segment_available_prelim": avail,
        # The planned contrast is the IGT -> IGT2 change between groups, with
        # LB as the randomisation check, so a subject needs all three.
        "include_subject_prelim": all(avail.values()),
        "note": "Final inclusion also requires the retention rule of the spectral analysis",
    }
    qdir = subject_qc_dir(subj_id)
    (qdir / "analysis_qc_segments.csv").write_text(df_dec.to_csv(index=False), encoding="utf-8")
    (qdir / "analysis_qc_subject.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_prelim_qc_overview(
    subj_id: Union[int, str],
    df_summary: pd.DataFrame,
) -> None:
    """Print the preliminary include/exclude decisions after preprocessing."""
    if df_summary.empty:
        return
    df_dec = _prelim_decisions(df_summary)
    print("-" * 70)
    print(f"  Subject {subj_id} -- PRELIM QC DECISIONS")
    print("-" * 70)
    print(f"  Rules: bad_ratio <= {ANALYSIS_MAX_BAD_TIME_RATIO:.2f}, "
          f"interp_ratio <= {ANALYSIS_MAX_INTERP_RATIO:.2f}")
    for _, row in df_dec.iterrows():
        if bool(row["Include main (prelim)"]):
            print(f"    {row['Segment']:<5} included")
            continue
        reasons = [name for flag, name in (
            (row.get("Exclude bad_ratio", False), f"bad_ratio {float(row['BAD ratio']):.2f}"),
            (row.get("Exclude interp_ratio", False), f"interp_ratio {float(row['Interp ratio']):.2f}"),
            (str(row.get("Preprocess", "yes")).lower() != "yes", "not preprocessed"),
        ) if bool(flag)]
        print(f"    {row['Segment']:<5} FAILS: {', '.join(reasons) or 'other'}")
    missing = [s for s in PREPROCESS_SEG_TYPES if s not in set(df_dec["Segment"])]
    for s in missing:
        why = SEGMENT_EXCLUSIONS.get(s, "not preprocessed")
        print(f"    {s:<5} absent: {why}")
    avail = [s for s in PREPROCESS_SEG_TYPES
             if ((df_dec["Segment"] == s) & df_dec["Include main (prelim)"]).any()]
    ok = len(avail) == len(PREPROCESS_SEG_TYPES)
    print(f"  Subject usable for LB + IGT -> IGT2 (prelim): {ok}  "
          f"({len(avail)}/{len(PREPROCESS_SEG_TYPES)} segments)")
    print("  NOTE: the >=50% clean-window retention rule is applied in the spectral analysis.\n")


def save_and_report_clean(
    raw_clean: mne.io.Raw,
    subj_id: Union[int, str],
    segments: Dict[str, List[Tuple[float, Optional[float]]]],
    bad_channels: List[str],
    ica_info: Dict[str, Any],
    segment_qc: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """
    Print summary, save cleaned data, and save summary CSV.
    """
    if raw_clean is None:
        print("!! Cleaned data not available. Cannot save.")
        return

    df_summary = print_preprocessing_summary(
        raw_clean, subj_id, segments, bad_channels, ica_info, segment_qc
    )
    print_prelim_qc_overview(subj_id, df_summary)

    # Save cleaned data
    print("\n" + "-" * 70)
    clean_path = save_clean_data(raw_clean, subj_id)
    print(f"  Saved -> {clean_path}")
    print(f"  Channels: {len(raw_clean.ch_names)}")
    print(f"  Duration: {raw_clean.times[-1]:.1f} s")
    print(f"  Sfreq:    {raw_clean.info['sfreq']:.0f} Hz\n")

    # Save summary to QC folder
    if SAVE_QC_FILES:
        summary_path = subject_qc_dir(subj_id) / "preprocessing_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"  Summary saved -> {summary_path}\n")
        save_analysis_qc_decisions(subj_id, df_summary)
        print(f"  Analysis QC saved -> {subject_qc_dir(subj_id) / 'analysis_qc_segments.csv'}")
        print(f"  Subject QC saved  -> {subject_qc_dir(subj_id) / 'analysis_qc_subject.json'}\n")
    print(">>> Preprocessing finished.\n")


def plot_segment_overview(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    seg_name: str,
    start: float,
    end: float,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Static inline overview for one segment:
      - Top panel: butterfly plot (all EEG channels overlaid)
      - Bottom panel: PSD (Welch) per channel

    The figure is displayed inline and optionally saved to disk.
    """
    sfreq = raw.info["sfreq"]
    i0, i1 = int(start * sfreq), min(int(end * sfreq), raw.n_times)
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False)
    data = raw.get_data(picks=eeg_picks, start=i0, stop=i1) * 1e6  # -> uV
    times = np.arange(data.shape[1]) / sfreq

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(f"Subject {subj_id}  |  {seg_name}  "
                 f"({end - start:.1f} s)", fontsize=12)

    # -- Top: butterfly plot --
    ax = axes[0]
    for ch_i in range(data.shape[0]):
        ax.plot(times, data[ch_i], linewidth=0.3, alpha=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (uV)")
    ax.set_title("Butterfly plot (all EEG channels)")

    # -- Bottom: PSD --
    ax = axes[1]
    n_fft = min(INSPECT_PSD_N_FFT, data.shape[1])
    n_overlap = min(INSPECT_PSD_N_OVERLAP, n_fft - 1)
    try:
        psd, freqs = mne.time_frequency.psd_array_welch(
            data * 1e-6,  # back to V for MNE
            sfreq=sfreq, fmin=INSPECT_PSD_FMIN, fmax=INSPECT_PSD_FMAX,
            n_fft=n_fft, n_overlap=n_overlap, verbose=False,
        )
        psd_uv2 = psd * 1e12  # V^2/Hz -> uV^2/Hz
        for ch_i in range(psd_uv2.shape[0]):
            ax.semilogy(freqs, psd_uv2[ch_i], linewidth=0.4, alpha=0.6)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (uV^2/Hz)")
        ax.set_title("Power Spectral Density (Welch)")
    except Exception:
        ax.text(0.5, 0.5, "PSD computation failed (segment too short?)",
                transform=ax.transAxes, ha="center")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = seg_name.replace(" ", "_")
        fig.savefig(str(save_dir / f"overview_{safe_name}.png"), dpi=120)

    # Show in a GUI window when possible (VS Code interactive defaults to inline).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.show(block=True)

    plt.close(fig)


def _run_viewer_subprocess(cmd: List[str]) -> int:
    """Run a viewer subprocess; relay only [Viewer]/[ICA viewer]/[!] tagged lines."""
    import platform
    import threading
    import time

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

    def _relay(stream: Any) -> None:
        for line in iter(stream.readline, ""):
            stripped = line.rstrip()
            if any(tag in stripped for tag in ("[Viewer]", "[ICA viewer]", "[!]")):
                print(stripped)

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True,
    )
    t = threading.Thread(target=_relay, args=(proc.stdout,), daemon=True)
    t.start()

    if platform.system() == "Darwin":
        time.sleep(2.0)
        subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to set frontmost of true '
             'of (first process whose name contains "Python")'],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    rc = int(proc.wait() or 0)
    t.join(timeout=2.0)
    return rc


def launch_interactive_viewer(
    raw: mne.io.Raw,
    subj_id: Union[int, str],
    seg_label: str,
    start: float,
    end: float,
    *,
    tag: str = "raw",
    apply_edits: bool = True,
) -> Dict[str, Any]:
    """
    Launch the interactive EEG segment viewer in a **separate process**.

    This saves a temporary .fif file and calls ``eeg_segment_viewer.py``
    (which uses the TkAgg backend) via :func:`subprocess.run`, so the
    interactive MNE browser window pops up even when the main script runs
    inside an IPython / VSCode cell.

    The call is **blocking** — the cell waits until you close the viewer
    window, then reads back any edits (bad channels, BAD annotations) and
    optionally applies them to the original ``raw`` object.

    Parameters
    ----------
    raw         : loaded Raw object (not cropped — cropping is done in the viewer)
    subj_id     : subject identifier (for the window title)
    seg_label   : segment label, e.g. "LB", "IGT2"
    start       : segment start in seconds
    end         : segment end in seconds
    tag         : "raw" or "clean" (used in the temp filename)
    apply_edits : if True, bad channels and annotations are applied to ``raw``

    Returns
    -------
    dict with keys:
        bad_channels_added : list of channel names marked bad
        annotations_added  : list of dicts with onset/duration/description
    """
    viewer_script = BASE_DIR / "scripts" / "eeg_segment_viewer.py"
    if not viewer_script.exists():
        print(f"  [!] Viewer script not found: {viewer_script}")
        return {"bad_channels_added": [], "annotations_added": []}

    # --- Save a temporary .fif so the viewer can load it -----------------
    # Only the inspected window is written, not the whole recording: the viewer
    # crops to [start, end] anyway, so saving the full raw cost a few hundred MB
    # per segment. Annotation onsets and first_time keep the recording's
    # absolute time frame across crop/save/reload, so the edits read back below
    # stay in the same reference as `raw`.
    tmp_dir = OUTPUT_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    safe_label = seg_label.replace(" ", "_")
    tmp_fif = tmp_dir / f"S{subj_id}_{tag}_{safe_label}_eeg.fif"

    t_offset = raw.first_time
    t_last = t_offset + raw.times[-1]
    crop_start = min(max(float(start), t_offset), t_last)
    crop_end = min(max(float(end), crop_start), t_last)
    seg_key = seg_label
    _seg_copy = raw.copy().crop(tmin=crop_start - t_offset, tmax=crop_end - t_offset)
    if tag == "raw" and not APPLY_MANUAL_BAD_CHANNELS_GLOBALLY:
        # Show the channels already marked by hand for this segment, so they
        # can be reviewed and un-marked by clicking again.
        _pre = [ch for ch in MANUAL_BAD_CHANNELS.get(seg_key, []) if ch in _seg_copy.ch_names]
        _seg_copy.info["bads"] = sorted(set(_seg_copy.info["bads"]) | set(_pre))
    _seg_copy.save(str(tmp_fif), overwrite=True, verbose=False)

    # Output JSON for edits
    edits_json = tmp_dir / f"S{subj_id}_{tag}_{safe_label}_edits.json"

    scale_uv = RAW_INTERACTIVE_SCALE_UV if tag == "raw" else CLEAN_INTERACTIVE_SCALE_UV
    title = f"S{subj_id} | {seg_label} ({tag})  [{start:.1f} – {end:.1f} s]"
    cmd = [
        _project_python(),
        str(viewer_script),
        str(tmp_fif),
        str(start),
        str(end),
        "--title", title,
        "--output", str(edits_json),
        "--scale-uv", str(scale_uv),
    ]
    print(f"  >> Opening viewer for {seg_label} …")
    try:
        _run_viewer_subprocess(cmd)
    except Exception as exc:
        print(f"  [!] Error launching viewer: {exc}")
        return {"bad_channels_added": [], "annotations_added": []}
    finally:
        # The viewer subprocess has finished with the temp .fif by now and the
        # edits come back through edits_json, so nothing reads it again. Drop it
        # (and any split parts) rather than letting it accumulate on disk.
        for part in tmp_dir.glob(f"{tmp_fif.stem}*.fif"):
            try:
                part.unlink()
            except OSError as exc:
                print(f"  [!] Could not remove temp file {part.name}: {exc}")

    # --- Read back edits -------------------------------------------------
    edits = {"bad_channels_added": [], "annotations_added": []}
    if edits_json.exists():
        try:
            edits = json.loads(edits_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    # --- Apply edits to the original raw object --------------------------
    if apply_edits:
        # Bad channels. Kept PER SEGMENT and added to that segment's automatic
        # detection in Step 4. Only marks on the RAW data count: the clean data
        # is already interpolated, so a channel marked there cannot be re-cleaned.
        added = list(edits.get("bad_channels_added", []))
        if APPLY_MANUAL_BAD_CHANNELS_GLOBALLY:
            for ch in added:
                if ch not in raw.info["bads"]:
                    raw.info["bads"].append(ch)
        elif tag == "raw":
            final = edits.get("bad_channels_final")
            if final is None:  # older viewer: only additions are reported
                final = sorted(set(MANUAL_BAD_CHANNELS.get(seg_key, [])) | set(added))
            base = set(raw.info["bads"])
            before = MANUAL_BAD_CHANNELS.get(seg_key, [])
            MANUAL_BAD_CHANNELS[seg_key] = sorted(ch for ch in final if ch not in base)
            if MANUAL_BAD_CHANNELS[seg_key] or before:
                print(f"  [i] {seg_key}: channels marked bad by hand -> "
                      f"{MANUAL_BAD_CHANNELS[seg_key] or 'none'} (applied in Cell 4, Step 4)")
        elif added:
            print("  [i] Channels marked in the CLEAN viewer are not applied: the data "
                  "is already interpolated. Mark bad channels in Cell 3 and re-run Cell 4.")

        # BAD annotations: use annotations_final (full replacement) if available,
        # otherwise fall back to annotations_added (append-only, legacy behaviour).
        seg_tmin = edits.get("segment_tmin", start)
        seg_tmax = edits.get("segment_tmax", end)
        annotations_final = edits.get("annotations_final")

        if annotations_final is not None:
            # Remove all existing BAD annotations that overlap this segment
            keep = []
            for ann in raw.annotations:
                if ann["description"].upper().startswith("BAD"):
                    ann_start = float(ann["onset"])
                    ann_end = ann_start + float(ann["duration"])
                    if ann_end > seg_tmin and ann_start < seg_tmax:
                        continue
                keep.append(ann)

            new_annotations = mne.Annotations(
                onset=[float(a["onset"]) for a in keep],
                duration=[float(a["duration"]) for a in keep],
                description=[a["description"] for a in keep],
                orig_time=raw.annotations.orig_time,
            )

            # Add back BAD annotations as they are after the user edited them
            for ann in annotations_final:
                new_annotations.append(
                    onset=ann["onset"],
                    duration=ann["duration"],
                    description=ann["description"],
                )

            raw.set_annotations(new_annotations)
        else:
            # Legacy: only add new annotations
            for ann in edits.get("annotations_added", []):
                raw.annotations.append(
                    onset=ann["onset"],
                    duration=ann["duration"],
                    description=ann["description"],
                )

    return edits


# ---------------------------------------------------------------------------
# 1.13  WINDOWING
# ---------------------------------------------------------------------------

def apply_segment_rules(
    seg_type: str, start: float, end: Optional[float],
) -> Tuple[float, float]:
    """
    Segment-specific truncation.

    - LB        : keep the LAST 300 s of the marked block, so the window ends
                  at the closing marker (last 5 min of habituation).
    - IGT, IGT2 : self-paced, kept whole.
    """
    if end is None:
        return start, start
    if seg_type == "LB" and end - start > LB_MAX_SEC:
        start = end - LB_MAX_SEC
    return start, end


# ---------------------------------------------------------------------------
# 1.14  SUBJECTS AND CONDITION
# ---------------------------------------------------------------------------

def subject_condition(subj_id: Union[int, str]) -> Dict[str, Any]:
    """Condition from the last letter of the file name: C = control (cond 0),
    E = stress (cond 1). Verified against the master database's cond column
    with no disagreement."""
    m = re.match(r"^(\d+)([CE])$", str(subj_id).strip(), flags=re.IGNORECASE)
    if not m:
        return {"num": None, "letter": "", "cond": None, "group": "unknown"}
    letter = m.group(2).upper()
    return {"num": int(m.group(1)), "letter": letter,
            "cond": 0 if letter == "C" else 1,
            "group": "control" if letter == "C" else "stress"}


def header_id_mismatch(vhdr_path: Path) -> Optional[str]:
    """The ID a .vhdr header claims, when it differs from the file name."""
    try:
        text = Path(vhdr_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.strip().lower().startswith("datafile="):
            ref_stem = Path(line.split("=", 1)[1].strip()).stem
            if ref_stem and ref_stem != Path(vhdr_path).stem:
                return ref_stem
    return None


# ---------------------------------------------------------------------------
# 1.15  DOCUMENTED MARKER REPAIRS  (applied automatically in Cell 2)
# ---------------------------------------------------------------------------
# End times recovered from the BIOPAC record, which was cut into the same task
# phases. EEG and BIOPAC phase durations agree at r = 0.910 (IGT1, n = 108) and
# r = 0.917 (IGT2, n = 96): median difference 0 s, 90% within 6 s.
# {subject: {segment: (start_s, end_s)}}: start = surviving EEG marker,
# end = start + BIOPAC phase duration.
BIOPAC_REPAIRS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "24E": {"IGT2": (3272.52, 3509.50)},
    "29C": {"IGT": (1233.92, 1626.86)},
    "30E": {"IGT2": (2982.30, 3270.05)},
    "34C": {"IGT2": (2940.50, 3194.57)},
    "37C": {"IGT2": (2643.06, 2906.47)},
    "50C": {"IGT2": (2886.08, 3176.76)},
    "67E": {"IGT2": (3408.70, 3646.90)},
    "70C": {"IGT2": (2448.70, 2719.36)},
    "80C": {"IGT2": (2562.36, 2805.78)},
    "82C": {"IGT2": (2957.44, 3277.64)},
    "84C": {"IGT2": (2986.54, 3304.61)},
    "88C": {"IGT2": (3039.50, 3318.72)},
    "89E": {"IGT": (1745.96, 2032.90)},
    "999E": {"IGT2": (3285.76, 3560.34)},
}


def _stray_press_after(onsets: Dict[str, List[float]], orphan: float,
                       label: str = "TSST", lo: float = 210.0, hi: float = 560.0):
    """A press of `label` a plausible segment-length after `orphan`.

    Commonest failure in this dataset: the task ended and the marker was
    pressed, but the label had not been switched back from TSST. The press is
    real and correctly timed — only its name is wrong. Rejected if any other
    marker falls inside the proposed window (a segment cannot contain another
    task's boundary): without that guard the rule mis-fires when the orphan is
    the segment's END rather than its start (73C).
    """
    every = sorted(t for lab in onsets for t in onsets[lab])
    for t in sorted(onsets.get(label, [])):
        if not lo <= t - orphan <= hi:
            continue
        if any(orphan + 1.0 < u < t - 1.0 for u in every):
            continue
        return t
    return None


def apply_subject_patches(subj_id: Union[int, str],
                          segments: Dict[str, List[Tuple[float, Optional[float]]]],
                          onsets: Optional[Dict[str, List[float]]] = None,
                          ) -> Dict[str, List[Tuple[float, Optional[float]]]]:
    """Repair markers that cannot be paired automatically, printing each repair.

    Every repair is a documented decision, supported by one of:
      1. a press of the WRONG LABEL at the right time (timing real, name wrong);
      2. the BIOPAC record (BIOPAC_REPAIRS above);
      3. the "last 5 minutes" convention for LB, as used for the physiology.
    Where 1 and 2 both apply they agree within seconds (37C, 50C, 999E exactly;
    80C within 7 s; 73C on duration). An unpaired marker with no such evidence
    is NOT guessed: the segment stays unbuildable and Cell 2 says so.
    """
    sid = str(subj_id)
    onsets = onsets or {}

    def orphans(label):
        return [a for a, b in segments.get(label, []) if b is None]

    def pairs(label):
        return [(a, b) for a, b in segments.get(label, []) if b is not None]

    def presses(label):
        return sorted([a for a, _ in segments.get(label, [])]
                      + [b for _, b in segments.get(label, []) if b is not None])

    # 1. IGT/IGT2 closed by a TSST-labelled press (24E, 30E, 37C, 50C, 80C, 999E).
    #    TSST already has its 6 markers; this is a 7th, 237-291 s after the start.
    for seg in ("IGT2", "IGT"):
        if orphans(seg) and not pairs(seg):
            o = orphans(seg)[0]
            t = _stray_press_after(onsets, o)
            if t is not None:
                segments[seg] = [(o, t)]
                print(f"    [{sid}] {seg}: end taken from a press labelled TSST at "
                      f"{t / 60:.1f} min ({t - o:.0f} s after the start)")

    # 2. 73C: the only IGT press is the END. A TSST press 5.0 min earlier is the
    #    start; without it TSST has exactly 6 markers in three clean blocks, and
    #    BIOPAC gives IGT1 = 299 s against the 300 s this implies. Reading the
    #    IGT press as a start is impossible: a TSST press would fall inside.
    if sid == "73C" and not pairs("IGT"):
        o = orphans("IGT")
        pre = [t for t in onsets.get("TSST", []) if o and 210 <= o[0] - t <= 560]
        if o and pre:
            segments["IGT"] = [(pre[-1], o[0])]
            print(f"    [73C] IGT: {pre[-1] / 60:.1f} -> {o[0] / 60:.1f} min "
                  "(the IGT press was the end; the start was labelled TSST)")

    # 3. 52E: IGT2 recorded entirely under the TSST label. TSST has 8 markers:
    #    the first 6 are its three blocks, the trailing pair (4.2 min) is IGT2.
    if sid == "52E" and not pairs("IGT2"):
        t = sorted(onsets.get("TSST", []))
        if len(t) >= 8:
            segments["IGT2"] = [(t[6], t[7])]
            print(f"    [52E] IGT2: {t[6] / 60:.1f} -> {t[7] / 60:.1f} min "
                  "(both presses labelled TSST)")

    # 4. LB press failures.
    if sid == "42C":
        # LB pressed at 4.6, 5.1 and 9.5 min: the 5.1 is a second press 30 s
        # after the first, so default pairing gives a 30-second "baseline".
        lb = presses("LB")
        if len(lb) >= 3:
            segments["LB"] = [(lb[0], lb[-1])]
            print(f"    [42C] LB: {lb[0] / 60:.1f} -> {lb[-1] / 60:.1f} min "
                  "(duplicate press in between ignored)")
    if sid == "25E":
        # Both LB presses in the same second: the start was never marked.
        # Take the 5 minutes ending at the press.
        lb = presses("LB")
        if lb and lb[-1] - lb[0] < 60:
            end = lb[-1]
            segments["LB"] = [(max(0.0, end - LB_MAX_SEC), end)]
            print(f"    [25E] LB: the 5 min ending at the press "
                  f"({max(0.0, end - LB_MAX_SEC) / 60:.1f} -> {end / 60:.1f} min)")
    if sid == "64C":
        # A single LB press at 13.0 min, INTERO starts at 13.9: the press is
        # the END of the baseline.
        lb = orphans("LB") or [a for a, _ in segments.get("LB", [])]
        if lb and not pairs("LB"):
            end = lb[0]
            segments["LB"] = [(max(0.0, end - LB_MAX_SEC), end)]
            print(f"    [64C] LB: the 5 min ending at the single LB press "
                  f"({max(0.0, end - LB_MAX_SEC) / 60:.1f} -> {end / 60:.1f} min)")

    # 5. Ends recovered from BIOPAC (29C, 34C, 67E, 70C, ...). Applied last,
    #    so a repair found in the EEG itself always takes precedence.
    for seg, (t0, t1) in BIOPAC_REPAIRS.get(sid, {}).items():
        if not pairs(seg):
            segments[seg] = [(float(t0), float(t1))]
            print(f"    [{sid}] {seg}: end taken from the BIOPAC phase duration "
                  f"({t0 / 60:.1f} -> {t1 / 60:.1f} min, {t1 - t0:.0f} s)")
    return segments


# ---------------------------------------------------------------------------
# 1.16  MANUAL MARKS: BAD TIME, SAVE / RESTORE, PROGRESS
# ---------------------------------------------------------------------------

_AUTO_BAD_DESCRIPTIONS: Tuple[str, ...] = ("BAD_artifact", "BAD_muscle")


def _merged_bad_seconds(raw: mne.io.Raw, start: float, end: float,
                        kind: str = "all") -> float:
    """Seconds of [start, end] covered by BAD annotations, overlaps counted once.

    kind = "all", "manual" (anything except the automatic BAD_artifact /
    BAD_muscle) or "auto".
    """
    ivs: List[Tuple[float, float]] = []
    for a in raw.annotations:
        d = str(a["description"])
        if not d.upper().startswith("BAD"):
            continue
        is_auto = d in _AUTO_BAD_DESCRIPTIONS
        if (kind == "manual" and is_auto) or (kind == "auto" and not is_auto):
            continue
        s = max(float(a["onset"]), start)
        e = min(float(a["onset"]) + float(a["duration"]), end)
        if e > s:
            ivs.append((s, e))
    total = 0.0
    cur_s = cur_e = None
    for s, e in sorted(ivs):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def save_manual_marks(raw: mne.io.Raw, subj_id: Union[int, str],
                      segments: Dict[str, List[Tuple[float, Optional[float]]]],
                      stage: str = "raw") -> Path:
    """Write every human decision for this subject — BAD sections, channels
    marked by hand, excluded segments — to qc/subject_X/manual_marks*.json.

    An audit trail, and a way back after a crash (restore_manual_marks).
    stage="raw" after Cell 3, stage="clean" (final state) in Cell 6.
    """
    marks = [{"onset": round(float(a["onset"]), 3),
              "duration": round(float(a["duration"]), 3),
              "description": str(a["description"])}
             for a in raw.annotations
             if str(a["description"]).upper().startswith("BAD")
             and str(a["description"]) not in _AUTO_BAD_DESCRIPTIONS]
    payload = {
        "subject": str(subj_id),
        "stage": stage,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "bad_sections": marks,
        "bad_channels_by_segment": {k: list(v) for k, v in MANUAL_BAD_CHANNELS.items()},
        "excluded_segments": dict(SEGMENT_EXCLUSIONS),
        "segments_inspected": dict(SEGMENTS_INSPECTED),
        "segments_used": {
            s: [round(t0, 3), round(t1, 3)]
            for s in PREPROCESS_SEG_TYPES
            for _, _, t0, t1 in get_segments_by_type(segments, s, apply_time_rules=True)[:1]
        },
    }
    name = "manual_marks.json" if stage == "raw" else f"manual_marks_{stage}.json"
    path = subject_qc_dir(subj_id) / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def restore_manual_marks(raw: mne.io.Raw, subj_id: Union[int, str]) -> bool:
    """Re-apply the marks saved by Cell 3e (e.g. after a kernel restart).
    Run it right after Cell 2, instead of marking again."""
    path = subject_qc_dir(subj_id) / "manual_marks.json"
    if not path.exists():
        print(f"  No saved manual marks for {subj_id}.")
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    existing = {(round(float(a["onset"]), 3), round(float(a["duration"]), 3), str(a["description"]))
                for a in raw.annotations}
    n = 0
    for m in data.get("bad_sections", []):
        key = (round(m["onset"], 3), round(m["duration"], 3), m["description"])
        if key not in existing:
            raw.annotations.append(m["onset"], m["duration"], m["description"])
            n += 1
    MANUAL_BAD_CHANNELS.clear()
    MANUAL_BAD_CHANNELS.update({k: list(v) for k, v in data.get("bad_channels_by_segment", {}).items()})
    SEGMENTS_INSPECTED.update({k: bool(v) for k, v in data.get("segments_inspected", {}).items()})
    print(f"  Restored {n} BAD sections and bad channels {dict(MANUAL_BAD_CHANNELS) or 'none'} "
          f"from {path.name} (saved {data.get('saved_at', '?')}). "
          "Exclusions come from Cell 3d — run it again.")
    return True


def cleanup_intermediate_fif(subj_id: Union[int, str]) -> None:
    """Delete this subject's intermediate .fif files (called by Cell 6)."""
    d = FIF_DIR / f"subject_{subj_id}"
    if not d.exists():
        return
    freed = 0
    for f in d.glob("*.fif"):
        freed += f.stat().st_size
        f.unlink()
    if freed:
        print(f"  Removed intermediate files ({freed / 1e9:.2f} GB). Re-run Cell 4 to "
              "recreate them if you need the ICA viewer again.")


def log_progress(subj_id: Union[int, str]) -> Path:
    """Add or replace this subject's row in output_manual/progress_log.csv."""
    sid = str(subj_id)
    qdir = subject_qc_dir(sid)
    row: Dict[str, Any] = {
        "subject": sid,
        "group": subject_condition(sid)["group"],
        "processed_at": datetime.now().isoformat(timespec="minutes"),
        "pipeline_version": PIPELINE_VERSION,
    }
    summ = qdir / "preprocessing_summary.csv"
    df = pd.read_csv(summ) if summ.exists() else pd.DataFrame()
    for seg in PREPROCESS_SEG_TYPES:
        r = df[df["Segment"].astype(str) == seg] if not df.empty else df
        if r.empty:
            row[f"{seg}_status"] = (f"excluded: {SEGMENT_EXCLUSIONS[seg]}"
                                    if seg in SEGMENT_EXCLUSIONS else "not preprocessed")
            continue
        r = r.iloc[0]
        ok = str(r.get("Include (prelim)", "")).strip().lower() == "true"
        row[f"{seg}_status"] = "ok" if ok else "fails QC"
        row[f"{seg}_bad_ratio"] = float(r["BAD ratio"])
        row[f"{seg}_manual_bad_s"] = float(r.get("Manual BAD (s)", 0.0))
        row[f"{seg}_bad_channels"] = str(r["Bad channels"])
        row[f"{seg}_bad_channels_by_hand"] = str(r.get("Bad channels (by hand)", "-"))
    ica_p = qdir / "ica_exclude.json"
    if ica_p.exists():
        ica = json.loads(ica_p.read_text(encoding="utf-8"))
        row["ica_excluded"] = str(ica.get("exclude", []))
        row["ica_changed_by_hand"] = bool(ica.get("manual_override", False))
    subj_p = qdir / "analysis_qc_subject.json"
    if subj_p.exists():
        row["usable_prelim"] = bool(json.loads(subj_p.read_text(encoding="utf-8"))
                                    .get("include_subject_prelim", False))
    log_p = OUTPUT_DIR / "progress_log.csv"
    old = pd.read_csv(log_p, dtype={"subject": str}) if log_p.exists() else pd.DataFrame()
    if not old.empty:
        old = old[old["subject"].astype(str) != sid]
    new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    new["_k"] = new["subject"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
    new.sort_values("_k").drop(columns="_k").to_csv(log_p, index=False)
    return log_p


# ========================  END OF CELL 1  ==================================


#%%
# =============================================================================
# CELL 2 — LOAD RAW + MARKERS + SEGMENTS  (BREAKPOINT)
# =============================================================================
# Loads one subject and checks its markers. Nothing is preprocessed yet.
#   1. Set SUJETO to the file name without extension (e.g. "15C", "24E").
#   2. Run the cell and read its three reports:
#        MARKERS           every marker, in minutes, with the gap to the next
#        MARKER REPAIRS    documented repairs applied to this subject, if any
#        SEGMENT DURATION  LB, IGT and IGT2 exactly as they will be analysed
#   3. A WARNING that is not explained by a repair means: STOP and ask before
#      going on. A wrong segment boundary cannot be fixed by cleaning.
# Re-running this cell reloads the file and DISCARDS any marks from Cell 3.
# =============================================================================

# ============================================================================
# >>> EDIT HERE: subject = file name without extension (e.g. "15C", "24E")
# ============================================================================
SUJETO = "15C"
# ============================================================================

mne.set_log_level("WARNING")
setup_logging()
ensure_output_dirs()

EEG_FILE = subject_vhdr(SUJETO)
if not EEG_FILE.exists():
    raise FileNotFoundError(
        f"EEG file not found: {EEG_FILE}\n"
        f"SUJETO must be a file name without extension, like '15C' or '24E', "
        f"and the recordings must be in {INPUT_DIR}"
    )

print(f"Pipeline version {PIPELINE_VERSION}")
print(f"Loading subject {SUJETO}  ({subject_condition(SUJETO)['group']})  "
      f"from {EEG_FILE.name} ...")
RAW = step0_load_raw(SUJETO)
print(f"  {len(RAW.ch_names)} channels, {RAW.info['sfreq']:.0f} Hz, "
      f"{RAW.times[-1] / 60:.1f} min")

ann_report = list_raw_annotations(RAW, SUJETO)
(subject_qc_dir(SUJETO) / "raw_annotations.txt").write_text(ann_report, encoding="utf-8")

_label_onsets = parse_annotations(RAW)
SEGMENTS = reconstruct_segments(_label_onsets)

print("\n" + "=" * 84)
print(f"  MARKER REPAIRS  (SUBJECT {SUJETO})")
print("=" * 84)
_before = json.dumps({k: SEGMENTS.get(k) for k in PREPROCESS_SEG_TYPES})
SEGMENTS = apply_subject_patches(SUJETO, SEGMENTS, _label_onsets)
if json.dumps({k: SEGMENTS.get(k) for k in PREPROCESS_SEG_TYPES}) == _before:
    print("    none needed — the LB, IGT and IGT2 markers are complete")
print("=" * 84)
save_segments_json(SUJETO, SEGMENTS)

_seg_report = inspect_segments(SUJETO, SEGMENTS, RAW.info["sfreq"])
(subject_qc_dir(SUJETO) / "segment_inspection.txt").write_text(_seg_report, encoding="utf-8")

# Fresh state for this subject: nothing inspected, excluded or marked yet.
SEGMENTS_INSPECTED = {s: False for s in PREPROCESS_SEG_TYPES}
SEGMENT_EXCLUSIONS = {}
MANUAL_BAD_CHANNELS = {}

_missing = [s for s in PREPROCESS_SEG_TYPES if not get_segments_by_type(SEGMENTS, s)]
if _missing:
    print(f"\n!! {SUJETO}: cannot build {', '.join(_missing)}. Do NOT continue — "
          "write it in the subject list and ask.")
else:
    print("\n>>> Segments OK. Continue with Cell 3a.")
if (subject_qc_dir(SUJETO) / "manual_marks.json").exists():
    print(f">>> Marks from an earlier session exist for {SUJETO}. To reuse them "
          f"instead of marking again, run:  restore_manual_marks(RAW, SUJETO)")


#%%
# =============================================================================
# CELL 3a — INSPECT RAW: LB (baseline)
# =============================================================================
# 1. A static overview (butterfly + spectrum) opens first: close it to go on.
# 2. The interactive viewer opens (about 100 µV; + / - change the scale):
#      click a channel NAME -> mark / unmark it as bad for THIS segment
#      press 'a'            -> annotation mode: create the label "BAD" (once),
#                              then click-drag over the signal to mark it
#      right-click a mark   -> delete it
#      close the window     -> everything is saved automatically
# 3. Mark only what ICA cannot fix (see the guide). Do NOT mark blinks.
# =============================================================================
INTERACTIVE_VIEW = True   # False = re-run without opening the viewer

inspect_and_mark_segment(
    RAW, SUJETO, "LB", SEGMENTS,
    tag="raw", interactive=INTERACTIVE_VIEW, apply_time_rules=True,
)
SEGMENTS_INSPECTED["LB"] = True


#%%
# =============================================================================
# CELL 3b — INSPECT RAW: IGT (first Iowa Gambling Task)
# =============================================================================
INTERACTIVE_VIEW = globals().get("INTERACTIVE_VIEW", True)
inspect_and_mark_segment(
    RAW, SUJETO, "IGT", SEGMENTS,
    tag="raw", interactive=INTERACTIVE_VIEW, apply_time_rules=True,
)
SEGMENTS_INSPECTED["IGT"] = True


#%%
# =============================================================================
# CELL 3c — INSPECT RAW: IGT2 (second Iowa Gambling Task)
# =============================================================================
INTERACTIVE_VIEW = globals().get("INTERACTIVE_VIEW", True)
inspect_and_mark_segment(
    RAW, SUJETO, "IGT2", SEGMENTS,
    tag="raw", interactive=INTERACTIVE_VIEW, apply_time_rules=True,
)
SEGMENTS_INSPECTED["IGT2"] = True


#%%
# =============================================================================
# CELL 3d — SEGMENT EXCLUSIONS  (run it every time)
# =============================================================================
# Every inspected segment enters preprocessing by default. Exclude a segment
# ONLY if it is unusable as a whole (noise throughout, cap displaced,
# electrodes off) — never because it contains some BAD sections.
#
# To exclude, add the subject and the segment WITH THE REASON, e.g.
#       "24E": {"IGT2": "cap displaced after the first minute"},
# Decisions are stored per subject, so they never carry over to the next one,
# and this block is the permanent record of every exclusion.
# =============================================================================
EXCLUDE: Dict[str, Dict[str, str]] = {
    # "24E": {"IGT2": "reason"},
}

SEGMENT_EXCLUSIONS = dict(EXCLUDE.get(SUJETO, {}))
_unknown = [s for s in SEGMENT_EXCLUSIONS if s not in PREPROCESS_SEG_TYPES]
if _unknown:
    raise ValueError(f"Unknown segment name(s) in EXCLUDE[{SUJETO!r}]: {_unknown}. "
                     f"Use {list(PREPROCESS_SEG_TYPES)}.")
print(f"Subject {SUJETO} — segments for Cell 4:")
for _seg in PREPROCESS_SEG_TYPES:
    print_segment_preproc_status(_seg, SEGMENTS)


#%%
# =============================================================================
# CELL 3e — SUMMARY OF MANUAL EDITS  (also saved to manual_marks.json)
# =============================================================================
MANUAL_SUMMARY = print_manual_edits_summary(RAW, SUJETO, SEGMENTS)
print(f"  Manual marks saved -> {save_manual_marks(RAW, SUJETO, SEGMENTS)}")


#%%
# =============================================================================
# CELL 4 — PREPROCESSING  (Steps 1-8)
# =============================================================================
#   Step 1  Channel selection + types          Step 5  Artifact annotation
#   Step 2  Montage                            Step 6  ICA (fit on 1 Hz copy)
#   Step 3  Two filtered copies (1 / 0.1 Hz)   Step 7  Interpolate bad channels
#   Step 4  Bad channels (automatic + manual)  Step 8  Average reference
# Takes a few minutes. Safe to re-run (e.g. after editing ica_exclude.json).
# =============================================================================
print("=" * 70)
print(f"  Subject {SUJETO} -- Preprocessing pipeline")
print("=" * 70 + "\n")

_skipped = [s for s in PREPROCESS_SEG_TYPES if not segment_run_used_in_preproc(s)]
if _skipped:
    print(f"  NOTE: not entering preprocessing: {', '.join(_skipped)} "
          "(not inspected in Cell 3, or excluded in Cell 3d)\n")

BAD_CHANNELS: List[str] = []
ICA_INFO: Dict[str, Any] = {}
SEGMENT_QC: Dict[str, Dict[str, Any]] = {}

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # Expected numerical warnings from ICA inversion / spline interpolation.
        warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
        RAW_CLEAN, BAD_CHANNELS, ICA_INFO, SEGMENT_QC = run_preprocessing_steps(
            RAW, SUJETO, SEGMENTS
        )
    print(f"\n{'=' * 70}")
    print(f"  PREPROCESSING COMPLETE  --  Subject {SUJETO}")
    print(f"{'=' * 70}\n")
except Exception as _exc:
    print(f"\n!! PREPROCESSING FAILED: {_exc}")
    logger.error(f"S{SUJETO} | PREPROCESSING FAILED:\n{traceback.format_exc()}")
    RAW_CLEAN = None


#%%
# =============================================================================
# OPTIONAL — ICA component time courses  (QC)
# =============================================================================
# Scrollable ICA component time courses on one segment (pre-ICA data).
# Red overlays = BAD sections, excluded from the ICA fit.
# To change which components are removed:
#   1. look at output_manual/qc/subject_X/ica_topomap_*.png (and this viewer)
#   2. edit the "exclude" list in output_manual/qc/subject_X/ica_exclude.json
#   3. re-run Cell 4 — it uses your list for this decomposition
# =============================================================================
ICA_VIEW_SEGMENT = "LB"     # "LB", "IGT" or "IGT2"
ICA_SOURCES_PICKS = None    # None = all components; or e.g. [0, 1, 5]

if RAW_CLEAN is None:
    print("!! Preprocessing did not complete. Run Cell 4 first.")
else:
    _ica_viewer_script = BASE_DIR / "scripts" / "eeg_ica_viewer.py"
    _step5_path = subject_fif_dir(SUJETO) / "step5_annotated_raw.fif"
    _ica_path = subject_qc_dir(SUJETO) / "ica_solution-ica.fif"
    _segs = get_segments_by_type(SEGMENTS, ICA_VIEW_SEGMENT, apply_time_rules=True)
    if not _ica_viewer_script.exists():
        print("!! eeg_ica_viewer.py not found next to the pipeline.")
    elif not _step5_path.exists():
        print("!! step5 file not found — re-run Cell 4 (Cell 6 deletes it).")
    elif not _ica_path.exists():
        print("!! ICA solution not found — re-run Cell 4.")
    elif not _segs:
        print(f"!! No {ICA_VIEW_SEGMENT} segment for this subject.")
    else:
        _seg_name, _idx, _start, _end = _segs[0]
        _picks_json = "null" if ICA_SOURCES_PICKS is None else str(ICA_SOURCES_PICKS)
        _title = (f"ICA sources — {SUJETO} {ICA_VIEW_SEGMENT}  |  "
                  f"removed: {ICA_INFO.get('excluded', [])}")
        _cmd = [
            _project_python(), str(_ica_viewer_script),
            str(_ica_path), str(_step5_path), str(_start), str(_end),
            "--title", _title, "--picks", _picks_json,
        ]
        print(f"  >> Opening ICA viewer for {ICA_VIEW_SEGMENT} …")
        _run_viewer_subprocess(_cmd)


#%%
# =============================================================================
# CELL 5a — INSPECT CLEANED: LB
# =============================================================================
# Residual check on the cleaned signal (about 40 µV). You may add a BAD mark
# for something still clearly wrong, or right-click to remove a mark that now
# looks fine. Channels cannot be re-marked here — do that in Cell 3.
# =============================================================================
INTERACTIVE_VIEW_CLEAN = True   # False = re-run without opening the viewer

if RAW_CLEAN is None:
    print("!! Cleaned data not available. Fix Cell 4 first.")
else:
    inspect_and_mark_segment(
        RAW_CLEAN, SUJETO, "LB", SEGMENTS,
        tag="clean", interactive=INTERACTIVE_VIEW_CLEAN, apply_time_rules=True,
        segment_qc=SEGMENT_QC, only_preprocessed=True,
    )


#%%
# =============================================================================
# CELL 5b — INSPECT CLEANED: IGT
# =============================================================================
if RAW_CLEAN is None:
    print("!! Cleaned data not available.")
else:
    INTERACTIVE_VIEW_CLEAN = globals().get("INTERACTIVE_VIEW_CLEAN", True)
    inspect_and_mark_segment(
        RAW_CLEAN, SUJETO, "IGT", SEGMENTS,
        tag="clean", interactive=INTERACTIVE_VIEW_CLEAN, apply_time_rules=True,
        segment_qc=SEGMENT_QC, only_preprocessed=True,
    )


#%%
# =============================================================================
# CELL 5c — INSPECT CLEANED: IGT2
# =============================================================================
if RAW_CLEAN is None:
    print("!! Cleaned data not available.")
else:
    INTERACTIVE_VIEW_CLEAN = globals().get("INTERACTIVE_VIEW_CLEAN", True)
    inspect_and_mark_segment(
        RAW_CLEAN, SUJETO, "IGT2", SEGMENTS,
        tag="clean", interactive=INTERACTIVE_VIEW_CLEAN, apply_time_rules=True,
        segment_qc=SEGMENT_QC, only_preprocessed=True,
    )


#%%
# =============================================================================
# CELL 6 — SUMMARY, SAVE CLEANED DATA, PROGRESS LOG
# =============================================================================
# Saves the cleaned .fif (output_manual/clean_data/), the QC files
# (output_manual/qc/subject_X/), updates output_manual/progress_log.csv and
# removes this subject's intermediate files. Then fill in the subject's row
# in the subject list and move on to the next subject in Cell 2.
# =============================================================================
save_and_report_clean(
    RAW_CLEAN, SUJETO, SEGMENTS,
    BAD_CHANNELS,
    ICA_INFO,
    SEGMENT_QC,
)
if RAW_CLEAN is not None:
    print(f"  Final manual marks -> {save_manual_marks(RAW_CLEAN, SUJETO, SEGMENTS, stage='clean')}")
    print(f"  Progress log       -> {log_progress(SUJETO)}")
    if DELETE_INTERMEDIATE_AFTER_SAVE:
        cleanup_intermediate_fif(SUJETO)
    print(f"\n>>> {SUJETO} done. Fill in its row in the subject list, then set SUJETO "
          "to the next subject in Cell 2.\n")

# %%
