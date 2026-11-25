"""
eeg_ica_viewer.py — ICA component source viewer  (launched as a subprocess)
============================================================================
Displays ICA component time courses for the LB segment so you can verify
which components are eye artifacts before accepting or overriding the
auto-exclusion stored in output/qc/subject_X/ica_exclude.json.

Usage (called automatically by the pipeline Optional cell):
    python eeg_ica_viewer.py <ica_fif> <raw_fif> <tmin> <tmax>
                             [--title TITLE] [--picks JSON]

Arguments:
    ica_fif   Path to the saved ICA solution  (.fif)
    raw_fif   Path to the step5_annotated pre-ICA filtered raw  (.fif)
    tmin      Segment start in seconds
    tmax      Segment end in seconds
    --title   Window title (optional)
    --picks   JSON list of component indices to show, e.g. "[0,1,2]"
              Omit or pass "null" to show all components.
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# Keep matplotlib config inside the project (avoids permission errors).
_BASE = Path(__file__).resolve().parent.parent
_MPL_DIR = _BASE / ".mplconfig"
_MPL_DIR.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_DIR)

import matplotlib
if sys.platform == "darwin":
    matplotlib.use("macosx")
elif sys.platform == "win32":
    matplotlib.use("TkAgg")
else:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import mne

# Use the matplotlib browser for plot_sources (mirrors eeg_segment_viewer.py).
# Without this, MNE defaults to the Qt browser (mne-qt-browser), which clashes
# with the macosx backend inside this subprocess and segfaults ("Python quit
# unexpectedly") on macOS.
mne.viz.set_browser_backend("matplotlib")

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*overflow.*", category=RuntimeWarning)


def main() -> None:
    parser = argparse.ArgumentParser(description="ICA source viewer")
    parser.add_argument("ica_fif",  help="ICA solution .fif")
    parser.add_argument("raw_fif",  help="Pre-ICA filtered raw .fif")
    parser.add_argument("tmin",     type=float, help="Segment start (s)")
    parser.add_argument("tmax",     type=float, help="Segment end (s)")
    parser.add_argument("--title",  default="ICA components", help="Window title")
    parser.add_argument("--picks",  default="null", help='JSON list or "null" for all')
    args = parser.parse_args()

    picks = json.loads(args.picks)  # None or list[int]

    ica = mne.preprocessing.read_ica(args.ica_fif, verbose=False)
    raw = mne.io.read_raw_fif(args.raw_fif, preload=True, verbose=False)

    tmin_eff = max(args.tmin, raw.times[0])
    tmax_eff = min(args.tmax, raw.times[-1])
    raw.crop(tmin=tmin_eff, tmax=tmax_eff)

    n_comp = ica.n_components_
    n_excl = len(ica.exclude)
    print(
        f"  [ICA viewer] {tmin_eff:.1f}–{tmax_eff:.1f} s  "
        f"({tmax_eff - tmin_eff:.0f} s)  |  "
        f"{n_comp} components  |  auto-excluded: {ica.exclude}  |  "
        f"close window when done"
    )

    ica.plot_sources(raw, picks=picks, show=True, block=True, title=args.title)
    plt.show(block=True)


if __name__ == "__main__":
    main()
