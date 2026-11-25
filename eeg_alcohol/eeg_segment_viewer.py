"""
Interactive EEG segment viewer  (launched as a subprocess)
==========================================================
Uses the MNE matplotlib browser on all platforms, so the familiar
"Annotations" popup window (with BAD_ list and drag-to-mark) works the
same on macOS, Windows, and Linux.

Features:
  - Scroll with arrow keys or mouse wheel
  - Click a channel name to mark it as bad (toggles)
  - Press 'a' to open the Annotations dialog, then click-drag to add BAD marks
  - Right-click on a BAD region to delete it
  - Close the viewer window when done; edits are saved automatically

Usage (called automatically by the pipeline):
    python eeg_segment_viewer.py <fif_path> <tmin> <tmax>
        [--title TITLE] [--output JSON_PATH] [--scale-uv FLOAT]
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Put the matplotlib config inside the project so we never hit permission
# errors on ~/.matplotlib (common on managed Macs).
_BASE = Path(__file__).resolve().parent.parent
_MPL_DIR = _BASE / ".mplconfig"
_MPL_DIR.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_DIR)

import matplotlib

# Choose the best non-inline GUI backend for each OS.
# All three give the same MNE matplotlib-based annotation dialog.
if sys.platform == "darwin":
    matplotlib.use("macosx")      # native Mac window; same annotation popup as Windows
elif sys.platform == "win32":
    matplotlib.use("TkAgg")       # original PC backend
else:
    matplotlib.use("TkAgg")       # Linux fallback

import mne
mne.viz.set_browser_backend("matplotlib")

# MNE browser scaling compensation (the browser renders ~2x the requested
# scale, so halving it gives the visually expected amplitude in µV).
VIEWER_SCALE_COMPENSATION: float = 0.5


def main(
    fif_path: str,
    tmin: float,
    tmax: float,
    title: str = "",
    output_json: str = "",
    scale_uv: float = 100.0,
) -> None:
    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    initial_bads = set(raw.info["bads"])

    # tmin/tmax arrive in the recording's absolute time frame, while raw.times
    # is always relative to the start of the file. The pipeline hands us a file
    # already cropped to the segment, so that offset (first_time) is non-zero;
    # deriving the bounds from it keeps this correct for cropped and
    # full-length input alike.
    #
    # Clamp in the segment's own relative frame (rel = abs - t_offset), not by
    # adding t_offset to raw.times[-1] and subtracting it back out for crop().
    # That round trip left a ~1e-13 float error that pushed tmax_rel a hair
    # past raw.times[-1], which mne.crop()'s strict bound check rejected --
    # crashing before the "[Viewer]" print below, silently, for whichever
    # segment's timestamps happened not to round-trip cleanly.
    t_offset = raw.first_time
    tmax_rel = min(tmax - t_offset, raw.times[-1])
    tmin_rel = max(tmin - t_offset, 0.0)
    tmin_rel = min(tmin_rel, tmax_rel)
    segment = raw.copy().crop(tmin=tmin_rel, tmax=tmax_rel)
    tmin_clamped = tmin_rel + t_offset
    tmax_clamped = tmax_rel + t_offset

    # Remember annotations that were already there before this session.
    initial_bad_ann_keys = set()
    for ann in segment.annotations:
        if ann["description"].upper().startswith("BAD"):
            initial_bad_ann_keys.add(
                (round(float(ann["onset"]), 6),
                 round(float(ann["duration"]), 6),
                 str(ann["description"]))
            )

    print(f"  [Viewer] {tmin_clamped:.1f}–{tmax_clamped:.1f} s  |  scale {scale_uv:.0f} µV  "
          f"|  press 'a' to annotate BAD, close when done")

    # On macOS the window opens behind other apps when launched from a Jupyter
    # kernel.  Run the osascript in a background thread so it fires ~1 s after
    # the window renders without blocking segment.plot().
    if sys.platform == "darwin":
        import subprocess as _sp
        import threading
        _pid = os.getpid()
        def _raise_window():
            import time
            time.sleep(1.0)
            _sp.run(
                ["osascript", "-e",
                 f'tell application "System Events" to set frontmost of '
                 f'(first process whose unix id is {_pid}) to true'],
                capture_output=True,
            )
        threading.Thread(target=_raise_window, daemon=True).start()

    segment.plot(
        title=title or f"EEG Segment  [{tmin_clamped:.1f} – {tmax_clamped:.1f} s]",
        n_channels=31,
        scalings={
            "eeg": scale_uv * VIEWER_SCALE_COMPENSATION * 1e-6,
            "eog": scale_uv * VIEWER_SCALE_COMPENSATION * 1e-6,
        },
        block=True,
        show=True,
    )

    # Collect changes made during this session.
    new_bads = [ch for ch in segment.info["bads"] if ch not in initial_bads]

    new_annotations = []
    for ann in segment.annotations:
        if not ann["description"].upper().startswith("BAD"):
            continue
        key = (round(float(ann["onset"]), 6),
               round(float(ann["duration"]), 6),
               str(ann["description"]))
        if key in initial_bad_ann_keys:
            continue
        new_annotations.append({
            "onset":       float(ann["onset"]),
            "duration":    float(ann["duration"]),
            "description": ann["description"],
        })

    final_annotations = [
        {
            "onset":       float(ann["onset"]),
            "duration":    float(ann["duration"]),
            "description": str(ann["description"]),
        }
        for ann in segment.annotations
        if ann["description"].upper().startswith("BAD")
    ]

    result = {
        "segment_tmin":       tmin_clamped,
        "segment_tmax":       tmax_clamped,
        "bad_channels_added": new_bads,
        "annotations_added":  new_annotations,   # backward compat
        "annotations_final":  final_annotations, # full state after edits
        "bad_channels_final": list(segment.info["bads"]),  # full state after edits
    }

    if output_json:
        out_path = Path(output_json)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  [Viewer] Edits saved -> {out_path.name}")
    else:
        print(json.dumps(result))

    if new_bads:
        print(f"  [Viewer] Bad channels marked : {new_bads}")
    if new_annotations:
        print(f"  [Viewer] BAD annotations added: {len(new_annotations)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive EEG segment viewer")
    parser.add_argument("fif_path",          help="Path to the .fif file")
    parser.add_argument("tmin",   type=float, help="Segment start (seconds)")
    parser.add_argument("tmax",   type=float, help="Segment end (seconds)")
    parser.add_argument("--title",    type=str,   default="",    help="Window title")
    parser.add_argument("--output",   type=str,   default="",    help="Output JSON path")
    parser.add_argument("--scale-uv", type=float, default=100.0, help="Initial scale µV")
    args = parser.parse_args()
    main(args.fif_path, args.tmin, args.tmax, args.title, args.output, args.scale_uv)
