# eeg-alcohol-preprocessing

EEG preprocessing and spectral extraction used in a study of acute psychosocial stress (TSST-VR)
and decision-making under ambiguity (Iowa Gambling Task, IGT). Peripheral physiology from the same
sessions was processed with
[biopac-physio-preprocessing](https://github.com/MaragdaPP/biopac-physio-preprocessing).

## Recording

BrainVision actiCHamp, 30 scalp electrodes (10–20 system), 500 Hz, online reference TP10; TP9 was
placed below the left eye and is used as the electro-oculographic channel. Segments analysed:
baseline (LB, last 5 min of the marked block), IGT before the manipulation and IGT after it
(marked `IGT_ALC` in the files; no alcohol was given in the session).

## Files

| File | |
|---|---|
| `eeg_alcohol/eeg_preprocessing_alcohol.py` | cell-by-cell preprocessing, one participant at a time (MNE) |
| `eeg_alcohol/eeg_segment_viewer.py`, `eeg_ica_viewer.py` | viewers the pipeline opens for visual inspection |
| `eeg_alcohol/spectral_extract.py` | spectra and band power from the cleaned files |
| `requirements.txt` | exact versions the pipeline was verified with (Python 3.11) |

## Preprocessing (pipeline version 2026-09-16)

1. Channel types and standard 10–20 montage; TP9 typed as EOG.
2. Two filtered copies: 1–40 Hz for fitting ICA, 0.1–40 Hz for the final data; 50-Hz notch.
3. Bad channels per segment: local outlier factor, flat lines and persistent variance, united with
   channels marked by hand. A segment with more than three bad channels is not processed.
4. Artefacts: peak-to-peak > 250 µV and 30–40 Hz muscle activity, united with stretches marked by hand.
5. FastICA (fitted on the 1–40 Hz copy); at most two ocular components, identified by their
   correlation with TP9, Fp1 and Fp2, are removed from the final data.
6. Bad channels are interpolated and each segment is re-referenced to the average.
7. A segment is retained when no more than 35 % of it is marked as artefact
   (`ANALYSIS_MAX_BAD_TIME_RATIO = 0.35`).

Marker repairs for individual recordings (renamed files, mislabelled or missing markers,
boundaries recovered from the BIOPAC recording) are applied automatically and documented in
`apply_subject_patches()`.

## Spectra

Hann-windowed spectra of artefact-free 4-s pieces (50 % overlap), averaged per segment. Relative
power = band power / 1–40 Hz power. Posterior alpha = 8–13 Hz over Pz, P3, P4, P7, P8, O1, Oz and
O2. Periodic alpha and the aperiodic component with specparam (3–40 Hz); 30–40 Hz power as an
index of muscle activity.

## Use

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# preprocessing: open eeg_alcohol/eeg_preprocessing_alcohol.py in an editor with cells,
# put the recordings in raw_data/, set SUJETO in Cell 2 and run cell by cell
python eeg_alcohol/spectral_extract.py --check output_manual/clean_data/subject_<ID>_clean_raw.fif --qc output_manual/qc --subject <ID>
```

No data are included: the recordings belong to study participants.

## Credits

Pipeline: Maragda Puigcerver Pérez. Preprocessing of the recordings and spectral analysis:
Anđela Bošnjak. Universitat de València.

MIT licence.
