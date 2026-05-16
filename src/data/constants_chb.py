# CHB-MIT bipolar channel list (22 channels, matching the EvoBrain paper which
# states "22-channel scalp EEG"). The first 18 are the universally-present
# bipolar pairs; the last 4 (P7-T7, T7-FT9, FT9-FT10, FT10-T8) are the extended
# montage channels listed in PhysioNet's chb01-summary.txt as channels 19-22.
# Files missing any of these 22 (e.g. chb12 sessions 27-29 which use a CS2
# reference montage) will be flagged as failed during preprocessing.
INCLUDED_CHANNELS_CHB = [
    "FP1-F7", "F7-T7",   "T7-P7",     "P7-O1",
    "FP1-F3", "F3-C3",   "C3-P3",     "P3-O1",
    "FP2-F4", "F4-C4",   "C4-P4",     "P4-O2",
    "FP2-F8", "F8-T8",   "T8-P8",     "P8-O2",
    "FZ-CZ",  "CZ-PZ",
    "P7-T7",  "T7-FT9",  "FT9-FT10",  "FT10-T8",
]

NUM_NODES_CHB = len(INCLUDED_CHANNELS_CHB)

CHB_RAW_FREQ = 256
FREQUENCY = 200

# Patients used (EvoBrain paper: 22 patients). chb24 has different montage; chb21
# is a re-recording of the chb01 subject. Use chb01-22 to match paper count.
CHB_PATIENTS = [f"chb{i:02d}" for i in range(1, 23)]
