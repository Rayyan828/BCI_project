# src/utils.py
import os
import re
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import welch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

DATA_PATH = 'data/Data/filtered_data/'
LABELS_PATH = 'data/Data/scales.xls'

# frequency bands
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta' : (13, 30),
    'gamma': (30, 50)
}

# ----------------------------------------------------
#                 DATA LOADING
# ----------------------------------------------------
def load_mat_data(data_path=DATA_PATH, labels_path=LABELS_PATH, skip_relax=True):
    labels_df = pd.read_excel(labels_path, header=[0,1])
    labels_df.columns = [
        "_".join([str(c) for c in col if "Unnamed" not in str(c)]).strip()
        for col in labels_df.columns
    ]
    subj_col = next((c for c in labels_df.columns if "Subject" in c), None)
    if subj_col:
        labels_df.rename(columns={subj_col: "Subject"}, inplace=True)
    else:
        raise KeyError("Could not find subject column in Excel file.")

    X_raw, y, subjects, tasks = [], [], [], []

    for file in sorted(os.listdir(data_path)):
        if not file.endswith('.mat'):
            continue
        if skip_relax and "Relax" in file:
            continue

        mat = scipy.io.loadmat(os.path.join(data_path, file))
        if 'Clean_data' not in mat:
            continue

        eeg = np.real(mat['Clean_data'])
        if eeg.ndim == 1:
            eeg = np.expand_dims(eeg, axis=0)

        m = re.search(r'_sub_(\d+)_trial(\d+)', file)
        if not m:
            continue

        subject_id = int(m.group(1))
        trial_num  = int(m.group(2))

        if "Arithmetic" in file: task = "Maths"
        elif "Mirror" in file:  task = "Symmetry"
        elif "Stroop" in file:  task = "Stroop"
        else: continue

        trial_col = f"Trial_{trial_num}_{task}"
        row = labels_df[labels_df["Subject"] == subject_id]
        if row.empty or trial_col not in labels_df.columns:
            continue

        label_value = int(row[trial_col].values[0])

        X_raw.append(eeg)
        y.append(label_value)
        subjects.append(subject_id)
        tasks.append(task)

    X_raw = np.array(X_raw, dtype=object)
    y = np.array(y, dtype=int)
    subjects = np.array(subjects, dtype=int)
    tasks = np.array(tasks, dtype=object)

    print(f"Loaded {len(X_raw)} samples; example shape: {X_raw[0].shape if len(X_raw)>0 else 'N/A'}")
    return X_raw, y, subjects, tasks


# ----------------------------------------------------
#                 SIGNAL PROCESSING HELPERS
# ----------------------------------------------------
def _welch_real(x, sf=256, nperseg=512):
    """Convert ANY input (object, nested, complex) into real float64 & run Welch safely."""
    x = np.asarray(x)               # force ndarray
    x = np.real(x).astype(np.float64, copy=False)  # guaranteed pure float, no complex left

    freqs, psd = welch(x, fs=sf, nperseg=nperseg)
    psd = np.real(psd).astype(np.float64, copy=False)
    return freqs, psd



def bandpower_psd(eeg_channel, sf=256, nperseg=512, bands=BANDS):
    freqs, psd = _welch_real(eeg_channel, sf=sf, nperseg=nperseg)

    total_power = float(np.trapz(psd, freqs)) + 1e-12
    out = {}
    for name, (low, high) in bands.items():
        idx = (freqs >= low) & (freqs <= high)
        bp = float(np.trapz(psd[idx], freqs[idx]))
        out[name] = bp / total_power
    return out


def spectral_entropy(signal, sf=256, nperseg=512, eps=1e-12):
    """Compute spectral entropy safely without complex numbers."""
    signal = np.real(signal)
    freqs, psd = _welch_real(signal, sf=sf, nperseg=nperseg)
    psd = np.real(psd)

    psd_sum = float(np.sum(psd)) + eps
    psd_norm = psd / psd_sum
    psd_norm = np.real(psd_norm)

    psd_norm[psd_norm <= 0] = eps
    se = -np.sum(psd_norm * np.log2(psd_norm))
    return float(se)


def hjorth_parameters(x):
    x = np.real(np.asarray(x))
    d1 = np.diff(x)
    d2 = np.diff(d1)

    var_x = np.var(x) + 1e-12
    var_d1 = np.var(d1) + 1e-12
    var_d2 = np.var(d2) + 1e-12

    activity = var_x
    mobility = np.sqrt(var_d1 / var_x)
    complexity = np.sqrt(var_d2 / var_d1) / (mobility + 1e-12)

    return float(activity), float(mobility), float(complexity)


def frontal_asymmetry(band_list, idx_left=0, idx_right=1):
    try:
        return float(band_list[idx_left]["alpha"] - band_list[idx_right]["alpha"])
    except:
        return 0.0


# ----------------------------------------------------
#                 ADVANCED FEATURE EXTRACTION
# ----------------------------------------------------
def extract_features_advanced(
        X_raw, sf=256, nperseg=512,
        bands=BANDS, use_hjorth=True, use_entropy=True, use_ratios=True):
    """
    Compute:
     - Bandpowers per channel
     - Spectral entropy
     - Hjorth parameters
     - Band ratios (theta/alpha etc.)
     - Frontal asymmetry
    """
    feats = []

    for arr in X_raw:
        arr = np.real(arr)
        ch_count = arr.shape[0]
        channel_feats = []
        band_list = []

        for ch in range(ch_count):
            sig = arr[ch, :]

            bp = bandpower_psd(sig, sf=sf, nperseg=nperseg, bands=bands)
            band_list.append(bp)

            # band powers
            channel_feats.extend([bp['delta'], bp['theta'], bp['alpha'], bp['beta'], bp['gamma']])

            if use_entropy:
                channel_feats.append(spectral_entropy(sig, sf=sf, nperseg=nperseg))

            if use_hjorth:
                a, m, c = hjorth_parameters(sig)
                channel_feats.extend([a, m, c])

            if use_ratios:
                t = bp['theta'] + 1e-12
                a = bp['alpha'] + 1e-12
                b = bp['beta']  + 1e-12
                channel_feats.extend([t/a, t/b, a/b])

        # frontal asymmetry
        fa = frontal_asymmetry(band_list)
        channel_feats.append(fa)

        feats.append(channel_feats)

    return np.array(feats, dtype=float)


# ----------------------------------------------------
#                 LABELS + SPLITS + NORMALIZATION
# ----------------------------------------------------
def labels_to_3class(y):
    y3 = np.zeros_like(y, dtype=int)
    y3[(y >= 1) & (y <= 3)] = 0
    y3[(y >= 4) & (y <= 7)] = 1
    y3[(y >= 8) & (y <= 10)] = 2
    return y3


def subject_wise_split(X, y, subjects, test_size=0.2, val_size=0.1, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss.split(X, y, groups=subjects))

    X_train_val, X_test = X[train_val_idx], X[test_idx]
    y_train_val, y_test = y[train_val_idx], y[test_idx]
    subj_train_val = subjects[train_val_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=random_state)
    train_idx, val_idx = next(gss2.split(X_train_val, y_train_val, groups=subj_train_val))

    return (
        X_train_val[train_idx], X_train_val[val_idx], X_test,
        y_train_val[train_idx], y_train_val[val_idx], y_test
    )


def normalize_dataset(X_train, X_val, X_test, scaler_path=None):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    if scaler_path:
        import joblib
        joblib.dump(scaler, scaler_path)

    return X_train_s, X_val_s, X_test_s, scaler
