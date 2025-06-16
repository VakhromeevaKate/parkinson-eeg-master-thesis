import numpy as np
import mne
from autoreject import get_rejection_threshold

def convert_bdf_to_tensor_list(path, duration=2.0):
    epochs_list = []
    raw = mne.io.read_raw_bdf(path, preload=True)
    raw.drop_channels(['EXG1', 'EXG2', 'EXG3', 'EXG4','EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status']) #drop extra channels
    raw.set_eeg_reference(ref_channels='average')
    raw.filter(0.5, None, fir_design='firwin',phase='zero-double') #remove drifts
    epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=False, proj=True)
    reject = get_rejection_threshold(epochs)
    epochs.drop_bad(reject=reject)
    epochs_list.append(epochs)
    epochs_data = np.concatenate([e.get_data() for e in epochs_list])
    return epochs_data
    