import librosa
from scipy.io import wavfile
import numpy as np


def q_transform(path_to_wav, config):
    """
    Performs Constant-Q Transform for wav audio file

    Args:
        path_to_wav (string): Full path to the wav file
        config(dict): Config dictionary with the following attributes:
        samples_between_cqt_columns, n_frequency_bins, bins_per_octave

    Returns:
        np.ndarray[shape=(n_bins, t), dtype=np.complex or np.float]:
        Absolute constant-Q value each frequency at each time.

    """
    sample_rate, wav_stereo = wavfile.read(path_to_wav)
    wav_samples = np.mean(wav_stereo, axis=1)

    hop_length_in = config["samples_between_cqt_columns"]
    n_bins_in = config["n_frequency_bins"]
    bins_octaves_in = config["bins_per_octave"]

    cqt = np.abs(librosa.cqt(
        wav_samples, sample_rate, hop_length=hop_length_in, n_bins=n_bins_in, bins_per_octave=bins_octaves_in
    ))

    return cqt, sample_rate


if __name__ == "__main__":
    qtr, sample_rate = q_transform("/home/andrew/mapsParts/MAPS_MUS-chpn_op25_e2_AkPnBcht.wav",
                                   {"bins_per_octave": 36, "n_frequency_bins": 252, "samples_between_cqt_columns": 512})

    # Easy plot
    # import matplotlib.pyplot as plt
    # plt.pcolormesh(qtr)
    # plt.show()

    # Librose plot
    # import librosa.display
    # librosa.display.specshow(
    #     librosa.amplitude_to_db(qtr, ref=np.max), sr=sample_rate, x_axis='time', y_axis='cqt_note'
    # )
    #
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrum')
    # plt.tight_layout()
    #
    # plt.show()
