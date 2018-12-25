import numpy as np
import wave
import math
import contextlib
from scipy.io import wavfile
from scipy.signal import butter, lfilter


def running_mean(x, windowSize):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize


def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved=True):
    if sample_width == 1:
        dtype = np.uint8  # unsigned char
    elif sample_width == 2:
        dtype = np.int16  # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.fromstring(raw_bytes, dtype=dtype)

    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels


def remove_low_freq(infile, cut_off_freq):
    with contextlib.closing(wave.open(infile, 'rb')) as spf:
        sampleRate = spf.getframerate()
        ampWidth = spf.getsampwidth()
        nChannels = spf.getnchannels()
        nFrames = spf.getnframes()

        # Extract Raw Audio from multi-channel Wav File
        signal = spf.readframes(nFrames * nChannels)
        spf.close()
        channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)

        # get window size
        # from http://dsp.stackexchange.com/questions/9966/what-is-the-cut-off-frequency-of-a-moving-average-filter
        freqRatio = (cut_off_freq / sampleRate)
        N = int(math.sqrt(0.196196 + freqRatio ** 2) / freqRatio)

        # Use moviung average (only on first channel)
        filtered = running_mean(channels[0], N).astype(channels.dtype)

        return filtered

        # wav_file = wave.open(filtered_file,  "w")
        # wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
        # wav_file.writeframes(filtered.tobytes('C'))
        # wav_file.close()


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    print(cutoff, fs)
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def remove_high_freq(samples_original, sample_rate_original, cut_off_freq):
    # samples_original = samples_original[:, 0]
    print()
    filtered_samples = butter_lowpass_filter(samples_original, cut_off_freq, sample_rate_original, 6)

    return filtered_samples


if __name__ == '__main__':
    pass
