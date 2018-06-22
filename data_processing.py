import os
import pickle
import numpy as np
import librosa
import soundfile as sf

def mel_filterbank(n_filters, n_fft, sr):
    """
    Mel-warping filterbank.
    You do not need to edit this code; it is needed to contruct the mel filterbank
    which we will use to extract features.
    --------
    :in:
    n_filters, number of filters
    n_fft, window size over which fft is performed
    sr, sampling rate of signal
    --------
    :out:
    mel_filter, 2d-array of (n_fft / 2, n_filters) used to get mel features
    mel_inv_filter, 2d-array of (n_filters, n_fft / 2) used to invert
    melpoints, 1d-array of frequencies converted to mel-scale
    """
    freq2mel = lambda f: 2595. * np.log10(1 + f / 700.)
    mel2freq = lambda m: 700. * (10 ** (m / 2595.) - 1)

    lowfreq = 0
    highfreq = sr // 2

    lowmel = freq2mel(lowfreq)
    highmel = freq2mel(highfreq)

    melpoints = np.linspace(lowmel, highmel, 1 + n_filters + 1)

    # must convert from freq to fft bin number
    fft_bins = ((n_fft + 1) * mel2freq(melpoints) // sr).astype(np.int32)

    filterbank = np.zeros((n_filters, n_fft // 2))
    for j in range(n_filters):
        for i in range(fft_bins[j], fft_bins[j + 1]):
            filterbank[j, i] = (i - fft_bins[j]) / (fft_bins[j + 1] - fft_bins[j])
        for i in range(fft_bins[j + 1], fft_bins[j + 2]):
            filterbank[j, i] = (fft_bins[j + 2] - i) / (fft_bins[j + 2] - fft_bins[j + 1])

    mel_filter = filterbank.T / filterbank.sum(axis=1).clip(1e-16)
    mel_inv_filter = filterbank

    return mel_filter, mel_inv_filter, melpoints


def log_melfbank(signal_dir, n_filters=40, n_fft=512):
    # Load audio
    signal, sr_ = sf.read(signal_dir)
    # Short time fourier transform to complex number
    spectrum = librosa.core.stft(signal, n_fft=n_fft)
    # Cut 1 since librosa output shape + 1
    spectrum = spectrum[:-1]
    # Get magnitude
    magnitude = np.abs(spectrum)
    # Power
    power = (magnitude ** 2).T
    # Set mel filterbanks
    mel_filter, _, _ = mel_filterbank(n_filters = n_filters, n_fft = n_fft, sr = sr_)
    # Get mel fbank features
    mel_fbanks = power.dot(mel_filter).clip(1e-16)
    # Get log scale mel filterbanks
    log_melfbanks = np.log10(mel_fbanks)

    return log_melfbanks

def data_process(set_dir, save_name, n_filters=40, n_fft=512):
    # Process data and save into np array
    print('Extracting features..')
    res_data = []
    count = 1

    for signal_dir in set_dir:
        log_mel_feature = np.around(log_melfbank(signal_dir, n_filters=n_filters, n_fft=n_fft), decimals=6)
        res_data.append(log_mel_feature)

        # Print processing status
        print(count/len(set_dir)*100)
        count += 1
    # Save data into numpy array
    np.save(save_name + '_' + str(n_filters), np.asarray(res_data))
    print(np.asarray(res_data).shape)

def data_pre_window(file_name, save_name, frame_size=19, step=8):
    # Cut utterance into different frame sizes with overlaps (steps)
    print('Frame segmentation..')
    features = np.load(file_name)
    final_data = []
    count = 1
    for utterance in features:
        # Save each frame into final data
        for i in range(0,len(utterance)-frame_size, step):
            final_data.append(utterance[i:i+frame_size,:].flatten())

        print(count/len(features)*100)
        count+=1
    np.save(save_name + '_' + str(frame_size) + '_' + str(step), np.asarray(final_data))


# load train/dev/test directory
with open("train_dir.txt", "rb") as fp:
    train_dir = pickle.load(fp)
with open("dev_dir.txt", "rb") as fp:
    dev_dir = pickle.load(fp)
with open("test_dir.txt", "rb") as fp:
    test_dir = pickle.load(fp)


data_dirs = [train_dir, dev_dir, test_dir]
save_name = ['train', 'dev', 'test']
n_filters_list = [40, 64, 80]
n_fft = 512
frame_size = 19
step = 8



for data_dir, save_n in zip(data_dirs, save_name):
    for n_filters in n_filters_list:
        data_process(set_dir=data_dir, save_name=save_n, n_filters=n_filters, n_fft=n_fft)
        file_name = save_n + '_' + str(n_filters) + '.npy'
        save_name_ = save_n + '_' + str(n_filters)
        data_pre_window(file_name=file_name, save_name = save_name_, frame_size=frame_size, step=step)
