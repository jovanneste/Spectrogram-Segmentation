from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

sample_rate, samples = wavfile.read('5927.230919235004.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.figure(figsize=(0, 5))
plt.subplot(1, 2, 1)
plt.pcolormesh(times, frequencies, np.log(spectrogram))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Full Spectrogram')

plt.subplot(1, 2, 2)

n = 5
# Find the index of the time point closest to the nth second
index_ns = np.argmin(np.abs(times - n))
plt.pcolormesh(times[:index_ns+1], frequencies, np.log(spectrogram[:, :index_ns+1]))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title(f'Spectrogram of {n}th Second')

plt.tight_layout()
plt.show()
