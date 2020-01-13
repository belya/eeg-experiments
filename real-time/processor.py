import numpy as np
import mne

class Processor():
    channels = None
    frequency = None
    baseline_duration = None
    signal_duration = None
    baseline_signal = None
    window = None

    def _cut_signal(self, signal, duration):
        samples = duration * self.frequency
        signal = signal[:, -samples:]
        return signal

    def set_baseline(self, baseline_signal):
        self.baseline_signal = self._cut_signal(baseline_signal, self.baseline_duration)

    def update(self, signal_window):
        signal_window = self._cut_signal(signal_window, self.signal_duration)
        self.window = np.hstack([
            self.baseline_signal, 
            signal_window
        ])

    def _get_evoked_window(self):
        info = mne.create_info(
            self.channels, 
            self.frequency,
            ["eeg"] * len(self.channels)
        )
        evoked = mne.EvokedArray(self.window, info)
        # evoked.filter(5, 50)
        return evoked

    def _get_erds(self):
        frequencies = np.arange(7, 45, 1)
        evoked_window = self._get_evoked_window()
        baseline = [0, self.baseline_duration]
        erds = {}

        for i, channel in enumerate(self.channels):
            channel_pick = mne.pick_channels(self.channels, [
                channel
            ])
            power = mne.time_frequency.tfr_multitaper(
                evoked_window, 
                freqs=frequencies, 
                n_cycles=frequencies,
                return_itc=False, 
                decim=3, 
                n_jobs=1,
                picks=channel_pick
            )
            print(baseline)
            power.apply_baseline(baseline, mode='percent')
            erds[channel] = power

        return erds