import unittest
import numpy as np
from processor import Processor

class ProcessorTests(unittest.TestCase):    
    def setUp(self):
        self.processor = Processor()

    def test_set_baseline(self):
        self.processor.frequency = 125
        self.processor.baseline_duration = 5

        baseline_signal = np.random.rand(3, 125 * 5 + 1)
        self.processor.set_baseline(baseline_signal)

        self.assertSequenceEqual(
            self.processor.baseline_signal.tolist(), 
            baseline_signal[:, 1:].tolist()
        )

    def test_update(self):
        self.processor.signal_duration = 2
        self.processor.frequency = 125
        self.processor.baseline_signal = np.random.rand(3, 100)
        signal_window = np.random.rand(3, 125 * 2 + 1)

        self.processor.update(signal_window)

        self.assertSequenceEqual(
            self.processor.window[:, :100].tolist(),
            self.processor.baseline_signal.tolist()
        )
        self.assertSequenceEqual(
            self.processor.window[:, 100:].tolist(),
            signal_window[:, 1:].tolist()
        )

    def test_get_window(self):
        self.processor.channels = ["C3", "Cz", "C4"]
        self.processor.frequency = 125
        self.processor.window = np.load("./tests/test_data.npy") 

        evoked = self.processor._get_evoked_window()
        self.assertSequenceEqual(
            evoked.data.tolist(),
            self.processor.window.tolist()
        )        

    def test_get_erds(self):
        self.processor.channels = ["C3", "Cz", "C4"]
        self.processor.frequency = 125
        self.processor.baseline_duration = 4
        self.processor.signal_duration = 4
        # self.processor.window = np.random.rand(3, 125*8) 
        self.processor.window = np.load("./tests/test_data.npy") 

        erds = self.processor._get_erds()
        erds["Cz"].plot()
        erds["C3"].plot()
        erds["C4"].plot()

    def test_predict(self):
        pass