import numpy as np
import pandas as pd
from tqdm import tqdm
from waveletdetection.dataset.base_dataset import DatasetBase
from waveletdetection.dataset.data_generator import DataGenerator
from waveletdetection.wavelet.wavelet import Wavelet


class SignalDataset(DatasetBase):

    def __init__(self, config):
        super(SignalDataset, self).__init__(config)

        self.load_dataset()

    def load_dataset(self):
        self.df = self.generate_dataset()
        self.split_dataset()

    def generate_dataset(self, num_samples=None, stochastic_ratio=None, max_period=None,
                     signal_length=None, sample_rate=None, wavelet_name=None, max_scale=None):
        if num_samples is None:
            num_samples = self.config.num_samples()

        if stochastic_ratio is None:
            stochastic_ratio = self.config.stochastic_ratio()

        if max_period is None:
            max_period = self.config.max_period()

        if signal_length is None:
            signal_length = self.config.signal_length()

        if sample_rate is None:
            sample_rate = self.config.sample_rate()

        if wavelet_name == None:
            wavelet_name = self.config.wavelet_name()

        if max_scale == None:
            max_scale = self.config.max_scale()

        num_stochastic_samples = int(num_samples * stochastic_ratio)
        num_synthetic_samples = int(num_samples * (1 - stochastic_ratio))

        rnd_periods = np.random.randint(low=1, high=max_period, size=num_synthetic_samples)
        rnd_freqs = 1 / rnd_periods

        rnd_noises = np.linspace(start=0, stop=0.7, num=num_synthetic_samples)
        rnd_noises = np.around(rnd_noises, decimals=1)

        scales = np.arange(1, max_scale)

        times = []
        periods = []
        wavelet_coefs = []
        type_generators = []
        frequencies = []
        noises = []

        desc = 'Generating synthetic signal data'
        for frequency, noise in tqdm(zip(rnd_freqs, rnd_noises), total=num_synthetic_samples, desc=desc):
            # TODO: add blank spots
            time, signal = DataGenerator.synthesized_signal(signal_length, frequency, sample_rate, noise)

            coef, period = Wavelet.calculate_wavelet(time, signal, scales, wavelet_name)

            coef = self.coef_preprocessing(coef)

            times.append(time)
            periods.append(period)
            wavelet_coefs.append(coef)
            frequencies.append(frequency)
            noises.append(noise)
            #type_generators.append('synthetic')
            type_generators.append(1)

        desc = 'Generating stochastic signal data'
        for _ in tqdm(range(num_stochastic_samples), desc=desc):
            # TODO: add blank spots
            time, signal = DataGenerator.stochastic_signal(signal_length, sample_rate)

            coef, period = Wavelet.calculate_wavelet(time, signal, scales, wavelet_name)

            coef = self.coef_preprocessing(coef)

            times.append(time)
            periods.append(period)
            wavelet_coefs.append(coef)
            frequencies.append(np.nan)
            noises.append(np.nan)
            #type_generators.append('random')
            type_generators.append(0)

        return self.create_dataframe(times, periods, wavelet_coefs, type_generators, frequencies, noises)

    def create_dataframe(self, times, periods, wavelet_coefs, type_generators, frequencies, noises):

        df = pd.DataFrame()

        df['time'] = times
        df['period'] = periods
        df['coef'] = wavelet_coefs
        df['label'] = type_generators
        df['type_generator'] = type_generators
        df['freq'] = frequencies
        df['noise'] = noises

        # shuffle the dataset
        df = df.sample(frac=1).reset_index(drop=True)

        return df

    def coef_preprocessing(self, coef, preprocess_method=None):

        if preprocess_method == None:
            preprocess_method = self.config.coef_preprocess()

        if preprocess_method == 'mean_sub':
            coef = self.mean_sub(coef)
        elif preprocess_method == 'normalization':
            coef = self.normalization(coef)

        return coef

    def mean_sub(self, coef):

        print(coef)

    def normalization(self, coef):

        coef_norm = coef - coef.mean()
        coef_norm = coef_norm / coef_norm.max()

        return coef_norm




if __name__ == '__main__':

    from waveletdetection.config.config_reader import ConfigReader

    dataset = SignalDataset(config=ConfigReader())

    coef = dataset.train_df['coef']

    coef = np.array(coef.values.tolist())
    coef = np.expand_dims(coef, axis=3)

    print(coef)

    print(dataset.df.head())

