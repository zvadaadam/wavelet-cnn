import numpy as np

class DataGenerator(object):

    def __init__(self):
        pass

    @staticmethod
    def stochastic_signal(num_samples, sample_rate=1, sparse_coef=0):
        # TODO: add blank spots
        time = np.linspace(0, num_samples * sample_rate, num_samples, endpoint=False)

        signal = np.random.randint(low=0, high=2, size=num_samples)

        for _ in range(0, sparse_coef):
            for index, inverse in enumerate(np.random.randint(low=0, high=2, size=num_samples)):
                if inverse:
                    signal[index] = 0

        return time, signal

    @staticmethod
    def synthesized_signal(num_samples, freq, sample_rate, noise=0):
        # TODO: add blank spots
        if freq == 0:
            raise Exception('Frequency cannot be zero.')

        signal = np.zeros(int(num_samples))
        time = np.linspace(0, num_samples * sample_rate, num_samples, endpoint=False)
        period = 1 / freq

        sample_range = range(0, int(num_samples))

        for i in sample_range:
            rnd_period_index = np.random.randint(low=0, high=period, size=1)[0]
            is_applied = (1 - noise) < np.random.random_sample()
            index = i + rnd_period_index * is_applied

            if index % period == 0:
                signal[i] += 1

        return time, signal


