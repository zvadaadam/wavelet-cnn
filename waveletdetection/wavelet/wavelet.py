import pywt

class Wavelet(object):

    @staticmethod
    def calculate_wavelet(time, signal, scales, wavelet_name='morl'):

        # sample rate
        dt = time[1] - time[0]

        # perfrom CWT
        [coefficients, frequencies] = pywt.cwt(signal, scales, wavelet_name, dt)

        # Normalized wavelet power spectrum
        power = (coefficients) ** 2

        period = 1 / frequencies

        return power, period

    def plot_wavelet(self):
        raise NotImplemented