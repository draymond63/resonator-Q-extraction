import numpy as np
from matplotlib import pyplot as plt

# Returns IQ data
# ! w0 setting is incorrect
def createData(Q, w0, F0=1, num_points=1000, noise=0.01, plot=False):
	freqs = np.linspace(w0 / 2, w0*2 + 10, num_points)
	lorentzFunc = lambda f: F0 / np.sqrt((1-(f**2)/(w0**2))**2 + (f/(Q*w0))**2)
	lorentz = lorentzFunc(freqs) + noise*np.random.randn(len(freqs))

	arcFunc = lambda f: np.pi/2 - np.arctan(Q*(f/w0-w0/f))
	arc = arcFunc(freqs) + noise*np.random.randn(len(freqs))

	iq = lorentz * np.exp(arc * 1j)

	if plot:
		_, axs = plt.subplots(2, 1)
		axs[0].plot(lorentz)
		axs[1].plot(arc)
		plt.show()
	return iq

# Returns polynomial
def fitAmpVsPhase(iq, plot=False):
	amp = np.abs(iq)
	phase = np.angle(iq)
	coef = np.polynomial.polynomial.polyfit(phase, amp, deg=2, w=amp)

	if plot:
		fit_x = np.linspace(0, np.pi, len(phase))
		fit_y = np.polynomial.polynomial.polyval(fit_x, coef)
		plt.plot(phase, amp, 'o')
		plt.plot(fit_x, fit_y)
		plt.show()
	return coef

# Gets Q
def getQ(coef: tuple):
	print(coef)
	assert len(coef) == 3, "Polyfit must be degree=3"
	# ! Coefficients must be in root form, which they are not
	c, b, a = coef
	Q = c / (2 * b)
	return Q

if __name__ == '__main__':
	Q = 10
	iq = createData(Q=Q, w0=5, F0=1/Q, plot=True)
	coef = fitAmpVsPhase(iq)
	print(getQ(coef))