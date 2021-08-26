import numpy as np
from matplotlib import pyplot as plt

# Returns IQ data
def createData(Q, w0, F0=1, num_points=1000, noise=0.01, plot=False):
	freqs = np.linspace(w0 / 2, w0*2 + 10, num_points)
	lorentzFunc = lambda f: F0 / np.sqrt((1-(f**2)/(w0**2))**2 + (f/(Q*w0))**2)
	lorentz = lorentzFunc(freqs) + noise*np.random.randn(len(freqs))

	arcFunc = lambda f: np.pi/2 - np.arctan(Q*(f/w0-w0/f))
	arc = arcFunc(freqs) + noise*np.random.randn(len(freqs))

	iq = lorentz * np.exp(arc * 1j)

	if plot:
		fig, axs = plt.subplots(2, 1)
		axs[0].plot(lorentz)
		axs[1].plot(arc)
		plt.show()
	return iq

# Returns polynomial
def fitAmpVsPhase(iq, plot=False):
	amp = np.abs(iq)
	phase = np.unwrap(np.angle(iq))

	poly = np.polynomial.polynomial.Polynomial.fit(phase, amp, deg=2).convert().coef
	print(poly)
	if plot:
		fit = np.polynomial.polynomial.polyval(phase, poly)
		plt.plot(phase, amp, 'o')
		plt.plot(phase, fit)
		plt.show()
	return poly

# Gets Q
def getQ(coef: tuple):
	c, b, a = coef
	Q = b / (2 * c)
	return Q

if __name__ == '__main__':
	Q = 1000
	iq = createData(Q=Q, w0=5, F0=1/Q, noise=0, num_points=10000, plot=True)
	coef = fitAmpVsPhase(iq)
	print(getQ(coef))