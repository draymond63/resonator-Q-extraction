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
		plt.plot(lorentz)
		plt.plot(arc)
		plt.show()
		plt.plot(np.real(iq), np.imag(iq), 'o')
		plt.show()
	return iq

# Returns polynomial
def fitAmpVsPhase(iq, plot=False):
	# amp = np.abs(iq)
	# phase = 
	test_coef = (1, 1, 1)
	amp = np.polynomial.polynomial.polyval(phase, test_coef)

	poly = np.polynomial.polynomial.Polynomial.fit(phase, amp, deg=2)
	print(poly.coef)
	if plot:
		fit = np.polynomial.polynomial.polyval(phase, poly.coef)
		plt.plot(phase, fit)
		plt.plot(phase, amp)
		plt.show()
	return poly.coef

# Gets Q
def getQ(coef: tuple):
	c, b, a = coef
	Q = c / (2 * b)
	return Q

if __name__ == '__main__':
	Q = 100
	iq = createData(Q=Q, w0=5, F0=1/Q, noise=0, num_points=10000)
	coef = fitAmpVsPhase(iq, plot=True)
	print(getQ(coef))