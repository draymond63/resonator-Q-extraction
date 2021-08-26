import numpy as np
from matplotlib import pyplot as plt

# Returns IQ data
# ! w0 setting is incorrect
def createData(Q, w0, F0=1, num_points=1000, noise=0.01, plot=False):
	freqs = np.linspace(w0-w0/Q, w0+w0/Q, num_points)
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
	c, b, a = coef
	p_c = np.pi/2
	# Q = (c + b*p_c + a*(p_c**2)) / (2*b + 4*a*p_c)
	Q = (4*a*c - b**2) / (8*a*(b + a*np.pi))
	return np.abs(Q)

if __name__ == '__main__':
	Q = 50000
	iq = createData(Q=Q, w0=5000, F0=1/Q, noise=0, plot=True)
	coef = fitAmpVsPhase(iq, plot=True)
	print(getQ(coef))