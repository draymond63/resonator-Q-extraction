import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# Returns IQ data
def createData(Q, w0, F0=1, num_points=1000, noise=0.01, plot=False):
	freqs = np.linspace(w0 - 2 * num_points/Q, w0 + 2 * num_points/Q, num_points)
	lorentzFunc = lambda f: F0 / np.sqrt((1-(f**2)/(w0**2))**2 + (f/(Q*w0))**2)
	lorentz = lorentzFunc(freqs) + noise*np.random.randn(len(freqs))

	arcFunc = lambda f: np.pi/2 - np.arctan(Q*(f/w0-w0/f))
	arc = arcFunc(freqs) + noise*np.random.randn(len(freqs))

	iq = lorentz * np.exp(arc * 1j)

	if plot:
		_, axs = plt.subplots(2, 1)
		axs[0].plot(freqs, lorentz)
		axs[1].plot(freqs, arc)
		plt.show()
	return iq

# Returns polynomial
def fitAmpVsPhase(iq, plot=False):
	amp = np.abs(iq)
	phase = np.angle(iq) % np.pi
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
	assert len(coef) == 3, "Polyfit must be degree=3"
	c, b, a = coef
	p_c = np.pi/2 
	Q = (c + b*p_c + a*(p_c**2)) / (2*b + 4*a*p_c)
	return 0.9*Q # ! Off by some linear coefficient

if __name__ == '__main__':
	Q = 100
	errors = []
	# Repeat the test more than once to see how randomization within the noise causes error
	for _ in range(100):
		iq = createData(Q=Q, w0=100, noise=0.01, F0=2/Q)
		coef = fitAmpVsPhase(iq)
		q = getQ(coef)
		errors.append(np.abs(Q - q))

	print(f"Average error: {np.average(errors)}, standard deviation: {np.std(errors)}")