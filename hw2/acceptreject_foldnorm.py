#
# acceptreject_foldnorm.py
#
# use the acceptance rejection method to sample from a folded normal
#

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

# Acceptance-Rejection sampling
# folded normal f(x) ~ sqrt(2/pi)*exp(-x^2/2), x>=0
N = int(1e5)                        # number of samples
lam = 1                             # exponential rate
c = np.sqrt(2/np.pi)*np.exp(0.5)    # envelope constant (peak at x=1)

samples = np.zeros(N)               # storage
i = 0
while i < N:
    # Proposal: exponential
    Y = -np.log(rng.random())/lam
    # Target density f(Y)
    fY = np.sqrt(2/np.pi)*np.exp(-Y**2/2)
    # Proposal density g(Y)
    gY = lam*np.exp(-lam*Y)
    # Acceptance test
    if rng.random() <= fY/(c*gY):
        samples[i] = Y
        i += 1

# Compare empirical histogram with true folded Gaussian
x = np.linspace(0, 5, 400)
f = np.sqrt(2/np.pi)*np.exp(-x**2/2)

plt.figure(1, figsize=(8, 6))
plt.clf()
plt.hist(samples, bins='fd', density=True, label='Empirical pdf')
plt.plot(x, f, 'r-', linewidth=2, label='True folded Gaussian')
plt.xlim([0, 5])
plt.grid(True)
plt.legend(fontsize=16)
plt.tick_params(labelsize=18)
plt.xlabel('$x$', fontsize=24)
plt.ylabel('$f(x)$', fontsize=24)
plt.tight_layout()
plt.show()
