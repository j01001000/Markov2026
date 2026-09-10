#
# acceptreject_foldnorm.py
#
# use the acceptance rejection method to sample
#

import time
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

# Acceptance-Rejection sampling
# folded normal f(x) ~ sqrt(2/pi)*exp(-x^2/2), x>=0
N = int(1e4)                        # number of samples
lam = 0.5                             # exponential rate
c = 4/np.e    # envelope constant (peak at x=2)

samples = np.zeros(N)               # storage
i = 0
proposal_count = 0
accepted_count = 0
start_time = time.perf_counter()

while i < N:
    proposal_count += 1
    # Proposal: exponential
    Y = -np.log(1-rng.random())/lam
    # Target density f(Y)
    fY = Y*np.exp(-Y)
    # Proposal density g(Y)
    gY = lam*np.exp(-lam*Y)
    # Acceptance test
    if rng.random() <= fY/(c*gY):
        samples[i] = Y
        i += 1
        accepted_count += 1

elapsed_time = time.perf_counter() - start_time
acceptance_rate = accepted_count / proposal_count
mean_time_per_accepted = elapsed_time / accepted_count
print(f"Acceptance rate: {acceptance_rate:.4f}")
print(f"1/c: {1/c:.4f}")
print(f"Mean time per accepted sample: {mean_time_per_accepted:.6f} seconds")

# Compare empirical histogram with true exponential
x = np.linspace(0, 5, 400)
f = x*np.exp(-x)

plt.figure(1, figsize=(8, 6))
plt.clf()
plt.hist(samples, bins='fd', density=True, label='Empirical pdf')
plt.plot(x, f, 'r-', linewidth=2, label='True exponential (lambda = %0.1f)' % lam)
plt.xlim([0, 5])
plt.grid(True)
plt.legend(fontsize=16)
plt.tick_params(labelsize=18)
plt.xlabel('$x$', fontsize=24)
plt.ylabel('$f(x)$', fontsize=24)
plt.tight_layout()
plt.show()