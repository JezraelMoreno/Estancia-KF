import numpy as np
import math
import time
from scipy import linalg as lin
import givensMethod as orig
import givensMod as opt

np.random.seed(0)

samplingRate = 128
numberSensors = 14

def calculateF_Taylor(samplingRate, numSensors):
    c = np.zeros([numSensors, numSensors])
    for i in range(numSensors):
        v = (1/samplingRate**i)/math.factorial(i)
        a = np.empty(numSensors)
        a.fill(v)
        np.fill_diagonal(c[i:], a)
    return c.T

F = calculateF_Taylor(samplingRate, numberSensors)
Q = np.eye(numberSensors)

print(f"{'Caso':<12}{'Match':<10}{'Diff max':<12}{'t_orig (ms)':<14}{'t_opt (ms)':<14}{'Speedup':<10}")
print("-" * 68)

reps = 500

for trial in range(10):
    # simula una senal EEG y calcula su covarianza inicial
    simulated_signal = np.random.randn(numberSensors, samplingRate)
    P = np.cov(simulated_signal)

    newP = (P + P.T) / 2
    lu, d, perm = lin.ldl(newP, lower=1)
    L = lu.dot(lin.fractional_matrix_power(d, 0.5))

    t0 = time.perf_counter()
    for _ in range(reps):
        r_orig = orig.givensMethod(F, Q, L)
    t_orig = (time.perf_counter() - t0) / reps * 1000

    t0 = time.perf_counter()
    for _ in range(reps):
        r_opt = opt.givensMethod(F, Q, L)
    t_opt = (time.perf_counter() - t0) / reps * 1000

    ok = np.allclose(r_orig, r_opt, atol=1e-8)
    diff = np.max(np.abs(r_orig - r_opt))
    speedup = t_orig / t_opt if t_opt > 0 else float('inf')

    print(f"{trial:<12}{str(ok):<10}{diff:<12.2e}{t_orig:<14.4f}{t_opt:<14.4f}{speedup:<10.2f}")