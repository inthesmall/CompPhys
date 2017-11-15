import math
import matplotlib.pyplot as plt
import numpy as np
import rand


r = rand.create_a()
y = rand.create_b()
y1 = rand.create_c()

# plt.figure(1)
plt.subplot(321)
plt.title("Uniform Deviate")
plt.xlabel("x")
plt.ylabel("Frequency")
plt.hist(r, bins=50)

plt.subplot(322)
plt.title("Uniform Deviate with expected value")
plt.xlabel("x")
plt.ylabel("Frequency")
plt.hist(r, bins=50)
plt.plot([0, 1], [2000, 2000], 'r-', label="Expected value")
plt.legend()
# plt.figure(2)

plt.subplot(323)
plt.title(r"$10^5$ Values distributed P=$\frac{1}{2}sin(x)$")
plt.xlabel("x")
plt.ylabel("Frequency")
plt.hist(y, bins=50)

plt.subplot(324)
plt.title(r"Normalized Values P=$\frac{1}{2}sin(x)$")
plt.xlabel("x")
plt.ylabel("Probability")
plt.hist(y, bins=50, normed=True)
x = np.linspace(0, math.pi, 300)
real = [0.5 * math.sin(i) for i in x]
plt.plot(x, real, 'r-', label="Expected value")
plt.legend()

# plt.figure(3)

plt.subplot(325)
plt.title(r"$10^5$ Values distributed P=$\frac{2}{\pi}sin^2(x)$")
plt.xlabel("x")
plt.ylabel("Frequency")
plt.hist(y1, bins=50)

plt.subplot(326)
plt.title(r"Normalized Values P=$\frac{2}{\pi}sin^2(x)$")
plt.xlabel("x")
plt.ylabel("Probability")
plt.hist(y1, bins=50, normed=True)
real1 = [(2 / math.pi) * math.sin(i)**2 for i in x]
plt.plot(x, real1, 'r-', label="Expected value")
plt.legend()

plt.subplots_adjust(wspace=0.2, hspace=0.8)
plt.show()
