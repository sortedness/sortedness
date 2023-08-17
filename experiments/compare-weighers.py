from math import pi

from scipy.stats import cauchy

from sortedness.local import hyperbolic, cauchy as ca

n = 38
# print(cauchy(1).pdf(n))
# print(multivariate_normal(0, 1).pdf(n))
# print(genhyperbolic(0, 1, 0).pdf(n))
s = 0
c = cauchy(0).cdf
od = 0
old = 0
for i in range(1000):
    d = abs(c(i) - old)
    old = c(i)
    if i in [50, 89, 999]:
        print()
    m = 3 * pi * ca(i, 3)
    if i < 15 or 95 < i < 100 or i == 999:
        # print(f"{i}\t{3 * pi * ca(i, 3):.5f}\t{hyperbolic(i):.5f}\t\t{2 * d:.15f}")
        print(f"{i}\t{m:.5f}\t{hyperbolic(i):.5f}\t\t{2 * d:.15f}")
    if 0.00099 < m < 0.00101:
        print(f"{i}\t{m:.5f}\t{hyperbolic(i):.5f}\t\t{2 * d:.15f}")
        print()
    if m < 0.0001:
        print(f"{i}\t{m:.5f}\t{hyperbolic(i):.5f}\t\t{2 * d:.15f}")
        break
    # print(hyperbolic(i))
    # print(genhyperbolic(1, 1, 0).pdf(i))
    # print(multivariate_normal(0, 1).pdf(i))
    # print()
