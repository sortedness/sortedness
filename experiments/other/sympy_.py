import sympy as sy
sy.init_printing(use_unicode=True)
X, X_, r, r_ = sy.IndexedBase("X"), sy.IndexedBase("X_"), sy.IndexedBase("r"), sy.IndexedBase("r_")
i, j, n = sy.symbols("i j n", cls=sy.Idx)
λ = sy.symbols("\\lambda")

s = sy.Function("s") #lambda v: 1 - 2 / (1 + sy.exp(2 * v / λ))
sum = lambda v: sy.Sum(sy.Sum(v, (i, 1, n)), (j, i + 1, n))
f = sum(s(r[i] - r[j]) * s(r_[i] - r_[j])) / sy.sqrt(sum(s(r[i] - r[j]) ** 2) * sum(s(r_[i] - r_[j]) ** 2))
f = sy.simplify(f)
expr = f.diff(r[i])
# expr = f.diff(r[j])
# out = sy.simplify(expr)
out = expr
print(out)
sy.print_latex(out)
sy.pretty_print(out)
