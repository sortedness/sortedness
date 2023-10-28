from lange import ap

n = 5
a = 0
for i in ap[1, 2, ..., n]:
    for j in ap[i + 1, i + 2, ..., n]:
        c0 = (i - 1) * n + j
        print((n - 1) * n)
        for c in ap[c0, c0 + 1, ..., 19]:
            u, v = divmod(c, n)
            u, v = u + 1, v + 1
            if u < v:
                a += 1
                print(i, j, "     ", u, v)
        print()
print(a)
