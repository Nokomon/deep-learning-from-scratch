a = list(map(int, "1 2 3 4 5".split()))
b = a
a[0] = 100
print(a == b)
print(a is b)
print(a[0] is b[0])