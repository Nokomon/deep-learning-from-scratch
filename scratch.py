import sys, os

sys.path.append(os.getcwd())
print(sys.path)


sys.path.append('..')
print(sys.path)
print(os.pardir)
sys.path.append(os.pardir)
print(sys.path)

print(sys.path[-1])
print(sys.path[-2])
print(sys.path[-1] == sys.path[-2])