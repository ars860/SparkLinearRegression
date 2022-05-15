# Simple script to generate data for tests

import numpy as np

def f(x):
    return x.dot(np.array([1, 2])) + 3

points = [np.random.rand(2) * 100 for _ in range(100)]
xys = list(map(lambda p: (p, f(p)), points))

def printArray(arr):
    return ", ".join(map(str, arr))

result =  ",\n".join(map(lambda xy: f"    (Vectors.dense({printArray(xy[0])}), Vectors.dense({xy[1]}))", xys))
result = f"  lazy val _vectors = Seq(\n{result}\n  )"
with open("tmp.txt", "w+") as f:
    f.write(result)