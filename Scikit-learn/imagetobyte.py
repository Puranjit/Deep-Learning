import numpy as np

with open("puran.jpg", "rb") as image:
    f = image.read()
    b = np.array(bytearray(f))
print(b.shape)
print(b[1:100])