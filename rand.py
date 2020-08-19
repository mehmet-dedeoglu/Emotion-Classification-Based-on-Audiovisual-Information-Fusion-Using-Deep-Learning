import random
import numpy as np
numbers = np.zeros([32])
k = 0
while k < 32:
    number = random.randint(1, 197)
    k = k+1
    if number in numbers:
        k = k-1
    else:
        numbers[k-1] = number

numbers = np.sort(numbers)
print(numbers)