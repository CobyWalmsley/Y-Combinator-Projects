import numpy as np
import matplotlib.pyplot as plt

values = [1, -1]             
probabilities = [0.56, 0.44]
array_size = 350
for n in range(20):
    random_array = np.random.choice(values, size=array_size, p=probabilities)
    list1 = []
    total = 1000
    for n in random_array:
        total = total+n
        list1.append(total)
    trend = np.array(list1)
    plt.plot(trend)
plt.show()