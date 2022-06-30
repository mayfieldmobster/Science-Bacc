import pickle
from numpy import float64
import statistics
import matplotlib.pyplot as plt

batch_size = 16

with open(f'./data/{batch_size}.pickle', 'rb') as f:
    loss_results = pickle.load(f)



plt.plot([float64(loss_results[i][0].numpy())for i in range(len(loss_results))])
plt.show()

"""
for i in loss_results:
    print(float64(i[0].numpy()))
"""
for i in range(len(loss_results)):
    print(f"{i+1}: ", -(float64(loss_results[i][0].numpy()) - float64(loss_results[i-1][0].numpy())))

quit()
num = 50

first = [float64(loss_results[i][0].numpy()) for i in range(num)]
last = [float64(loss_results[-(i+1)][0].numpy()) for i in range(num)]
last = last[::-1]
    
#print(last_20)
print(f"Batch Size of {batch_size} first {num}: ", -1*statistics.mean([first[i+1]-first[i] for i in range(len(first)-1)]))
print(f"Batch Size of {batch_size} last {num}:  ", -1*statistics.mean([last[i+1]-last[i] for i in range(len(last)-1)]))
