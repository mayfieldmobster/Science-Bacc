from numpy import float64
import models
import data_handler
import tensorflow.keras as keras
import tensorflow as tf
import time
import pickle
import os
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#reduce the console output

model = models.cifar10_model()

ds = data_handler.cifar10_dataset()

loss_func = keras.losses.CategoricalCrossentropy()

def loss(model, x, y):
    y_ = model(x)
    return loss_func(y, y_)

def grad(model, x, y):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, y)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

opt = tf.keras.optimizers.SGD(learning_rate=0.001)

loss_results = []
accuracy_results = []

EPOCHS = 100

batch_size = 16

b_ds = ds.batch(batch_size)

factor = 0


for epoch in range(EPOCHS):
    print("\n")
    print(f"----- Epoch  {epoch} -----", )
    print("Factor: ", round(factor))
    print("Batch size: ", batch_size)
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    
    start = time.time()
    
    for x,y in b_ds:
        loss_value, grads = grad(model, x, y)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        
        epoch_loss_avg.update_state(loss_value)
        #epoch_accuracy.update_state(y, model(x))
        
    loss_results.append([epoch_loss_avg.result(),time.time() - start])
    #accuracy_results.append(epoch_accuracy.result())
    
    print(f"Time For Epoch {epoch}: {time.time() - start}")
    print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result().numpy()))
    factor = math.exp(1)*(float64(epoch_loss_avg.result().numpy())**2)
    print()
    batch_size = int((2**10)/(2**round(factor)))
    if batch_size < 16:
        batch_size = 16
    b_ds = ds.batch(batch_size)

    
    

print()
#print(loss_results)

input("Press Enter to continue...")

with open(f'./data/{batch_size}.pickle', 'wb') as f:
    pickle.dump(loss_results, f)










