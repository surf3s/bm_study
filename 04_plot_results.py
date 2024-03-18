import matplotlib.pyplot as plt
import numpy as np
import os

# Load the history from the file
results_path = "/home/shannon/local/Source/Python/bm_study/results"
history = np.load(os.path.join(results_path, 'bm_history.npy'), allow_pickle=True).item()

acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
