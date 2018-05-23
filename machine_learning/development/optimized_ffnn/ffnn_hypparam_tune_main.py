from machine_learning.development.optimized_ffnn import ffnn_hyperparam_tune as hpt
import matplotlib.pyplot as plt
import time

#start time
start_time = time.time()

#intial hyperparameters
stock = '^GSPC'
start_date = '1950-01-01'
end_date = '2017-12-31'
window = 5
future_gap = 5
split = 0.8
dropout = None
neurons = [64, 64, 32, 1]
batch_size = 4026 
epochs = 1
validation_split = 0.1
verbose = 1

#optimal hyperparameters txt file
print("\n> finding the optimal hyperparameters...")
file = open("machine_learning/optimized_ffnn/ffnn_optimal_hyperparameters.txt", "wb") #ab+ to read and append to file
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig2, (ax4, ax5) = plt.subplots(2, 1)

#finding the optimal dropout
print("\n> finding the optimal dropout...")
dropout_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
dropout_result = hpt.optimal_dropout(stock, start_date, end_date, window, future_gap, split, neurons,
                                     batch_size, epochs, validation_split, verbose, dropout_list)

min_loss = min(dropout_result.values())
optimal_dropout = -1.0
for dout, loss in dropout_result.items():
    if loss == min_loss:
        optimal_dropout = dout

file.write(bytes("dropout: %.1f, " %(optimal_dropout), 'UTF-8'))
print("\nDropout:", optimal_dropout)
dropout = optimal_dropout

items = dropout_result.items()
x, y = zip(*items)
ax1.plot(x, y)
ax1.set_xlabel('Dropout')
ax1.set_ylabel('MSE')
ax1.grid(True)

#finding the optimal neurons
print("\n> finding the optimal neurons...")
neuronlist1 = [64, 128, 256]
neuronlist2 = [16, 32, 64]
neurons_result = hpt.optimal_neurons(stock, start_date, end_date, window, future_gap, split, dropout, 
                                     batch_size, epochs, validation_split, verbose, neuronlist1, neuronlist2)

min_loss = min(neurons_result.values())
optimal_neurons = ""
for n, loss in neurons_result.items():
    if loss == min_loss:
        optimal_neurons = n

file.write(bytes("neurons: %s, " %(str(optimal_neurons)), 'UTF-8'))
print("\nNeurons:", optimal_neurons)
neurons = optimal_neurons
neurons = neurons[1:-1]
neurons = neurons.split(", ")
neurons = [int(neuron_str) for neuron_str in neurons]

items = neurons_result.items()
x, y = zip(*items)
ax2.bar(range(len(items)), y, align='center')
plt.sca(ax2)
plt.xticks(range(len(items)), x, rotation=25)
ax2.set_xlabel('Neurons')
ax2.set_ylabel('MSE')
ax2.grid(True)

#finding the optimal decay
print("\n> finding the optimal decay...")
decay_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
decay_result = hpt.optimal_decay(stock, start_date, end_date, window, future_gap, split, dropout, 
                                  neurons, batch_size, epochs, validation_split, verbose, decay_list)

min_loss = min(decay_result.values())
optimal_decay = -1.0
for d, loss in decay_result.items():
    if loss == min_loss:
        optimal_decay = d

file.write(bytes("decay: %.1f, " %(optimal_decay), 'UTF-8'))
print("\nDecay:", optimal_decay)
decay = optimal_decay

items = decay_result.items()
x, y = zip(*items)
ax3.plot(x, y)
ax3.set_xlabel('Decay')
ax3.set_ylabel('MSE')
ax3.grid(True)

#finding the optimal batch size
print("\n> finding the optimal batch size...")
batch_size_list = [128, 256, 512, 1024, 2048, 4096]
batch_size_result = hpt.optimal_batch_size(stock, start_date, end_date, window, future_gap, split, dropout,
                                         neurons, epochs, validation_split, verbose, decay, batch_size_list)

min_loss = min(batch_size_result.values())
optimal_batch_size = -1
for bs, loss in batch_size_result.items():
    if loss == min_loss:
        optimal_batch_size = bs

file.write(bytes("batch_size: %d, " %(optimal_batch_size), 'UTF-8'))
print("\nBatch Size:", optimal_batch_size)
batch_size = optimal_batch_size

items = batch_size_result.items()
x, y = zip(*items)
ax4.plot(x, y)
ax4.set_xlabel('Batch Size')
ax4.set_ylabel('MSE')
ax4.grid(True)

#finding the optimal epochs
print("\n> finding the optimal epochs...")
epochs_list = [50, 60, 70, 80, 90, 100, 200, 300]
epochs_result = hpt.optimal_epochs(stock, start_date, end_date, window, future_gap, split, dropout, 
                                    neurons, batch_size, validation_split, verbose, epochs_list)

min_loss = min(epochs_result.values())
optimal_epochs = -1
for ep, loss in epochs_result.items():
    if loss == min_loss:
        optimal_epochs = ep

file.write(bytes("epochs: %d, " %(optimal_epochs), 'UTF-8'))
print("\nEpochs:", optimal_epochs)
epochs = optimal_epochs

items = epochs_result.items()
x, y = zip(*items)
ax5.plot(x, y)
ax5.set_xlabel('Epochs')
ax5.set_ylabel('MSE')
ax5.grid(True)

#end time
end_time = time.time()
time = end_time - start_time
file.write(bytes("time elapsed: %.3fs." %(time), 'UTF-8'))

#closing the file and showing the plot
print("\nOptimal Hyperparameters")
print("Dropout:", optimal_dropout)
print("Neurons:", optimal_neurons)
print("Decay:", optimal_decay)
print("Batch Size:", optimal_batch_size)
print("Epochs:", optimal_epochs)
print("Time Elapsed (s):", time)

file.close()
fig1.tight_layout()
fig2.tight_layout()
plt.show()