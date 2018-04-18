from machine_learning.flagship import hyperparamter_tunning as hpt
import matplotlib.pyplot as plt

stock = '^GSPC'
start_date = '1950-01-01'
end_date = '2017-12-31'
future_gap = 1
time_steps = 20
split = 0.9
dropout = None
neurons = [128, 128, 32, 1]
batch_size = 512 
epochs = 300
validation_split= 0.1
verbose = 1

#optimal hyperparameters txt file
file = open("optimal_hyperparameters.txt", "wb") #ab+ to read and append to file
_, (ax1, ax2, ax3) = plt.subplots(3, 1)
_, (ax4, ax5, ax6) = plt.subplots(3, 1)

#finding the optimal dropout
dropout_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
dropout_result = hpt.optimal_dropout(stock, start_date, end_date, future_gap, time_steps, split, neurons,
                                     batch_size, epochs, validation_split, verbose, dropout_list)

min_loss = min(dropout_result.values())
optimal_dropout = -1.0
for dout, loss in dropout_result.items():
    if loss == min_loss:
        optimal_dropout = dout

file.write(bytes("dropout: %f\n" %(optimal_dropout), 'UTF-8'))
dropout = optimal_dropout

items = dropout_result.items()
x, y = zip(*items)
ax1.plot(x, y)
ax1.set_xlabel('Dropout')
ax1.set_ylabel('MSE')
ax1.grid(True)

#finding the optimal epochs
epochs_list = [50, 60, 70, 80, 90, 100, 200, 300]
epochs_result = hpt.optimal_dropout(stock, start_date, end_date, future_gap, time_steps, split, dropout, 
                                    neurons, batch_size, validation_split, verbose, epochs_list)

min_loss = min(epochs_result.values())
optimal_epochs = -1
for ep, loss in epochs_result.items():
    if loss == min_loss:
        optimal_epochs = ep

file.write(bytes("epochs: %f\n" %(optimal_epochs), 'UTF-8'))
epochs = optimal_epochs

items = epochs_result.items()
x, y = zip(*items)
ax2.plot(x, y)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('MSE')
ax2.grid(True)

#finding the optimal neurons
neuronlist1 = [32, 64, 128, 256, 512]
neuronlist2 = [16, 32, 64]
neurons_result = hpt.optimal_neurons(stock, start_date, end_date, future_gap, time_steps, split, dropout, 
                                     batch_size, epochs, validation_split, verbose, neuronlist1, neuronlist2)

min_loss = min(neurons_result.values())
optimal_neurons = ""
for n, loss in neurons_result.items():
    if loss == min_loss:
        optimal_neurons = n

file.write(bytes("neurons: %f\n" %(optimal_neurons), 'UTF-8'))
neurons = optimal_neurons
neurons = neurons[1:-1]
neurons = neurons.split(", ")
neurons = [int(neuron_str) for neuron_str in neurons]

items = neurons_result.items()
x, y = zip(*items)
ax3.bar(range(len(items), y, align='center'))
ax3.xticks(range(len(items)), x)
ax3.xticks(rotation=90)
ax3.set_xlabel('Neurons')
ax3.set_ylabel('MSE')
ax3.grid(True)

#finding the optimal decay
decay_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
decay_result = hpt.optimal_decay(stock, start_date, end_date, future_gap, time_steps, split, dropout, 
                                  neurons, batch_size, epochs, validation_split, verbose, decay_list)

min_loss = min(decay_result.values())
optimal_decay = -1.0
for d, loss in decay_result.items():
    if loss == min_loss:
        optimal_decay = d

file.write(bytes("decay: %f\n" %(optimal_decay), 'UTF-8'))
decay = optimal_decay

items = decay_result.items()
x, y = zip(*items)
ax4.plot(x, y)
ax4.set_xlabel('Decay')
ax4.set_ylabel('MSE')
ax4.grid(True)

#finding the optimal time steps
time_steps_list = [5, 10, 15, 20, 40, 80, 100]
time_steps_result = hpt.optimal_time_steps(stock, start_date, end_date, future_gap, split, dropout, neurons,
                                         batch_size, epochs, validation_split, verbose, decay, time_steps_list)

min_loss = min(time_steps_result.values())
optimal_time_steps = -1
for ts, loss in time_steps_result.items():
    if loss == min_loss:
        optimal_time_steps = ts

file.write(bytes("time_steps: %f\n" %(optimal_time_steps), 'UTF-8'))
time_steps = optimal_time_steps

items = time_steps_result.items()
x, y = zip(*items)
ax5.plot(x, y)
ax5.set_xlabel('Time Steps')
ax5.set_ylabel('MSE')
ax5.grid(True)

#finding the optimal batch size
batch_size_list = [128, 256, 512, 1024, 2048, 4096]
batch_size_result = hpt.optimal_batch_size(stock, start_date, end_date, future_gap, time_steps, split, dropout,
                                         neurons, epochs, validation_split, verbose, decay, batch_size_list)

min_loss = min(batch_size_result.values())
optimal_batch_size = -1
for bs, loss in batch_size_result.items():
    if loss == min_loss:
        optimal_batch_size = bs

file.write(bytes("batch_size: %f\n" %(optimal_batch_size), 'UTF-8'))
batch_size = optimal_batch_size

items = batch_size_result.items()
x, y = zip(*items)
ax6.plot(x, y)
ax6.set_xlabel('Batch Size')
ax6.set_ylabel('MSE')
ax6.grid(True)

file.close()

plt.show()