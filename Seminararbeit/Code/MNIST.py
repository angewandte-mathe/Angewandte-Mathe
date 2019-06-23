import mnist
import time
import numpy as np
import NeuronalesNetz

"""

__author__ = "Bastian Kersting"
__version__ = "1.2"

"""

start_time = time.time()
overll_performance = []
round = 1

for _ in range(5):

    input_nodes = 784
    output_nodes = 10
    # 2/3 * input_nodes?
    hidden_nodes = 517
    hidden_layers = 5
    lr = 0.00125
    activation_function = "sigmoid"

    network = NeuronalesNetz.NeuronalesNetz(input_nodes, output_nodes, hidden_nodes, hidden_layers, lr, activation_function)

    image_train = mnist.train_images()
    label_train = mnist.train_labels()

    for index in range(len(image_train)):
        current_target = np.zeros(10) + 0.1
        current_target[label_train[index]] = 0.99
        current_input = []
        for index_w in range(28):
            for index_h in range(28):
                current_input.append((image_train[0:60000][index][index_w][index_h] / 255.0 * 0.99) + 0.01)
        network.train(current_input, current_target)

    # Testing

    scorecard = []
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()
    for index in range(len(test_images)):
        current_test = []
        for index_w in range(28):
            for index_h in range(28):
                current_test.append((test_images[0:10000][index][index_w][index_h] / 255.0 * 0.99) + 0.01)

        res = network.forward_propagation(current_test)

        if np.argmax(res) == test_labels[index]:
            scorecard.append(1.0)
        else:
            scorecard.append(0.0)

    performance = float(np.asarray(scorecard).sum()) / float(len(scorecard))
    overll_performance.append(performance)
    print(round, " round(s) finished")
    print("This rounds performance= ", performance)
    round += 1

end_time = time.time()

the_real_performance = np.asarray(overll_performance).sum() / float(len(overll_performance))
print("%.10f minutes" % ((end_time - start_time) / 60))
print("My network's performance (%): ", the_real_performance * 100)
