from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from pytorch_mlp import MLP

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None


def accuracy(predictions, labels):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        acc: scalar float, the accuracy of predictions.
    """
    predicted_classes = torch.argmax(predictions, dim=1)
    targets_classes = torch.argmax(labels, dim=1)
    correct = torch.sum(predicted_classes == targets_classes)
    acc = correct / predictions.shape[0]
    return acc


def train(data, dnn_hidden_units=DNN_HIDDEN_UNITS_DEFAULT,
          learning_rate=LEARNING_RATE_DEFAULT,
          max_steps=MAX_EPOCHS_DEFAULT,
          eval_freq=EVAL_FREQ_DEFAULT):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    torch.random.manual_seed(seed=42)

    # Load your data here
    train_x_ori, train_y_ori, test_x_ori, test_y_ori = data
    train_x = torch.from_numpy(train_x_ori).float()
    train_y = torch.from_numpy(train_y_ori).float()
    test_x = torch.from_numpy(test_x_ori).float()
    test_y = torch.from_numpy(test_y_ori).float()

    # Initialize your MLP model and loss function (CrossEntropy) here
    dnn_hidden_units = list(map(int, dnn_hidden_units.split(',')))
    model = MLP(n_inputs=2, n_hidden=dnn_hidden_units, n_classes=2)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    eval_steps = []
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    for step in range(max_steps):
        model.train()
        pred_y = model(train_x)
        loss = loss_fn(pred_y, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        if step % eval_freq == 0 or step == max_steps - 1:
            model.eval()
            train_acc = accuracy(pred_y, train_y)
            test_predictions = model.forward(test_x)
            test_acc = accuracy(test_predictions, test_y)

            eval_steps.append(step)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            test_loss = loss_fn(test_predictions, test_y).item()
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)

    return eval_steps, train_acc_list, test_acc_list, train_loss_list, test_loss_list
