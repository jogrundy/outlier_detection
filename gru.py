#GRU coded up to look for outliers.

import os
from time import time
import platform

import numpy as np
import pandas as pd


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import gc

#started with code from https://blog.floydhub.com/gru-with-pytorch/
#then massively refactored.



class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

def train(train_loader, learn_rate, device, batch_size, in_dim, out_dim, hidden_dim=256, EPOCHS=5, model_type="GRU"):

    # next(iter(train_loader))[0].shape[2]
    # Setting common hyperparameters
    input_dim = in_dim #next(iter(train_loader))[0].shape[2]
    output_dim = out_dim #next(iter(train_loader))[0].shape[2]
    n_layers = 2

    # print(next(iter(train_loader))[0].shape)
    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device)
    else:
        model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers, device)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    # raise
    model.train()
    # print("Starting Training of {} model".format(model_type))
    epoch_times = []
    # Start training loop
    for epoch in range(1,EPOCHS+1):
        start_time = time()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            # if counter%200 == 0:
            #     print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
        t1 = time()-start_time
        # print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)))
        # print("Total Time Elapsed: {}m {}s".format(int(t1//60), int(t1%60)))
        epoch_times.append(t1)
    if device =='cuda':
        gc.collect()
        torch.cuda.empty_cache()
    tt = sum(epoch_times)
    # print("Total Training Time: {}m {}s".format(int(tt//60), int(tt%60) ))
    return model

def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time()
    for i in test_x.keys():
        inp = torch.from_numpy(np.array(test_x[i]))
        labs = torch.from_numpy(np.array(test_y[i]))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
    t1 = time() - start_time
    print("Evaluation Time: {}m {}s".format(int(t1//60), int(t1%60)))
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i]-targets[i])/(targets[i]+outputs[i])/2)/len(outputs)
    print("sMAPE: {}%".format(sMAPE*100))
    return outputs, targets, sMAPE

def model_eval(model, data_loader, targets, scaler, device):
    """
    uses MSE to evaluate model
    """
    model.eval()
    model.cpu()

    batch_size, tw, p =next(iter(data_loader))[0].shape
    # n,tw,p = inputs.shape
    # print(n,tw, p)
    t0 = time()
    preds = []
    scores = []
    targets = []
    with torch.no_grad():
        for input, label in data_loader:

            # print(inputs.shape, targets.shape)
            h = model.init_hidden(batch_size)

            out, _ = model(input.float(), h)
            if scaler:

                pred = scaler.inverse_transform(out.detach().numpy())
                label = scaler.inverse_transform(label)
                score = ese(pred, label)
            else:
                pred = out.detach().numpy()
                label = label.detach().numpy()
                # print(pred, label)
                score = ese(pred, label)
            # print(score.shape)
            # print(scores.shape)
            # print(pred.shape)
            # print(preds.shape)
            # print(label.shape)
            # print(targets.shape)
            scores.append(score)
            preds.append(pred)
            targets.append(label)


    return preds, targets, scores

# def model_eval(model, inputs, targets, scaler, device, eval_metric=nn.MSELoss):
#     """
#     uses MSE to evaluate model
#     """
#     model.eval()
#     t0 = time()
#     input = torch.from_numpy(inputs)
#     targets = torch.from_numpy(targets)
#     # print(inputs.shape, targets.shape)
#     h = model.init_hidden(input.shape[0])
#     out, h = model(input.cpu().float(), h)
#     preds = scaler.inverse_transform(out.cpu().detach().numpy())
#     targets = scaler.inverse_transform(targets)
#     # print(preds.shape, targets.shape)
#     score = eval_metric(preds, targets)
#     return preds, targets, score

def compile_data(data, tw, pad=False, test_split = 0.2):
    """
    assumes data is n,p numpy matrix.
    produces set of serieses with or without padding
    """
    n,p = data.shape


    # print(data.shape)

    inputs = np.zeros((n,tw,p))
    targets = np.zeros((n, p))
    if pad:
        data = np.concatenate([np.zeros((tw,p)), data])
    for i in range(tw, n+tw):
        inputs[i-tw] = data[i-tw:i, :]
        targets[i-tw] = data[i,:]
    inputs = inputs.reshape(-1,tw,p)
    targets = targets.reshape(-1,p)
    # print(inputs.shape)
    # print(targets.shape)

    # Split data into train/test portions and combining all data from different files into a single array
    test_ind = int(test_split*len(inputs))
    # print(test_ind)

    train_x = inputs[:-test_ind, :, :]
    train_y = targets[:-test_ind, :]

    test_x = inputs[-test_ind:]
    test_y = targets[-test_ind:]

    return train_x, train_y, inputs, targets

def ese(pred, target):
    """
    takes in predicted values and actual values, returns elementwise squared error
    via (x-y)^2
    """
    # print(pred.shape, target.shape)
    errs = (np.subtract(pred, target))**2
    # print(errs)
    errs = np.sum(errs, axis=1)
    # print(errs.shape)
    return np.sqrt(errs)

def get_GRU_os(X):
    n,p = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if int(n//4) > 16:
        tw = 16
    else:
        tw = int(n//8)
    test_p = .2

    if n<8:
        batch_size=1
    elif n<16:
        batch_size=4
    elif n<64:
        batch_size=16
    elif n<128:
        batch_size=32
    else:
        batch_size = 128

    # # Scaling the input data
    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)
    scaler = None

    train_x, train_y, inputs, targets = compile_data(X, tw, pad=True, test_split = test_p)
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    data = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    data_loader = DataLoader(data, shuffle=False, batch_size=1, drop_last=True)
    lr = 0.001
    gru_model = train(train_loader, lr, device, batch_size, p, p, hidden_dim=64, EPOCHS=20, model_type="GRU")
    gru_preds, targets_scaled, score = model_eval(gru_model, data_loader, targets, None, device)

    # errs = ese(gru_preds, targets)
    # return gru_preds, targets_scaled, targets, score
    return score

def get_LSTM_os(X):
    n,p = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if int(n//4) > 16:
        tw = 16
    else:
        tw = int(n//8)
    test_p = 0.2

    if n<8:
        batch_size=1
    elif n<16:
        batch_size=4
    elif n<64:
        batch_size=16
    elif n<128:
        batch_size=32
    else:
        batch_size = 128
    # Scaling the input data
    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)
    scaler = None
    # print(batch_size, tw)
    in_dim = p
    out_dim = p


    train_x, train_y, inputs, targets = compile_data(X, tw, pad=True, test_split = test_p)
    # print(train_x.shape, train_y.shape)
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    data = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    data_loader = DataLoader(data, shuffle=False, batch_size=1, drop_last=True)
    lr = 0.001
    lstm_model = train(train_loader, lr, device, batch_size, in_dim, out_dim,
                        hidden_dim=64, EPOCHS=20, model_type="LSTM")
    lstm_preds, targets_scaled, score = model_eval(lstm_model, data_loader, targets, scaler, device)
    # errs = ese(lstm_preds, targets)
    # return lstm_preds, targets_scaled, targets, score
    return score

# def get_LSTM_os(X):
#     n,p = X.shape
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if int(n//4) > 64:
#         tw = 64
#     else:
#         tw = int(n//4)
#     test_p = 0.75
#
#     if n<8:
#         batch_size=2
#     elif n<16:
#         batch_size=4
#     elif n<64:
#         batch_size=16
#     elif n<128:
#         batch_size=32
#     else:
#         batch_size = 128
#     # Scaling the input data
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)
#
#     train_x, train_y, inputs, targets = compile_data(X, tw, pad=True, test_split = test_p)
#     train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
#     data = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
#     train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
#     data_loader = DataLoader(data, shuffle=True, batch_size=1, drop_last=True)
#     lr = 0.001
#     lstm_model = train(train_loader, lr, device, batch_size, EPOCHS=20, model_type="LSTM")
#     lstm_preds, targets, score = model_eval(lstm_model, data_loader, targets, scaler, device)
#     # errs = ese(lstm_preds, targets)
#     return score

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from test_data import generate_test
    sys = platform.system()
    # Define data root directory
    print("sys = {}".format(sys))
    if sys == "Windows":
        data_dir = 'H:\Data\energy_ts'
    elif sys == "Darwin":
        data_dir = "/Users/jojo/Data/energy_ts"
    else:
        raise

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print('Device = {}'.format(device))

    # ta=4 #simple sine wave
    # n = 100
    # p = 1 #single dimension
    # r = 5
    # p_frac = 0.5
    # p_quant = 0.5
    # gamma = 0.05
    # tw = 20
    # data, outs = generate_test(n,p,r, p_frac, p_quant,gamma, ta)

    n=100
    x=np.arange(n)
    # x=x.reshape(-1,2)
    y = np.sin(x/np.pi).reshape(-1,1)
    outs = [50,60,70,75]
    for out in outs:
        s = np.sign(np.random.rand()-0.5)
        y[out] = y[out] +(0.5*s)
    # scaler = MinMaxScaler()
    # y = scaler.fit_transform(y)
    # train_x, train_y, inputs, targets = compile_data(y, 3, pad=True, test_split = 0.2)
    # print(inputs[:4], targets[:4])
    # print(y[:4])
    print(x.shape, y.shape)
    plt.plot(x, y)
    for out in outs:
        plt.plot(out, y[out], 'ro')
    plt.show()
    lstm_preds, targets_scaled, targets,os = get_LSTM_os(y)
    lstm_preds = np.array(lstm_preds).reshape(-1)
    targets_scaled = np.array(targets_scaled).reshape(-1)
    plt.plot(x, lstm_preds, 'g-')
    plt.plot(x, y,'b-')
    plt.plot(x,targets_scaled,'m-')
    plt.plot(x,targets,'c-')
    plt.title('LSTM testing')
    plt.show()

    gru_preds, targets_scaled, targets,os = get_GRU_os(y)
    gru_preds = np.array(gru_preds).reshape(-1)
    targets_scaled = np.array(targets_scaled).reshape(-1)
    plt.plot(x, gru_preds, 'g-')
    plt.plot(x, y,'b-')
    plt.plot(x,targets_scaled,'m-')
    plt.plot(x,targets,'c-')
    plt.title('GRU testing')
    plt.show()




    #
    # plt.figure()
    # plt.plot(x, os)
    # for out in outs:
    #     ymin = np.min(os)
    #     ymax = np.max(os)
    #     plt.vlines(out, ymin, ymax, colors='r')
    # plt.show()
