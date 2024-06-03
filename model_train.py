import numpy as np
import torch
# import torch.nn as nn
import torch.nn.functional as F
import copy

def train_valid(model, device, train_loader, valid_loader, optimizer,
                epochs, intv_val, l_best, l_print):
    if l_best:
        best_acc = 0.0  # 记录模型测试时的最高准确率
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
#           print(data.shape, target.shape)
            optimizer.zero_grad()
            output = model(data)
            # loss = F.nll_loss(output, target)
            # loss_func = nn.CrossEntropyLoss()
            # loss = loss_func(output, target)
            loss = F.cross_entropy(output, target)
            train_loss += loss.item() * len(data)
            _, pred = torch.max(output.data, 1)
            correct += (pred == target).sum().item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader.dataset)
#       print('train data size',len(train_loader.dataset))
        train_dict = {
            "train loss": train_loss,
            "train accuracy": correct/len(train_loader.dataset)
        }
       
        if (epoch+1) % intv_val == 0: 
            model.eval()
            valid_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data) 
#                   print('output',output.size())
                    # sum up batch loss
#                   test_loss += F.nll_loss(
#                   output, target, reduction="sum").item()
                    valid_loss += F.cross_entropy(
                        output, target, reduction="sum").item()
                    _, pred = torch.max(output.data, 1)
                    correct += (pred == target).sum().item()
#           print('size',output.size(),target.size())
            valid_loss /= len(valid_loader.dataset)
            valid_dict = {
                "valid loss": valid_loss,
                "valid accuracy": correct/len(valid_loader.dataset)
            }
            if l_print:
                print("{} epoch, train results:\n{} ".format(epoch, {**train_dict, **valid_dict})) # 并行下，可用于检查hp和loss对应

            if l_best:
                if valid_dict['valid accuracy'] > best_acc:
                    best_acc = valid_dict['valid accuracy']
                    # print('which epoch: ', epoch)
                    best_results = {**train_dict, **valid_dict}
                    best_model_wts = copy.deepcopy(model.state_dict())


    results = {**train_dict, **valid_dict}
    if l_best:
        model.load_state_dict(best_model_wts)
        results = best_results
    return results

def train(model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        train_loss += loss.item() * len(data)
        _, pred = torch.max(output.data, 1)
        correct += (pred == target).sum().item()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader.dataset)
    return {
        "train loss": train_loss,
        "train accuracy": correct/len(train_loader.dataset)
    }

def valid(model, device, valid_loader):
    l_mse = 0
    model.eval()
    valid_loss = 0
    if l_mse:
        valid_mseloss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            valid_loss += F.cross_entropy(
                output, target, reduction="sum").item()
            if l_mse:
                tg = torch.eye(output.size()[1])[target, :].to(device)
                valid_mseloss += F.mse_loss(
                    F.softmax(output, dim=1), tg, reduction="sum").item()
            _, pred = torch.max(output.data, 1)
            correct += (pred == target).sum().item()
    valid_loss /= len(valid_loader.dataset)
    if l_mse:
        valid_mseloss /= len(valid_loader.dataset)
        valid_dict = {
            "valid loss": np.round(valid_loss, 8),
            "valid accuracy": np.round(100.*correct/len(valid_loader.dataset), 3),
            "mse loss": np.round(valid_mseloss, 8)
        }
    else:
        valid_dict = {
            "valid loss": valid_loss,
            "valid accuracy": correct / len(valid_loader.dataset)
        }
    return valid_dict

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output.data, 1)
            correct += (pred == target).sum().item()
    return {"test accuracy": correct/len(test_loader.dataset)}

