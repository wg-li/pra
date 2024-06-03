import torch
from filelock import FileLock
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from fix_seed import seed_torch

def dataset_creator(use_cuda, data_dir, seed, l_downld, batch_size):
    seed_torch(seed)
    kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {}
    with FileLock("./data.lock"):
        train_data = datasets.FashionMNIST(
                data_dir,
                train=True,
                download=l_downld,
                transform=transforms.Compose([
#                   transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
        test_data = datasets.FashionMNIST(
                data_dir,
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

#       train_size = int(0.6 * len(train_data))
#       valid_size = len(train_data) - train_size
#       train_dataset, valid_dataset = torch.utils.data.random_split(train_data,
#                                                                    [train_size, valid_size])
        train_dataset, valid_dataset = train_test_split(train_data, test_size=0.2, shuffle=True)
#       print(len(valid_dataset),type(valid_dataset))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **kwargs)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=256, # batch_size,
            shuffle=False,
            **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=256, #batch_size,
            shuffle=False,
            **kwargs)

    return train_loader, valid_loader, test_loader


