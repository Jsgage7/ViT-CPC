import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import datetime
import numpy as np

def train_supervised_baseline(model, optimizer, criterion, train_loader, val_loader, epochs, device):
    print("Beginning Training")
    for epoch in range(epochs):
        epoch_start = datetime.datetime.now()
        epoch_loss = 0
        epoch_accuracy = 0
        for data in train_loader:
            images, labels = data
            images.to(device)
            labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            guess = out.argmax(1)
            accuracy = (guess==labels).sum().item() /  labels.size(0)
            epoch_loss += loss.item()
            epoch_accuracy += accuracy
        epoch_end = datetime.datetime.now()
        #print(f'Epoch {epoch}: Loss = {epoch_loss/len(train_loader)} Accuracy = {epoch_accuracy/len(train_loader)}, Time = {epoch_end - epoch_start}')

        val_accuracy = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images.to(device)
                labels.to(device)
                out = model(images)
                guess = out.argmax(1)
                accuracy = (guess==labels).sum().item() /  labels.size(0)
                val_accuracy += accuracy
        #print(f'Epoch {epoch}: Accuracy = {val_accuracy/len(val_loader)}')
    return val_accuracy/len(val_loader)


# made separate function for this since no patch transformations needed
def setup_train_test_data(batch_size, dataset_pct=1.0, val_frac=0.2):
    data_norm = {
        "mean": [0.49139968, 0.48215827, 0.44653124],
        "std": [0.24703233, 0.24348505, 0.26158768]
    }

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=data_norm["mean"], std=data_norm["std"])])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=data_norm["mean"], std=data_norm["std"])])

    training_dataset = datasets.CIFAR10('../data/', download=True, transform=train_transform, train=True)
    test_dataset = datasets.CIFAR10('../data/', download=True, transform=test_transform, train=False)

    N = len(training_dataset)
    num_train_samples = int(N * dataset_pct)
    dataset_indices = np.random.choice(N, num_train_samples, replace=False)
    dataset_subset = torch.utils.data.Subset(training_dataset, dataset_indices)

    # split into train/val
    N_subset = len(dataset_subset)
    V = int(num_train_samples * val_frac)
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_subset, [N_subset - V, V])

    training_loader= torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(dataset_val, batch_size = batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return training_loader, validation_loader, test_loader


if __name__ == '__main__':
    device= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    learning_rates = [1e-3]
    percentages = [1.0]
    resnet = [50]
    for learning_rate in learning_rates:
        for percentage in percentages:
            for res in resnet:
                random_seed = 15009
                torch.manual_seed(random_seed)
                np.random.seed(random_seed)

                batch_size = 64
                epochs = 20

                #using a resnet from torchvision
                if res == 50:
                    model = models.resnet50(weights=None)
                else:
                    model = models.resnet34(weights=None)
                optimizer = torch.optim.Adam(model.parameters(), learning_rate)
                criterion = nn.CrossEntropyLoss()

                training_data, validation_data, test_data = setup_train_test_data(batch_size, dataset_pct=percentage)
                acc = train_supervised_baseline(model, optimizer, criterion, training_data, validation_data, epochs, device)
                print(f"ACC for lr {learning_rate}, percentage {percentage}, resent{res} = {acc}")
                
