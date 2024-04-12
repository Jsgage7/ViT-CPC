import os
import datetime
import time

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type='B', *args, **kwargs):
        super().__init__(*args, bias=False, **kwargs)

        device = kwargs.get('device', None)

        _, _, H, W = self.weight.shape
        # for (H, W) both odd, (U, V) is the center cell
        U, V = H // 2, W // 2

        # [0..(H-1)] x [0..(W-1)]
        # for H, W, both odd, H = 2U + 1, W = 2V + 1
        # rows 0 .. U-1 are all 1
        # rows U+1 .. 2U are all 0
        # row U: for Mask B, V+1 .. 2V are all 0, others are 1
        # row U: for Mask A,   V .. 2V are all 0, others are 1
        mask_row_U_col_start = V + (mask_type == 'B')
        self.mask = torch.zeros(self.weight.shape, device=device).float()
        self.mask[:, :, :U, :] = 1.0
        self.mask[:, :, U, :mask_row_U_col_start] = 1.0


    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)




# normal resnet is:
# > We adopt batch normalization (BN) [16] right after each convolution and
# > before activation, following [16].

#   F(x) = (Conv -> BN -> ReLU -> Conv)(x)
#   out = ReLU(F(x) + x)

# jzbontar uses (non-residual) repeating blocks of:
# Conv -> ReLU -> BatchNorm
#
# by the way, jzbontar's PixelCNN implementation is the first place I saw this, but
# this appears to be standard practice. From https://stackoverflow.com/a/59939495:
#
# > Andrew Ng says that batch normalization should be applied immediately before
# > the non-linearity of the current layer. The authors of the BN paper said that
# > as well, but now according to François Chollet on the keras thread, the BN paper
# > authors use BN after the activation layer. On the other hand, there are some
# > benchmarks such as the one discussed on this torch-residual-networks github issue
# > that show BN performing better after the activation layers.
# >
# > My current opinion (open to being corrected) is that you should do BN after the
# > activation layer, and if you have the budget for it and are trying to squeeze out
# > extra accuracy, try before the activation layer.

# approach: a residual block with 2 3x3 convolutions, separated by a Relu

# trying out:
#  batch norm before the first, no batch norm on second (ResNet style)
#  batch norm after both ReLUs, jzbontar style
class PixelCNNResBlockBNBeforeKK(nn.Module):
    def __init__(self, num_filters, kernel_size, device=None):
        super().__init__()

        if kernel_size % 2 != 1:
            raise Exception("kernel_size must be odd")

        padding = int((kernel_size-1)/2)

        self.layers = nn.Sequential(
            MaskedConv2d(mask_type='B', in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=padding, device=device),
            nn.BatchNorm2d(num_filters, device=device),
            nn.ReLU(),
            MaskedConv2d(mask_type='B', in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=padding, device=device),
        )

    def forward(self, x):
        return torch.relu(self.layers(x) + x)

# residual block, but using (ReLU -> BN), as in the style of jzbontar's implementation,
# instead of the (BN -> ReLU) in ResNet paper
class PixelCNNResBlockBNAfterKK(nn.Module):
    def __init__(self, num_filters, kernel_size, device=None):
        super().__init__()

        if kernel_size % 2 != 1:
            raise Exception("kernel_size must be odd")

        padding = int((kernel_size-1)/2)

        self.layers = nn.Sequential(
            MaskedConv2d(mask_type='B', in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=padding, device=device),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters, device=device),
            MaskedConv2d(mask_type='B', in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding=padding, device=device),
        )

        self.final_bn = nn.BatchNorm2d(num_filters, device=device)

    def forward(self, x):
        return self.final_bn(torch.relu(self.layers(x) + x))



class PixelCNNResBlock3NoBN(nn.Module):
    def __init__(self, num_filters, kernel_size_1, kernel_size_2, kernel_size_3, device=None):
        super().__init__()

        ks = [kernel_size_1, kernel_size_2, kernel_size_3]
        if any(k % 2 != 1 for k in ks):
            raise Exception("kernel sizes must be odd")

        paddings = [k // 2 for k in ks]

        self.layers = nn.Sequential(
            MaskedConv2d(mask_type='B', in_channels=num_filters, out_channels=num_filters, kernel_size=ks[0], stride=1, padding=paddings[0], device=device),
            nn.ReLU(),
            MaskedConv2d(mask_type='B', in_channels=num_filters, out_channels=num_filters, kernel_size=ks[1], stride=1, padding=paddings[1], device=device),
            nn.ReLU(),
            MaskedConv2d(mask_type='B', in_channels=num_filters, out_channels=num_filters, kernel_size=ks[2], stride=1, padding=paddings[2], device=device),
        )

    def forward(self, x):
        return self.layers(x) + x






class PixelCNN(nn.Module):
    def __init__(self, in_channels=3, num_filters=64, num_color_values=256, device=None, num_blocks=7):
        super().__init__()
        # Architecture from Table 1 of "Pixel Recurrent Neural Networks":
        #
        # 1. 7x7 conv, mask A
        # 2. multiple residual blocks (mask B):
        #  h(x) = x -> (ReLU -> 1x1 conv) -> (ReLU -> 3x3 conv) -> (ReLU -> 1x1 conv)
        #  o(x) = x + h(x)
        # 3. 2 layers of (ReLU -> 1x1 conv (mask B))
        # 4. 256-way softmax for each RGB color

        final_out_channels = in_channels * num_color_values

        layer1 = nn.Sequential(
            MaskedConv2d(mask_type='A', in_channels=in_channels, out_channels=num_filters, kernel_size=7, stride=1, padding=3, device=device),
            nn.ReLU(),
            # nn.BatchNorm2d(num_filters, device=device),
        )

        # should we use batch norms here?
        layer_logit = nn.Sequential(
            nn.ReLU(),
            # nn.BatchNorm2d(num_filters, device=device),
            MaskedConv2d(mask_type='B', in_channels=num_filters, out_channels=num_filters, kernel_size=1, stride=1, padding=0, device=device),
            nn.ReLU(),
            # nn.BatchNorm2d(num_filters, device=device),
            MaskedConv2d(mask_type='B', in_channels=num_filters, out_channels=final_out_channels, kernel_size=1, stride=1, padding=0, device=device),
        )

        self.pcnn = torch.nn.Sequential()
        self.pcnn.add_module('layer1', layer1)

        print(f'{num_blocks=}')
        for i in range(num_blocks):
            # block = PixelCNNResBlockKK(num_filters, kernel_size=5, device=device)
            block = PixelCNNResBlock3NoBN(num_filters, kernel_size_1=1, kernel_size_2=3, kernel_size_3=1, device=device)

            self.pcnn.add_module(f'block{i+1}', block)

        self.pcnn.add_module('layer_logit', layer_logit)

        # 256-way softmax is handled by cross-entropy loss

    def forward(self, x):
        return self.pcnn(x)





def get_split_data_train_val(dataset_train_val, batch_size, validation_frac=0.1):
    N = len(dataset_train_val)
    V = int(N * validation_frac)

    dataset_train, dataset_val = torch.utils.data.random_split(dataset_train_val, [N - V, V])
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    return (dataloader_train, dataloader_val)


def train(dataloader_train, dataloader_val, num_epochs=2, save_period_epochs=5, sample_period_epochs=1, num_filters=128, learning_rate=1e-3, experiment_name=None, num_blocks=7):
    time_global_start = datetime.datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\n{num_epochs=} {num_filters=}, {device=}')

    C, H, W = None, None, None
    for batch, target in dataloader_train:
        C, H, W = batch.shape[1:]
        break

    print(f'{C=} {H=}, {W=}')

    num_color_values = 256
    model = PixelCNN(in_channels=C, num_filters=num_filters, num_color_values=num_color_values, device=device, num_blocks=num_blocks)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    checkpoints_dir = 'checkpoints'
    samples_dir = 'samples'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_num = epoch + 1
        print(f'\n-------------------\nEpoch {epoch_num}\n-------------------')

        ######################
        # Train
        ######################
        print('#######\n# train\n#######')
        time_epoch_train_start = time.time()
        model.train()
        total_train_loss = 0.
        for batch_idx, (batch, labels) in enumerate(dataloader_train):
            batch = batch.to(device)
            target = (batch.data[:,:,:,:] * 255).long()
            # print(' >>> ')
            # print(f' ... batch.shape = {batch.shape}')
            # print(f' ... target.shape = {target.shape}')
            optimizer.zero_grad()
            output = model(batch)
            output_reshaped = output.reshape(batch.shape[0], num_color_values, C, H, W)
            # print(f' ... output.shape = {output.shape}')
            # print(f' ... output_reshaped.shape = {output_reshaped.shape}')
            loss = criterion(output_reshaped, target)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            print(f'  Train Batch: {batch_idx + 1}, Loss: %.4f' % loss.item())

        time_epoch_train = time.time() - time_epoch_train_start


        ######################
        # Eval
        ######################
        print('#######\n# val\n#######')
        time_epoch_eval_start = time.time()
        model.eval()
        total_val_loss = 0.

        with torch.no_grad():
            for batch_idx, (batch, _labels) in enumerate(dataloader_val):
                batch = batch.to(device)
                target = (batch.data[:,:,:,:] * 255).long()
                # optimizer.zero_grad()
                output = model(batch)
                output_reshaped = output.reshape(batch.shape[0], num_color_values, C, H, W)
                loss = criterion(output_reshaped, target)
                # loss.backward()
                # optimizer.step()

                total_val_loss += loss.item()
                print(f'  Val Batch: {batch_idx + 1}, Loss: %.4f' % loss.item())

        time_epoch_eval = time.time() - time_epoch_eval_start

        avg_train_loss = total_train_loss / len(dataloader_train)
        avg_val_loss = total_val_loss / len(dataloader_val)


        print(f'[Epoch {epoch_num}]  {time_epoch_train=}, {total_train_loss=}, {avg_train_loss=}, num train batches={len(dataloader_train)}')
        print(f'[Epoch {epoch_num}]  {time_epoch_eval=}, {total_val_loss=}, {avg_val_loss=}, num val batches={len(dataloader_val)}')


        # Save checkpoint
        if epoch_num % save_period_epochs == 0:
            global_start_display = time_global_start.strftime('%Y%m%d-%H%M%S')
            exp_name_file_tag = f'{experiment_name}_' if experiment_name is not None else ''
            checkpoint_file = f'pcnn_{exp_name_file_tag}{global_start_display}_epoch_{epoch_num}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss,
            }, f'{checkpoints_dir}/{checkpoint_file}')
            print(f'Saved checkpoint {checkpoint_file}')


        ######################
        # Sample
        ######################
        if epoch_num % sample_period_epochs == 0:
            num_samples = 144
            # initialize with all zeros
            samples = torch.zeros(size=(num_samples, C, H, W)).to(device)
            with torch.no_grad():
                # final number of output channels for PixelCNN is V*C, V = number of color values
                # therefore we can reshape (N, V*C, H, W) -> (N, V, C, H, W)
                # we subsequently softmax along the V dimension to determine what color intensity to sample
                for i in range(H):
                    for j in range(W):
                        for c in range(C):
                            output = model(samples)
                            output = output.view(output.shape[0], -1, C, H, W)
                            probs = F.softmax(output, dim=1)
                            chosen = torch.multinomial(probs[:, :, c, i, j], 1).squeeze().float() / 255.
                            samples[:, c, i, j] = chosen

            global_start_display = time_global_start.strftime('%Y%m%d-%H%M%S')
            exp_name_file_tag = f'{experiment_name}_' if experiment_name is not None else ''
            samples_file_name = f'pcnn_{exp_name_file_tag}{global_start_display}_samples_epoch_{epoch_num}.png'
            samples_file_path = f'{samples_dir}/{samples_file_name}'
            torchvision.utils.save_image(samples, samples_file_path, nrow=12, padding=0)


def main():
    # data_root = 'data/cifar10/'
    # dataset_train_val = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transforms.ToTensor())

    data_root = 'data/fashion-mnist'
    dataset_train_val = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transforms.ToTensor())

    # data_root = 'data/mnist'
    # dataset_train_val = datasets.MNIST(root=data_root, train=True, download=True, transform=transforms.ToTensor())

    batch_size = 256
    experiment_name = 'fmnist-7-vanilla-no-bn-131-nf-256-nb-10'
    dataloader_train, dataloader_val = get_split_data_train_val(dataset_train_val, batch_size)
    num_epochs = 25
    save_period_epochs = 5
    sample_period_epochs = 5
    num_filters = 256
    num_blocks = 10
    learning_rate = 1e-3
    train(dataloader_train, dataloader_val, num_epochs=num_epochs, save_period_epochs=save_period_epochs,
          sample_period_epochs=sample_period_epochs, num_filters=num_filters, experiment_name=experiment_name,
          learning_rate=learning_rate, num_blocks=num_blocks)


if __name__ == '__main__':
    main()
