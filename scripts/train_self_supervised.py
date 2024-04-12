import datetime
import os
import time

import torch
from torch import optim
from torchvision import datasets, transforms

from data import make_cifar10_dataloader, make_cifar100_dataloader
from cpc import CPCV1
from resnet_nh import ResNetV2_Encoder
from pixel_cnn_wfalcon import PixelCNNWFalcon



def train_self_supervised(dataloader, num_epochs=10, learning_rate=1e-3, input_channels=3, experiment_name=None,
          save_period_epochs=1, checkpoint=None):
    train_start_datetime = datetime.datetime.now()
    train_start_display = train_start_datetime.strftime('%Y%m%d-%H%M%S')

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(f'Using device: {device_name}')

    layer_num_blocks = [2, 2, 2, 2]
    # layer_num_blocks = [3, 3, 3, 3]
    layer_num_features = [64, 128, 256, 512]
    prediction_steps = 5
    num_neg_samples = 100

    encoder_net = ResNetV2_Encoder(input_channels, layer_num_features=layer_num_features, layer_num_blocks=layer_num_blocks)
    autoreg_net = PixelCNNWFalcon(input_channels=encoder_net.encoding_num_features)
    cpcv1 = CPCV1(encoder_net=encoder_net, autoreg_net=autoreg_net, num_pred_steps=prediction_steps, num_neg_samples=num_neg_samples,
                  device=device)

    grad_params = filter(lambda p: p.requires_grad, cpcv1.parameters())
    optimizer = optim.Adam(grad_params, lr=learning_rate)

    if checkpoint is not None:
        cpcv1.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    num_batches = len(dataloader)
    # use the last 10% of batches per epoch to calculate average training loss
    epoch_training_loss_frac = 0.1
    batch_start_loss_epoch = round((1 - epoch_training_loss_frac) * num_batches)

    start_epoch_idx = 0 if checkpoint is None else checkpoint['epoch_num']

    for epoch_idx in range(start_epoch_idx, start_epoch_idx + num_epochs):
        epoch_num = epoch_idx + 1
        print(f'\n#-------- Epoch {epoch_num} --------#')
        start_time = time.time()
        loss_epoch_train = 0.
        for i, batch in enumerate(dataloader):
            # `patches` is a batch of grids of overlapping patches made from an input image
            # patches.shape = (B, C, G, G, P, P)
            #  - B = batch_size
            #  - G = grid_size
            #  - P = patch_size
            patches, label = batch
            cpcv1.zero_grad()
            loss = cpcv1(patches.to(device))
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0 or i >= batch_start_loss_epoch:
                print(f"[{i+1}/{num_batches}] Batch Loss: {loss.item()}", flush=True)

            if i >= batch_start_loss_epoch:
                loss_epoch_train += loss.item()

        epoch_duration = time.time() - start_time
        avg_epoch_loss_train = loss_epoch_train / (num_batches - batch_start_loss_epoch)
        print(f'End of epoch. Took {epoch_duration}, approx. avg epoch training loss = {avg_epoch_loss_train}\n')

        # Save checkpoint
        exp_name_file_tag = f'{experiment_name}_' if experiment_name is not None else ''
        checkpoint_file_prefix = f'cpcv1_{exp_name_file_tag}{train_start_display}'
        if epoch_num % save_period_epochs == 0:
            checkpoint_file = f'{checkpoint_file_prefix}_epoch_{epoch_num}.pth'
        else:
            checkpoint_file = f'{checkpoint_file_prefix}_curr.pth'

        torch.save({
            'epoch_num': epoch_num,
            'enc_state_dict': cpcv1.encoder_net.state_dict(),
            'model_state_dict': cpcv1.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_train_loss': avg_epoch_loss_train,
        }, f'{checkpoints_dir}/{checkpoint_file}')
        print(f'Saved checkpoint {checkpoint_file}', flush=True)




if __name__ == '__main__':
    checkpoints_dir = 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)

    batch_size = 64
    is_grayscale = False
    dataloader_train = make_cifar10_dataloader(batch_size, is_grayscale)

    num_epochs = 30
    learning_rate = 5e-4
    input_channels = 3 if is_grayscale == False else 1
    experiment_name = "cifar10-resnet-18-aug-gray-color-jitter"
    save_period_epochs = 5

    # checkpoint = torch.load(f'{checkpoints_dir}/cpcv1_cifar100-exp-1_20230422-082840_epoch_5.pth')
    # checkpoint = torch.load(f'{checkpoints_dir}/cpcv1_cifar100-resnet-18-gray-aug_20230426-174653_epoch_15.pth')
    # checkpoint = torch.load(f'{checkpoints_dir}/cpcv1_cifar10-resnet-18-gray-aug_20230427-081556_epoch_25.pth')
    checkpoint = None

    train_self_supervised(dataloader_train, num_epochs, learning_rate=learning_rate,
          input_channels=input_channels, experiment_name=experiment_name, save_period_epochs=save_period_epochs, checkpoint=checkpoint)

