import datetime
import os
import time
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import torch
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cpc import CPCV1
from models.encoder.resnet import CPCEncoder_ResNetV2
from models.autoregressor.pixel_cnn_wfalcon import PixelCNNWFalcon
from utils.patches import MakePatches
from models.encoder import resnet



# based on rschwarz15
def make_training_transforms(args, data_norm):
    """ return a composite transform for all data preprocessing needed for training.
    args = {
      crop_size: 1-d size for the random crop
      crop_padding: padding for random crop
      patch_size: 1-d size for the patch
      p_horiz_flip: probability
      is_grayscale: boolean. if image isnt already grayscale, will be converted.
    }"""
    print(args)

    trans = [
        transforms.RandomCrop(args['crop_size'], args['crop_padding']),
        transforms.RandomHorizontalFlip(0.5 if args['p_horiz_flip'] is None else args['p_horiz_flip'])
    ]

    # convert to tensor and normalize.
    if args['is_grayscale']:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=data_norm["bw_mean"], std=data_norm["bw_std"]))
    else:
        # TODO: listed as step 4 in patch preprocessing of Appendix A
        trans.append(transforms.RandomGrayscale(p=0.25))
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=data_norm["mean"], std=data_norm["std"]))

    trans.append(MakePatches(crop_size=args['crop_size'], patch_size=args['patch_size']))

    trans = transforms.Compose(trans)

    print("training transforms: " + str(trans))

    return trans


def train(dataloader, num_epochs=10, learning_rate=1e-3, input_channels=3, experiment_name=None, save_period_epochs=1):
    train_start_datetime = datetime.datetime.now()
    train_start_display = train_start_datetime.strftime('%Y%m%d-%H%M%S')

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(f'Using device: {device_name}')

    layer_num_blocks = [2, 2, 2, 2]
    layer_num_features = [64, 128, 256, 512]
    prediction_steps = 5
    num_neg_samples = 100

    encoder_net = CPCEncoder_ResNetV2(input_channels, layer_num_features=layer_num_features, layer_num_blocks=layer_num_blocks)
    print(encoder_net)
    autoreg_net = PixelCNNWFalcon(input_channels=encoder_net.encoding_num_features)
    cpcv1 = CPCV1(encoder_net=encoder_net, autoreg_net=autoreg_net, num_pred_steps=prediction_steps,
                  num_neg_samples=num_neg_samples, device=device)

    grad_params = filter(lambda p: p.requires_grad, cpcv1.parameters())
    optimizer = optim.Adam(grad_params, lr=learning_rate)

    checkpoints_dir = 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)

    num_batches = len(dataloader)
    # use the last 10% of batches per epoch to calculate average training loss
    epoch_training_loss_frac = 0.1
    batch_start_loss_epoch = round((1 - epoch_training_loss_frac) * num_batches)

    for epoch_idx in range(num_epochs):
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
                print(f"[{i+1}/{num_batches}] Batch Loss: {loss.item()}")

            if i >= batch_start_loss_epoch:
                loss_epoch_train += loss.item()

        epoch_duration = time.time() - start_time
        avg_epoch_loss_train = loss_epoch_train / (num_batches - batch_start_loss_epoch)
        print(f'End of epoch. Took {epoch_duration}, approx. avg epoch training loss = {avg_epoch_loss_train}\n')

        # Save checkpoint
        if epoch_num % save_period_epochs == 0:
            exp_name_file_tag = f'{experiment_name}_' if experiment_name is not None else ''
            checkpoint_file = f'cpcv1_{exp_name_file_tag}{train_start_display}_epoch_{epoch_num}.pth'
            torch.save({
                'epoch_num': epoch_num,
                'model_state_dict': cpcv1.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_train_loss': avg_epoch_loss_train,
            }, f'{checkpoints_dir}/{checkpoint_file}')
            print(f'Saved checkpoint {checkpoint_file}')


if __name__ == '__main__':
    batch_size = 64
    num_workers = 2
    download_data = True
    data_root = 'data/cifar10/'

    args = {
      'crop_size': 32,
      'crop_padding': 0,
      'patch_size': 8,
      'p_horiz_flip': 0.5,
      'is_grayscale': True
    }

    # from https://github.com/rschwarz15/CPCV2-PyTorch/blob/master/data/data_handlers.py
    # for CIFAR-10
    data_norm = {
        "mean": [0.49139968, 0.48215827, 0.44653124],
        "std": [0.24703233, 0.24348505, 0.26158768],
        "bw_mean": [0.4808616],
        "bw_std": [0.23919088],
    }

    transform_train = make_training_transforms(args, data_norm)
    dataset_unsupervised = datasets.CIFAR10(root=data_root, train=True, download=download_data,
                                         transform=transform_train)

    dataloader_unsupervised = DataLoader(dataset_unsupervised, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers)

    num_epochs = 10
    learning_rate = 1e-3
    input_channels = 3 if args['is_grayscale'] == False else 1
    experiment_name = "grayscale-1"
    train(dataloader_unsupervised, num_epochs, learning_rate=learning_rate, input_channels=input_channels, experiment_name=experiment_name)