import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms

from data import make_cifar10_dataloader, make_cifar100_dataloader, make_cifar10_dataloader_train_val, make_cifar100_dataloader_train_val, make_cifar10_dataloader_test
from cpc import CPCV1
from resnet_nh import ResNetV2_Encoder
from pixel_cnn_wfalcon import PixelCNNWFalcon

checkpoints_dir = 'checkpoints'



def train_linear_classifier(dataloader_train, dataloader_val, ss_checkpoint_path, num_classes, num_epochs=10, learning_rate=1e-3, input_channels=3, experiment_name=None,

          save_period_epochs=1):
    train_start_datetime = datetime.datetime.now()
    train_start_display = train_start_datetime.strftime('%Y%m%d-%H%M%S')

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(f'Using device: {device_name}')

    layer_num_blocks = [2, 2, 2, 2]
    layer_num_features = [64, 128, 256, 512]
    prediction_steps = 5
    num_neg_samples = 100

    encoder_net = ResNetV2_Encoder(input_channels, layer_num_features=layer_num_features, layer_num_blocks=layer_num_blocks)
    autoreg_net = None

    # load the saved weights from self-supervised checkpoint and freeze them
    ss_saved = torch.load(ss_checkpoint_path)
    encoder_net.load_state_dict(ss_saved['enc_state_dict'])
    for param in encoder_net.parameters():
        param.requires_grad = False

    cpcv1 = CPCV1(encoder_net=encoder_net, autoreg_net=autoreg_net, num_pred_steps=prediction_steps, num_neg_samples=num_neg_samples,
                  device=device, learning_mode='linear-classification', num_classes=num_classes)

    criterion = nn.CrossEntropyLoss().to(device)


    grad_params = filter(lambda p: p.requires_grad, cpcv1.parameters())
    optimizer = optim.Adam(grad_params, lr=learning_rate)

    num_batches = len(dataloader_train)
    # use the last 10% of batches per epoch to calculate average training loss
    epoch_training_loss_frac = 0.1
    batch_start_loss_epoch = round((1 - epoch_training_loss_frac) * num_batches)

    best_avg_val_loss = 1e20
    best_avg_val_epoch_num = None

    for epoch_idx in range(num_epochs):
        cpcv1.train()

        epoch_num = epoch_idx + 1
        print(f'\n#-------- [Linear] Epoch {epoch_num} --------#')
        start_time = time.time()
        loss_epoch_train = 0.
        for i, batch in enumerate(dataloader_train):
            # `patches` is a batch of grids of overlapping patches made from an input image
            # patches.shape = (B, C, G, G, P, P)
            #  - B = batch_size
            #  - G = grid_size
            #  - P = patch_size
            patches, labels = batch
            labels = labels.to(device)
            cpcv1.zero_grad()
            class_out = cpcv1(patches.to(device))
            loss = criterion(class_out, labels)

            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0 or i >= batch_start_loss_epoch:

                print(f"[{i+1}/{num_batches}] Batch Loss: {loss.item()}", flush=True)


            if i >= batch_start_loss_epoch:
                loss_epoch_train += loss.item()

        epoch_duration = time.time() - start_time
        avg_epoch_loss_train = loss_epoch_train / (num_batches - batch_start_loss_epoch)
        print(f'End of epoch. Took {epoch_duration}, approx. avg epoch training loss = {avg_epoch_loss_train}\n')


        # val
        cpcv1.eval()
        val_correct = 0
        loss_epoch_val = 0
        num_val_batches = len(dataloader_val)
        num_val_samples = len(dataloader_val.dataset)
        with torch.no_grad():
            for batch in dataloader_val:
                images, labels = batch
                labels = labels.to(device)
                class_out = cpcv1(images.to(device)).view(-1, num_classes)
                preds = torch.argmax(class_out, dim=1)
                val_correct += (preds == labels).sum().item()
                loss = criterion(class_out, labels)
                loss_epoch_val += loss.item()

        avg_val_loss = loss_epoch_val / num_val_batches
        is_new_best = avg_val_loss < best_avg_val_loss
        prev_best_avg_val_epoch_num = None
        if is_new_best:
            best_avg_val_loss = avg_val_loss
            prev_best_avg_val_epoch_num = best_avg_val_epoch_num
            best_avg_val_epoch_num = epoch_num

        val_accuracy = float(val_correct / num_val_samples)
        print(f'Validation avg loss: {avg_val_loss}; accuracy: {val_accuracy} ({val_correct} / {num_val_samples})')


        # Save checkpoint
        is_periodic_save = epoch_num % save_period_epochs == 0
        if is_periodic_save or is_new_best:
            exp_name_file_tag = f'{experiment_name}_' if experiment_name is not None else ''
            checkpoint_file_prefix = f'cpcv1_{exp_name_file_tag}{train_start_display}'

            if is_new_best and prev_best_avg_val_epoch_num is not None:
                old_best_file = f'{checkpoint_file_prefix}_epoch_{prev_best_avg_val_epoch_num}_best.pth'
                old_best_full_path = f'{checkpoints_dir}/{old_best_file}'
                if os.path.exists(old_best_full_path):
                    os.remove(old_best_full_path)

            checkpoint_file = f'{checkpoint_file_prefix}_epoch_{epoch_num}{"_best" if is_new_best else ""}.pth'

            torch.save({
                'epoch_num': epoch_num,
                'model_state_dict': cpcv1.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_train_loss': avg_epoch_loss_train,
            }, f'{checkpoints_dir}/{checkpoint_file}')
            print(f'Saved checkpoint {checkpoint_file}')


def test_trained_classifier(dataloader_test, num_classes, input_channels=3, experiment_name=None,
        eff_checkpoint_path=None):
    test_start_datetime = datetime.datetime.now()
    test_start_display = test_start_datetime.strftime('%Y%m%d-%H%M%S')

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(f'Using device: {device_name}')





    layer_num_blocks = [2, 2, 2, 2]
    # layer_num_blocks = [3, 3, 3, 3]
    layer_num_features = [64, 128, 256, 512]
    prediction_steps = 5
    num_neg_samples = 100

    encoder_net = ResNetV2_Encoder(input_channels, layer_num_features=layer_num_features, layer_num_blocks=layer_num_blocks)
    autoreg_net = None

    cpcv1 = CPCV1(encoder_net=encoder_net, autoreg_net=autoreg_net, num_pred_steps=prediction_steps, num_neg_samples=num_neg_samples,
                  device=device, learning_mode='linear-classification', num_classes=num_classes)

    # we don't bother freezing here because we are using torch.no_grad() below
    eff_saved = torch.load(eff_checkpoint_path)
    cpcv1.load_state_dict(eff_saved['model_state_dict'])

    # Test set evaluation
    cpcv1.eval()
    test_correct = 0
    num_test_samples = len(dataloader_test.dataset)
    with torch.no_grad():
        for batch in dataloader_test:
            images, labels = batch
            labels = labels.to(device)
            class_out = cpcv1(images.to(device)).view(-1, num_classes)
            preds = torch.argmax(class_out, dim=1)
            test_correct += (preds == labels).sum().item()

    test_accuracy = float(test_correct / num_test_samples)
    print(f'Test accuracy: {test_accuracy} ({test_correct} / {num_test_samples})')


def train_linear():
    batch_size = 128
    is_grayscale = False
    dataloader_train, dataloader_val = make_cifar10_dataloader_train_val(batch_size, is_grayscale, fully_supervised=True)

    num_epochs = 15
    learning_rate = 1e-3
    input_channels = 3 if is_grayscale == False else 1
    # experiment_name = "cifar100-resnet-18-linear"
    experiment_name = "cifar10-resnet-18-aug-gray-color-jitter-linear"
    save_period_epochs = 5

    # ss_checkpoint_path = f'{checkpoints_dir}/cpcv1_cifar100-exp-1_20230422-104812_epoch_30.pth'
    # ss_checkpoint_path = f'{checkpoints_dir}/cpcv1_cifar100-resnet-18-gray-aug_20230426-212613_epoch_45.pth'

    # final
    # ss_checkpoint_path = f'{checkpoints_dir}/cpcv1_cifar10-resnet-18-actually-gray-aug_20230428-014642_epoch_30.pth'
    # ss_checkpoint_path = f'{checkpoints_dir}/cpcv1_cifar10-resnet-18-gray-aug_20230427-222834_epoch_30.pth'
    ss_checkpoint_path = f'{checkpoints_dir}/cpcv1_cifar10-resnet-18-aug-gray-color-jitter_20230429-171340_epoch_30.pth'


    num_classes = 10
    train_linear_classifier(dataloader_train, dataloader_val, ss_checkpoint_path, num_classes, num_epochs, learning_rate=learning_rate,
          input_channels=input_channels, experiment_name=experiment_name, save_period_epochs=save_period_epochs)



def test_linear_trained():
    batch_size = 128
    is_grayscale = False
    input_channels = 3 if is_grayscale == False else 1
    num_classes = 10

    checkpoint_filenames = [
        # 'cpcv1_cifar10-resnet-18-actually-gray-aug-linear-fully-supervised_20230428-140227_epoch_15_best.pth',
        # 'cpcv1_cifar10-resnet-18-gray-aug-linear-fully-supervised_20230428-153535_epoch_13_best-backup.pth',
        # 'cpcv1_cifar10-resnet-18-gray-aug-linear-fully-supervised_20230428-153535_epoch_14_best-backup.pth',
        # 'cpcv1_cifar10-resnet-18-gray-aug-linear-fully-supervised_20230428-153535_epoch_15_best.pth'
        'cpcv1_cifar10-resnet-18-aug-gray-color-jitter-linear_20230430-111626_epoch_15_best.pth'
    ]

    dataloader_test = make_cifar10_dataloader_test(batch_size, is_grayscale)

    for checkpoint_filename in checkpoint_filenames:
        eff_checkpoint_path = f'{checkpoints_dir}/{checkpoint_filename}'
        print(f'\nTesting linear checkpoint: {checkpoint_filename}')
        test_trained_classifier(dataloader_test, num_classes, input_channels=input_channels, experiment_name=None,
            eff_checkpoint_path=eff_checkpoint_path)




if __name__ == '__main__':
    random_seed = 15009
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    os.makedirs(checkpoints_dir, exist_ok=True)

    # train_linear()
    test_linear_trained()

