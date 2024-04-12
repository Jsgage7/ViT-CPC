import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms

from data import make_cifar10_dataloader, make_cifar100_dataloader, make_cifar100_dataloader_train_val, make_cifar10_dataloader_train_val, make_cifar10_dataloader_test
from cpc import CPCV1
from resnet_nh import ResNetV2_Encoder
from pixel_cnn_wfalcon import PixelCNNWFalcon

checkpoints_dir = 'checkpoints'


def train_efficient_classifier(dataloader_train, dataloader_val, num_classes, num_epochs=10, learning_rate=1e-3, input_channels=3, experiment_name=None,
          save_period_epochs=1, ss_checkpoint_path=None, eff_checkpoint_path=None):
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
    autoreg_net = None

    start_epoch_idx = 0

    # load the saved weights from self-supervised checkpoint, but don't freeze. we fine-tune instead
    if ss_checkpoint_path is not None:
        ss_saved = torch.load(ss_checkpoint_path)
        encoder_net.load_state_dict(ss_saved['enc_state_dict'])

        cpcv1 = CPCV1(encoder_net=encoder_net, autoreg_net=autoreg_net, num_pred_steps=prediction_steps, num_neg_samples=num_neg_samples,
            device=device, learning_mode='efficient-classification', num_classes=num_classes)
    else:
        cpcv1 = CPCV1(encoder_net=encoder_net, autoreg_net=autoreg_net, num_pred_steps=prediction_steps, num_neg_samples=num_neg_samples,
            device=device, learning_mode='efficient-classification', num_classes=num_classes)

        eff_saved = torch.load(eff_checkpoint_path)
        cpcv1.load_state_dict(eff_saved['model_state_dict'])
        start_epoch_idx = eff_saved['epoch_num']


    criterion = nn.CrossEntropyLoss().to(device)

    grad_params = filter(lambda p: p.requires_grad, cpcv1.parameters())
    optimizer = optim.Adam(grad_params, lr=learning_rate)

    num_batches = len(dataloader_train)
    print('-----')
    print(type(dataloader_train))
    # use the last 10% of batches per epoch to calculate average training loss
    epoch_training_loss_frac = 0.1
    batch_start_loss_epoch = round((1 - epoch_training_loss_frac) * num_batches)

    # avoid division by zero for very small fractions
    if (num_batches - batch_start_loss_epoch) <= 5:
        batch_start_loss_epoch = num_batches - 5

    print(f'{num_batches=}')
    print(f'{batch_start_loss_epoch=}')

    # best_avg_val_loss = 1e20
    # best_avg_val_epoch_num = None

    best_val_acc = 0.0
    best_val_acc_epoch_num = None


    for epoch_idx in range(start_epoch_idx, start_epoch_idx + num_epochs):
        cpcv1.train()
        epoch_num = epoch_idx + 1
        print(f'\n#-------- [Efficient] Epoch {epoch_num} --------#')
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
        num_val_batches = len(dataloader_val)
        num_val_samples = len(dataloader_val.dataset)
        loss_epoch_val = 0
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

        val_accuracy = float(val_correct / num_val_samples)
        print(f'Validation avg loss: {avg_val_loss}; accuracy: {val_accuracy} ({val_correct} / {num_val_samples})')

        is_new_best = val_accuracy > best_val_acc
        prev_best_val_acc_epoch_num = None
        if is_new_best:
            best_val_acc = val_accuracy
            prev_best_val_acc_epoch_num = best_val_acc_epoch_num
            best_val_acc_epoch_num = epoch_num
            # best_avg_val_loss = avg_val_loss
            # best_avg_val_epoch_num = epoch_num



        # Save checkpoint
        is_periodic_save = epoch_num % save_period_epochs == 0
        if is_periodic_save or is_new_best:
            exp_name_file_tag = f'{experiment_name}_' if experiment_name is not None else ''
            checkpoint_file_prefix = f'cpcv1_{exp_name_file_tag}{train_start_display}'

            if not is_periodic_save and prev_best_val_acc_epoch_num is not None:
                old_best_file = f'{checkpoint_file_prefix}_epoch_{prev_best_val_acc_epoch_num}_best.pth'
                old_best_full_path = f'{checkpoints_dir}/{old_best_file}'
                if os.path.exists(old_best_full_path):
                    print(f'Removing {old_best_full_path}')
                    os.remove(old_best_full_path)

            checkpoint_file = f'{checkpoint_file_prefix}_epoch_{epoch_num}{"_best" if is_new_best else ""}.pth'
            torch.save({
                'epoch_num': epoch_num,
                'model_state_dict': cpcv1.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_train_loss': avg_epoch_loss_train,
            }, f'{checkpoints_dir}/{checkpoint_file}')
            print(f'Saved checkpoint {checkpoint_file}', flush=True)



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
        device=device, learning_mode='efficient-classification', num_classes=num_classes)

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



def train_all_pcts():
    batch_size = 64
    is_grayscale = False
    num_epochs = 15
    learning_rate = 5e-4
    input_channels = 3 if is_grayscale == False else 1
    save_period_epochs = 5
    num_classes = 10

    dataset_pcts = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    # ss_checkpoint_path = None
    # ss_checkpoint_path = f'{checkpoints_dir}/cpcv1_rgb-1-fixed_20230418-085751_epoch_30.pth'
    # ss_checkpoint_path = f'{checkpoints_dir}/cpcv1_cifar100-exp-1_20230422-104812_epoch_30.pth'
    # ss_checkpoint_path = f'{checkpoints_dir}/cpcv1_cifar100-resnet-26_20230423-230855_epoch_30.pth'
    # ss_checkpoint_path = f'{checkpoints_dir}/cpcv1_cifar100-resnet-18-gray-aug_20230426-212613_epoch_45.pth'
    # ss_checkpoint_path = f'{checkpoints_dir}/cpcv1_cifar10-resnet-18-gray-aug_20230427-222834_epoch_30.pth'

    # final
    # ss_checkpoint_path = f'{checkpoints_dir}/cpcv1_cifar10-resnet-18-actually-gray-aug_20230428-014642_epoch_30.pth'
    # ss_checkpoint_path = f'{checkpoints_dir}/cpcv1_cifar10-resnet-18-gray-aug_20230427-222834_epoch_30.pth'
    ss_checkpoint_path = f'{checkpoints_dir}/cpcv1_cifar10-resnet-18-aug-gray-color-jitter_20230429-171340_epoch_30.pth'


    eff_checkpoint_path = None
    # eff_checkpoint_path = f'{checkpoints_dir}/cpcv1_cifar100-exp-1-eff-2-pct-01_20230423-083220_epoch_10.pth'
    # eff_checkpoint_path = f'{checkpoints_dir}/cpcv1_cifar10-resnet-18-actually-gray-aug-eff-pct-1.0_20230429-042521_epoch_10.pth'


    for dataset_pct in dataset_pcts:
        # dataloader_train, dataloader_val = make_cifar100_dataloader_train_val(batch_size, is_grayscale, dataset_pct=0.20)
        dataloader_train, dataloader_val = make_cifar10_dataloader_train_val(batch_size, is_grayscale, dataset_pct=dataset_pct, fully_supervised=True)

        # experiment_name = "cifar100-resnet-18-gray-aug-eff-pct-20"
        # experiment_name = f"cifar10-resnet-18-actually-gray-aug-eff-pct-{dataset_pct}"
        experiment_name = f"cifar10-resnet-18-aug-gray-color-jitter-eff-pct-{dataset_pct}"

        train_efficient_classifier(dataloader_train, dataloader_val, num_classes, num_epochs, learning_rate=learning_rate,
            input_channels=input_channels, experiment_name=experiment_name, save_period_epochs=save_period_epochs,
            ss_checkpoint_path=ss_checkpoint_path, eff_checkpoint_path=eff_checkpoint_path)




def test_eff_trained():
    batch_size = 64
    is_grayscale = False
    input_channels = 3 if is_grayscale == False else 1
    num_classes = 10

    checkpoint_filenames = [
        # 'cpcv1_cifar10-resnet-18-actually-gray-aug-eff-pct-0.02_20230429-042040_epoch_6_best.pth',
        # 'cpcv1_cifar10-resnet-18-actually-gray-aug-eff-pct-0.05_20230428-172415_epoch_13_best.pth',
        # 'cpcv1_cifar10-resnet-18-actually-gray-aug-eff-pct-0.1_20230428-173544_epoch_11_best.pth',
        # 'cpcv1_cifar10-resnet-18-actually-gray-aug-eff-pct-0.2_20230428-175841_epoch_6_best.pth',
        # 'cpcv1_cifar10-resnet-18-actually-gray-aug-eff-pct-0.5_20230428-184441_epoch_14_best.pth',
        # 'cpcv1_cifar10-resnet-18-actually-gray-aug-eff-pct-1.0_20230429-042521_epoch_12_best.pth',

        # 'cpcv1_cifar10-resnet-18-gray-aug-eff-pct-0.02_20230428-204259_epoch_13_best.pth',
        # 'cpcv1_cifar10-resnet-18-gray-aug-eff-pct-0.05_20230428-204740_epoch_13_best.pth',
        # 'cpcv1_cifar10-resnet-18-gray-aug-eff-pct-0.1_20230428-205920_epoch_9_best.pth',
        # 'cpcv1_cifar10-resnet-18-gray-aug-eff-pct-0.2_20230428-212234_epoch_15_best.pth',
        # 'cpcv1_cifar10-resnet-18-gray-aug-eff-pct-0.5_20230428-220917_epoch_15_best.pth',
        # 'cpcv1_cifar10-resnet-18-gray-aug-eff-pct-1.0_20230429-082439_epoch_15_best.pth'

        # 'cpcv1_cifar10-resnet-18-actually-gray-aug-eff-pct-1.0_20230429-042521_epoch_12_best.pth',
        # 'cpcv1_cifar10-resnet-18-actually-gray-aug-eff-pct-1.0_20230429-042521_epoch_15.pth',
        # 'cpcv1_cifar10-resnet-18-actually-gray-aug-eff-pct-1.0_20230429-042521_epoch_10.pth',
        # 'cpcv1_cifar10-resnet-18-actually-gray-aug-eff-pct-1.0_20230429-042521_epoch_5.pth',

        # 'cpcv1_cifar10-resnet-18-actually-gray-aug-eff-pct-1.0_20230429-143801_epoch_12.pth'
        'cpcv1_cifar10-resnet-18-aug-gray-color-jitter-eff-pct-0.02_20230430-025048_epoch_7_best.pth',
        'cpcv1_cifar10-resnet-18-aug-gray-color-jitter-eff-pct-0.05_20230430-025531_epoch_11_best.pth',
        'cpcv1_cifar10-resnet-18-aug-gray-color-jitter-eff-pct-0.1_20230430-030705_epoch_7_best.pth',
        'cpcv1_cifar10-resnet-18-aug-gray-color-jitter-eff-pct-0.2_20230430-033009_epoch_13_best.pth',
        'cpcv1_cifar10-resnet-18-aug-gray-color-jitter-eff-pct-0.5_20230430-041602_epoch_14_best.pth',
        'cpcv1_cifar10-resnet-18-aug-gray-color-jitter-eff-pct-1.0_20230430-061040_epoch_11_best.pth'
    ]


    dataloader_test = make_cifar10_dataloader_test(batch_size, is_grayscale)

    for checkpoint_filename in checkpoint_filenames:
        eff_checkpoint_path = f'{checkpoints_dir}/{checkpoint_filename}'
        print(f'\nTesting eff checkpoint: {checkpoint_filename}')
        test_trained_classifier(dataloader_test, num_classes, input_channels=input_channels, experiment_name=None,
            eff_checkpoint_path=eff_checkpoint_path)





if __name__ == '__main__':
    random_seed = 15009
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    os.makedirs(checkpoints_dir, exist_ok=True)

    # train_all_pcts()
    test_eff_trained()
