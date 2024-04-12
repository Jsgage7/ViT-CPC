import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms

from utils.data import make_cifar10_dataloader
from models.cpc import CPCVIT
from timm.models.layers import PatchEmbed
from models.encoder.vit import VisionTransformer



def train_linear_classifier(dataloader, ss_checkpoint_path, num_classes, num_epochs=10, learning_rate=1e-3, input_channels=3, experiment_name=None,
          save_period_epochs=1):
    train_start_datetime = datetime.datetime.now()
    train_start_display = train_start_datetime.strftime('%Y%m%d-%H%M%S')

    
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device_name = 'cpu'
    device = torch.device(device_name)
    print(f'Using device: {device_name}')

    prediction_steps = 5
    num_neg_samples = 100

    sample_shape = tuple(next(iter(dataloader))[0].shape)
    vit_patch_size = sample_shape[5] // 2

    encoder_net = VisionTransformer(img_size=sample_shape[5], patch_size=vit_patch_size, in_chans=input_channels, num_classes=10, embed_dim=16, depth=12,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='')
    
    autoreg_net = None

    # load the saved weights from self-supervised checkpoint and freeze them
    ss_saved = torch.load(ss_checkpoint_path)
    encoder_net.load_state_dict(ss_saved['enc_state_dict'])
    
    # for param in encoder_net.parameters():
    #     param.requires_grad = False

    cpcv1 = CPCVIT(encoder_net=encoder_net, autoreg_net=autoreg_net, num_pred_steps=prediction_steps,
                  num_neg_samples=num_neg_samples, device=device,  learning_mode='linear-classification', num_classes=10)

    criterion = nn.CrossEntropyLoss().to(device)

    grad_params = filter(lambda p: p.requires_grad, cpcv1.parameters())
    optimizer = optim.Adam(grad_params, lr=learning_rate)

    num_batches = len(dataloader)
    # use the last 10% of batches per epoch to calculate average training loss
    epoch_training_loss_frac = 0.1
    batch_start_loss_epoch = round((1 - epoch_training_loss_frac) * num_batches)

    best_model_loss = float('inf')
    for epoch_idx in range(num_epochs):
        epoch_num = epoch_idx + 1
        print(f'\n#-------- [Linear] Epoch {epoch_num} --------#')
        start_time = time.time()
        loss_epoch_train = 0.
        for i, batch in enumerate(dataloader):

            patches, labels = batch
            labels = labels.to(device)
            cpcv1.zero_grad()
            class_out = cpcv1(patches.to(device))
            loss = criterion(class_out, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0 or i >= batch_start_loss_epoch:
                print(f"[{i+1}/{num_batches}] Batch Loss: {loss.item()}")

            if i >= batch_start_loss_epoch:
                loss_epoch_train += loss.item()

        epoch_duration = time.time() - start_time
        avg_epoch_loss_train = loss_epoch_train / (num_batches - batch_start_loss_epoch)
        print(f'End of epoch. Took {epoch_duration}, approx. avg epoch training loss = {avg_epoch_loss_train}\n')

        avg_val_loss = validate(dataloader_val, cpcv1, criterion, device)


        # Save checkpoint
        if avg_val_loss < best_model_loss:
            best_model_loss = avg_val_loss
            exp_name_file_tag = f'{experiment_name}_' if experiment_name is not None else ''
            # checkpoint_file = f'cpcv1_{exp_name_file_tag}{train_start_display}_epoch_{epoch_num}.pth'
            checkpoint_file = f'best_model_vit_{experiment_name}.pth'
            torch.save({
                'epoch_num': epoch_num,
                'model_state_dict': cpcv1.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_train_loss': avg_epoch_loss_train,
            }, f'{checkpoints_dir}/{checkpoint_file}')
            print(f'Saved checkpoint {checkpoint_file}')

    return checkpoint_file


def validate(dataloader, model, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            patches, label = batch
            patches = patches.to(device)
            label = label.to(device)
            logits = model(patches)
            total_loss += criterion(logits, label)

    avg_loss = total_loss / len(dataloader)
    print(f'Validation Loss: {avg_loss}')

    return avg_loss


def test_linear_classifier(dataloader_test, classifier_checkpoint_path, num_classes):

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device_name = 'cpu'
    device = torch.device(device_name)
    print(f'Using device: {device_name}')
    
    prediction_steps = 5
    num_neg_samples = 100
    sample_shape = tuple(next(iter(dataloader_test))[0].shape)
    vit_patch_size = sample_shape[5] // 2

    encoder_net = VisionTransformer(img_size=sample_shape[5], patch_size=vit_patch_size, in_chans=input_channels, num_classes=10, embed_dim=512, depth=12,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='')
    
    autoreg_net = None

    cpcv1 = CPCVIT(encoder_net=encoder_net, autoreg_net=autoreg_net, num_pred_steps=prediction_steps,
                num_neg_samples=num_neg_samples, device=device,  learning_mode='linear-classification', num_classes=10)
    # load the saved weights from classifier checkpoint
    classifier_saved = torch.load(classifier_checkpoint_path)
    cpcv1.load_state_dict(classifier_saved['model_state_dict'])

    # eval
    eval_correct = 0
    with torch.no_grad():
        for sample in dataloader_test:
            image, label = sample
            # label.to(device)
            class_out = cpcv1(image.to(device)).view(num_classes)
            co_am = torch.argmax(class_out).item()
            if co_am == label:
                eval_correct += 1
    accuracy = float(eval_correct / len(dataloader_test))
    print(f'After training, linear classification test accuracy = {accuracy}')



if __name__ == '__main__':
    random_seed = 15009
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    checkpoints_dir = 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)

    batch_size = 64
    is_grayscale = False
    dataloader_train,  dataloader_val= make_cifar10_dataloader(batch_size, is_grayscale)

    dataloader_test = make_cifar10_dataloader(batch_size=1, is_grayscale=is_grayscale, is_train=False)

    num_epochs = 30
    learning_rate = 1e-4
    input_channels = 3 if is_grayscale == False else 1
    experiment_name = "rgb-efficient-net"
    save_period_epochs = 2


    ss_checkpoint_path = f'{checkpoints_dir}/best_model_vit_rgb-self-supervised_.pth'
    num_classes = 10
    train_linear_classifier(dataloader_train, ss_checkpoint_path, num_classes, num_epochs, learning_rate=learning_rate,
          input_channels=input_channels, experiment_name=experiment_name, save_period_epochs=save_period_epochs)
    
    #################################
    classifier_checkpoint_path = f'{checkpoints_dir}/best_model_vit_{experiment_name}.pth'
    test_linear_classifier(dataloader_test, classifier_checkpoint_path, num_classes)