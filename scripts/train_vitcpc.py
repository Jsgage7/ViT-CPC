import datetime
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import time
import numpy as np
import torch
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from models.cpc import CPCVIT
from timm.models.layers import PatchEmbed
from models.encoder.vit import VisionTransformer
from models.autoregressor.pixel_cnn_wfalcon import PixelCNNWFalcon
from utils.patches import MakePatches



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

    # device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_name = 'cpu'
    device = torch.device(device_name)
    print(f'Using device: {device_name}')

    prediction_steps = 5
    num_neg_samples = 100

    sample_shape = tuple(next(iter(dataloader))[0].shape)
    vit_patch_size = sample_shape[5] // 2

    encoder_net = VisionTransformer(img_size=sample_shape[5], patch_size=vit_patch_size, in_chans=input_channels, num_classes=10, embed_dim=512, depth=12,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='')
    
    autoreg_net = PixelCNNWFalcon(input_channels=encoder_net.encoding_num_features)
    cpcv1 = CPCVIT(encoder_net=encoder_net, autoreg_net=autoreg_net, num_pred_steps=prediction_steps,
                  num_neg_samples=num_neg_samples, device=device, learning_mode='self-supervised', num_classes=10)

    grad_params = filter(lambda p: p.requires_grad, cpcv1.parameters())
    optimizer = optim.Adam(grad_params, lr=learning_rate)

    checkpoints_dir = 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)

    num_batches = len(dataloader)
    # use the last 10% of batches per epoch to calculate average training loss
    epoch_training_loss_frac = 0.1
    batch_start_loss_epoch = round((1 - epoch_training_loss_frac) * num_batches)

    best_model_loss = float('inf')
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
            loss = cpcv1(patches.to(device))  #vit expects 4 dim hence patches shape
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0 or i >= batch_start_loss_epoch:
                print(f"[{i+1}/{num_batches}] Batch Loss: {loss.item()}")

            if i >= batch_start_loss_epoch:
                loss_epoch_train += loss.item()

        epoch_duration = time.time() - start_time
        avg_epoch_loss_train = loss_epoch_train / (num_batches - batch_start_loss_epoch)
        print(f'End of epoch. Took {epoch_duration}, approx. avg epoch training loss = {avg_epoch_loss_train}\n')
        
        avg_val_loss = validate(dataloader_unsupervised_val, cpcv1, device)
        
        # Save checkpoint
        if avg_val_loss < best_model_loss:
            best_model_loss = avg_val_loss
            exp_name_file_tag = f'{experiment_name}_' if experiment_name is not None else ''
            # checkpoint_file = f'cpcv1_{exp_name_file_tag}{train_start_display}_epoch_{epoch_num}.pth'
            checkpoint_file = f'best_model_vit_{exp_name_file_tag}.pth'
            torch.save({
                'epoch_num': epoch_num,
                'enc_state_dict': cpcv1.encoder_net.state_dict(),
                'model_state_dict': cpcv1.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_train_loss': avg_epoch_loss_train,
                'avg_val_loss': avg_val_loss,
            }, f'{checkpoints_dir}/{checkpoint_file}')
            print(f'Saved checkpoint {checkpoint_file}')


def validate(dataloader, model, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            patches, label = batch
            patches = patches.to(device)
            loss = model(patches)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Validation Loss: {avg_loss}')

    return avg_loss


if __name__ == '__main__':
    random_seed = 10000
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    batch_size = 64
    num_workers = 2
    download_data = False
    data_root = 'data/cifar10/'

    args = {
      'crop_size': 32,
      'crop_padding': 0,
      'patch_size': 8,
      'p_horiz_flip': 0.5,
      'is_grayscale': False
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


    val_size = 5000
    train_size = len(dataset_unsupervised) - val_size
    dataset_unsupervised,dataset_unsupervised_val = random_split(dataset_unsupervised, [train_size, val_size])
    

    dataloader_unsupervised = DataLoader(dataset_unsupervised, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers)
    
    dataloader_unsupervised_val = DataLoader(dataset_unsupervised_val, batch_size=batch_size, shuffle=True,
                                        num_workers=num_workers)

    # dataset_unsupervised (tuple(tuple)) (int, (torch.Size([7, 7, 1, 8, 8]), int)
    num_epochs = 40
    learning_rate = 1e-4
    input_channels = 3 if args['is_grayscale'] == False else 1
    experiment_name = "rgb-self-supervised"
    train(dataloader_unsupervised, num_epochs, learning_rate=learning_rate, input_channels=input_channels, experiment_name=experiment_name)