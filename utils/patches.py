import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms

# Greedy InfoMax augmentation:
# > We focus on the STL-10 dataset [Coates et al., 2011] which provides an additional
# > unlabeled training dataset. For data augmentation, we take random 64 × 64 crops from the
# > 96 × 96 images, flip horizontally with probability 0.5 and convert to grayscale.
#
def make_gim_augment(crop_size=64, p_horiz_flip=0.5):
    return transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(p_horiz_flip),
        # transforms.Grayscale(),
        transforms.ToTensor()
    ])

def make_overlapping_patches(images, crop_size, patch_size, patch_stride, num_images):
    transform = make_gim_augment(crop_size=crop_size)

    N = len(images)

    # (H, W) = (height, width) of image
    # K = the patch size
    # S = the patch stride
    #
    # we need S to evenly divide each:
    #  - S | H
    #  - S | W
    #  - S | K
    #
    # let:
    # H' = H / S
    # W' = W / S
    # K' = K / S
    #
    # in Greedy InfoMax:
    #   H = W = 64
    #   K = 16
    #   S = 8
    #   H' = W' = 64 / 8 = 8
    #   K' = 16 / 8 = 2
    #
    #   so you can think of it as effectively sliding a 2x2 window over an 8x8 image
    #
    # the number of resulting patches is:
    #  - vertically: H' - K' + 1
    #  - horizontally: W' - K' + 1
    #
    # alt: [(H - K)/S + 1] x [(W - K)/S + 1] patches

    # Validate that parameters satisfy requirements
    if crop_size < patch_size:
        raise Exception('crop_size must be >= patch_size')

    if crop_size % patch_stride != 0 or patch_size  % patch_stride != 0:
        raise Exception('crop_size and patch_size must each evenly divide crop_size')

    num_patches_1d = int((crop_size - patch_size)/patch_stride + 1)

    result_shape = (num_images, 3, num_patches_1d, num_patches_1d, patch_size, patch_size)
    result = torch.zeros(result_shape)

    # TODO: vectorize this. call transform on an entire batch of images, don't loop
    for index in range(num_images):
        image = images[index][0]
        image = transform(image)

        # after the transform, image.shape = (1, 64, 64)
        # print(image.shape)

        # > We divide each image of 64 × 64 pixels into a total of 7 × 7 local patches,
        # > each of size 16 × 16 with 8 pixels overlap.
        patches = image.unfold(1, patch_size, patch_stride).unfold(2, patch_size, patch_stride).contiguous()
        result[index, :, :, :, :, :] = patches

    return result


class MakePatches(object):
    """Make overlapping patches
    """

    def __init__(self, crop_size, patch_size, p_horiz_flip=0.5):
        if patch_size % 2 != 0:
            raise Exception('patch_size must be even')

        self.patch_stride = patch_size // 2

        if crop_size % self.patch_stride != 0 or patch_size  % self.patch_stride != 0:
            raise Exception('crop_size must each evenly divide patch_stride')

        self.crop_size = crop_size
        self.patch_size = patch_size
        self.p_horiz_flip = p_horiz_flip

    def __call__(self, sample):
        # num_patches_1d = 1 + int((self.crop_size - self.patch_size)/self.patch_stride)
        # result.shape = (C, num_patches_1d, num_patches_1d, patch_size, patch_size)

        # sample.shape = torch.Size([3, 32, 32])
        # print(f'sample.shape = {sample.shape}')


        patches = sample.unfold(1, self.patch_size, self.patch_stride).unfold(2, self.patch_size, self.patch_stride).contiguous()
        # patches.shape = torch.Size([3, 7, 7, 8, 8])
        # print(f'patches.shape = {patches.shape}')
        return patches.permute(1, 2, 0, 3, 4).contiguous()



def test_display_stl10(dataset, batch_size=4):
    # root = 'data/stl10/'
    # dataset = torchvision.datasets.STL10(root=root, split='test', transform=None)

    # Get a random sample of 4 images and their labels
    images = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for i in range(batch_size):
        instance = images.dataset[i][0]
        print(f" ::-->> {instance.shape}")
        plt.figure()
        plt.imshow(instance.permute(1, 2, 0).numpy())

    plt.show()

def test_display_stl10_overlapping(images, num_images=3):
    crop_size = 64
    patch_size = 16
    patch_stride = 8

    if crop_size < patch_size:
        raise Exception('crop_size must be >= patch_size')

    if crop_size % patch_stride != 0 or patch_size  % patch_stride != 0:
        raise Exception('patch_stride must evenly divide both crop_size and patch_size')

    num_patches_1d = int((crop_size - patch_size)/patch_stride + 1)

    results = make_overlapping_patches(images, crop_size, patch_size, patch_stride, num_images=num_images)
    print(f"results.shape = {results.shape}")

    # reshape tensor to (10, 7*7, 16, 16)
    N = results.shape[0]
    tensor = results.view(N, -1, patch_size, patch_size)
    images = tensor.numpy()

    for i in range(N):
        fig, axes = plt.subplots(figsize=(6,6), nrows=num_patches_1d, ncols=num_patches_1d)
        for j in range(49):
            ax = axes[j // num_patches_1d, j % num_patches_1d]
            ax.imshow(images[i][j])
            ax.axis('on')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        plt.subplots_adjust(wspace=0, hspace=0)

    plt.show()


def test_display_cifar10_overlapping(images, num_images=3):
    crop_size = 32
    patch_size = 8
    patch_stride = 4

    if crop_size < patch_size:
        raise Exception('crop_size must be >= patch_size')

    if crop_size % patch_stride != 0 or patch_size  % patch_stride != 0:
        raise Exception('patch_stride must evenly divide both crop_size and patch_size')

    num_patches_1d = int((crop_size - patch_size)/patch_stride + 1)

    results = make_overlapping_patches(images, crop_size, patch_size, patch_stride, num_images=num_images)
    print(f"results.shape = {results.shape}")

    # reshape tensor to (10, 7*7, 16, 16)
    N = results.shape[0]
    tensor = results.view(N, -1, patch_size, patch_size)
    images = tensor.numpy()

    for i in range(N):
        fig, axes = plt.subplots(figsize=(6,6), nrows=num_patches_1d, ncols=num_patches_1d)
        for j in range(49):
            ax = axes[j // num_patches_1d, j % num_patches_1d]
            ax.imshow(images[i][j])
            ax.axis('on')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        plt.subplots_adjust(wspace=0, hspace=0)

    plt.show()

def download_data(data_root):
    torchvision.datasets.STL10(root=data_root, split='unlabeled', download=True)



if __name__ == '__main__':
    data_root='data/stl10/'
    download_data(data_root)

    # to_tensor = transforms.Compose([transforms.ToTensor()])
    # images_to_tensor = torchvision.datasets.STL10(root=data_root, split='unlabeled', transform=to_tensor, download=False)
    # test_display_stl10(images_to_tensor)

    # inputs to make_overlapping_patches must be PIL.Image.Image, not a PyTorch tensor
    images = torchvision.datasets.STL10(root=data_root, split='unlabeled', download=False)
    test_display_stl10_overlapping(images)
