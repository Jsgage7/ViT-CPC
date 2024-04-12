# CPC with Vision Transformers
#### Repository for the paper [Uncovering the Potential of Contrastive Predictive Coding: A Comparative Study of Transformers and ResNets](https://osf.io/preprints/osf/az2j8)

**Abstract**: Data-efficient encoding of visual data is important when labeled data can be computationally expensive and time consuming to obtain. While there are several methods for efficient encoding, none use the Transformer architecture for image recognition. The current study proposes an efficient ViT paradigm that outperformed most others in accuracy and amount of labeled data required. Modern ResNet architectures also improved on previous results. Various data augmentation techniques are discussed.

## Brief Repo Walkthrough

### Pixel CNN implementation:
The implementation for PixelCNN and other autoregressors are in /models/autoregressors/. These methods are not setup to be run independently.

### InfoNCE:
The loss module that encoders utilize is /loss/**InfoNCE.py**, this is also not a standalone module.

### CPC:
utils/**patches.py** Used for data augmentation, creating patches
utils/**data.py** Used to produce dataloaders
The main CPC module that ties together the pixelCNN implementation with the infoNCE loss is cpc.py in /models/**cpc.py**. This file can be run on it's own to train a CPC encoder on CIFAR-10.


### Main Experiment Scripts:
The training files that were utilized in our experiments resided in /scripts/. These are meant to be run independently. Some of the relevant files to be run were:

**train_supervised_baseline.py** Provides a baseline ResNet-50 supervised learning

**train_self_supervised.py** Basic training loop for self-supervised learning

**train_linear_classification** Module for linear classification

**train_cpc_resnetv1.py** Implemented a CPC encoder using an original ResNet Architecture

**train_vitcpc.py** (and all derivatives, eg vitcpc_fixed) Provided encoding using a Vision Transformer on CIFAR-10 

**train_efficient_classification.py** Used to train the efficient CPC encoder on CIFAR-10

### Other files:
/models/benchmarks/**simclr_bench.ipynb** is a standalone jupyter notebook that can be run cell by cell to produce a benchmark accuracy that was used in results to compare with ViT performance.


## To Run Locally:
1. Clone the repository using `git clone https://github.com/Jsgage7/ViT-CPC.git`
2. Install the requirements using `pip install -r requirements.txt`
3. Run the desired script in /scripts/ to train the model of interest. Data should be downloaded automatically; be aware that the CIFAR datasets are approximately 160MB each.