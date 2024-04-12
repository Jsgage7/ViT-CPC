# Contrastive Predictive Coding Project Notes

### Link to Latex report:
- https://www.overleaf.com/7684489573myqndppgzvgm

## papers
- CPCv2
	- https://arxiv.org/abs/1905.09272
- CPCv1
	- van den Oord et al
	- https://arxiv.org/abs/1807.03748
- PixelCNN
	- "Pixel Recurrent Neural Networks" https://arxiv.org/abs/1601.06759
- PixelCNN++
	- we could compare PixelCNN and PixelCNN++. maybe implement PixelCNN ourselves, and compare against the `pclucas14/pixel-cnn-pp` impl
	- [An Explanation of Discretized Logistic Mixture Likelihood](https://medium.com/@smallfishbigsea/an-explanation-of-discretized-logistic-mixture-likelihood-bdfe531751f0)

- ResNet-v2
	- He et al 2016
- Greedy InfoMax
	- https://paperswithcode.com/paper/greedy-infomax-for-biologically-plausible

## code / articles / tutorials
- [Demystifying Noise Contrastive Estimation](https://jxmo.io/posts/nce)
- rschwarz15 impl of CPCv1, v2
	- https://github.com/rschwarz15/CPCV2-PyTorch
	- two PixelCNN variants are available
		- one adapted from PixelCNN Greedy InfoMax variant
		- one adapted from Pytorch Lightning impl
	- TODO finish reading impl
- PixelCNN explainers:
	- https://bjlkeng.github.io/posts/pixelcnn/
	- https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173
- PixelCNN Greedy_InfoMax variant
	- https://github.com/loeweX/Greedy_InfoMax/tree/master
	- https://github.com/loeweX/Greedy_InfoMax/blob/master/GreedyInfoMax/vision/models/PixelCNN.py
	- https://github.com/loeweX/Greedy_InfoMax/blob/master/GreedyInfoMax/vision/models/PixelCNN_Autoregressor.py
- PixelCNN Pytorch Lightning impl
	- https://github.com/Lightning-Universe/lightning-bolts/blob/master/pl_bolts/models/vision/pixel_cnn.py
- a third PixelCNN impl
	- https://github.com/pclucas14/pixel-cnn-pp
- https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html#
	- from the University of Amsterdam Deep Learning class
	- not Contrastive Predictive Coding, but SimCLR, so it's a related contrastive self-supervised learning method
	- TODO skim for tips on how to do this project.
		- for example, they use tensorboard, which might be worth setting up for experiments
		-
		  > One thing to note is that contrastive learning benefits a lot from long
		  training. The shown plot above is from a training that took approx. 1
		  day on a NVIDIA TitanRTX.
		-
		  > A common observation in contrastive learning is that the larger the
		  batch size, the better the models perform. A larger batch size allows us
		  to compare each image to more negative examples, leading to overall
		  smoother loss gradients. However, in our case, we experienced that a
		  batch size of 256 was sufficient to get good results.

- Greedy InfoMax
	- (maybe a tangent, but this caught my eye)
	- "Putting An End to End-to-End: Gradient-Isolated Learning of Representations" https://arxiv.org/abs/1905.11786
	- cites van den Oord et al paper, used in PixelCNN variant in rschwarz15's impl
	- also uses InfoNCE, just like CPC. compares performance against CPC
	- ![](https://loewex.github.io/GIM_media/LatentClassification.png)
	- blog introduction:
		- https://loewex.github.io/GreedyInfoMax.html
		-
		  > ... Our brains are extremely efficient learning machines. One possible explanation for this is that **biological synapses are mostly adjusted based on local information, i.e. based on information from their immediate neighboring neurons. They do not wait for a global signal to update their connection strengths.** Thus, different areas of the brain can learn highly asynchronously and in parallel.
		  >
		  > Inspired by this local learning found in the brain, we propose a new learning algorithm – Greedy InfoMax. With Greedy Infomax, we show that
		  **we can train a neural network _without_ end-to-end backpropagation** and achieve competitive performance. At the same time, Greedy InfoMax
		  makes use of a self-supervised loss function.
	- experimental setup:
	-
	  > We focus on the STL-10 dataset [Coates et al., 2011] which provides an additional unlabeled training dataset. For data augmentation, we take random 64 × 64 crops from the 96 × 96 images, flip horizontally with probability 0.5 and convert to grayscale. We divide each image of 64 × 64 pixels into a total of 7 × 7 local patches, each of size 16 × 16 with 8 pixels overlap. The patches are encoded by a ResNet-50 v2 [He et al., 2016] without batch normalization [Ioffe and Szegedy, 2015].
