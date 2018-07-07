BC learning for sounds PyTorch Port
============================
This is the port of [Between-class Examples for Deep Sound Recognition](https://github.com/mil-tokyo/bc_learning_sound) to PyTorch.
[Dataset generation](https://github.com/rerayne/bc_learning_pytorch/tree/master/dataset_gen) was taken from the [original repo](https://github.com/mil-tokyo/bc_learning_sound).

Implementation of [Learning from Between-class Examples for Deep Sound Recognition](https://arxiv.org/abs/1711.10282) by Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada (ICLR 2018).

This also contains training of EnvNet: [Learning Environmental Sounds with End-to-end Convolutional Neural Network](http://ieeexplore.ieee.org/document/7952651/) (Yuji Tokozume and Tatsuya Harada, ICASSP 2017).<sup>[1](#1)</sup>


## Contents

- Between-class (BC) learning
	- We generate between-class examples by mixing two training examples belonging to different classes with a random ratio.
	- We then input the mixed data to the model and
train the model to output the mixing ratio.
- Training of EnvNet on ESC-50, ESC-10 [[1]](#1), and UrbanSound8K [[2]](#2) datasets

## Setup
- Install [PyTorch](https://pytorch.org/).
- Prepare datasets following [this page](https://github.com/rerayne/bc_learning_pytorch/tree/master/dataset_gen).


## Training
- Template:

		python main.py --dataset [esc50, esc10, or urbansound8k] --netType [envnet or envnetv2] --data path/to/dataset/directory/ (--BC) (--strongAugment)
 
- Recipes:
	- Standard learning of EnvNet on ESC-50 (around 29% error<sup>[2](#2)</sup>):

			python main.py --dataset esc50 --netType envnet --data path/to/dataset/directory/
	
- Notes:
	- Please check [opts.py](https://github.com/rerayne/bc_learning_pytorch/blob/master/opts.py) for other command line arguments.


## See also
[Between-class Learning for Image Clasification](https://arxiv.org/abs/1711.10284) ([github](https://github.com/mil-tokyo/bc_learning_image))


#### Reference
<i id=1></i>[1] Karol J Piczak. Esc: Dataset for environmental sound classification. In *ACM Multimedia*, 2015.

<i id=2></i>[2] Justin Salamon, Christopher Jacoby, and Juan Pablo Bello. A dataset and taxonomy for urban sound research. In *ACM Multimedia*, 2014.
