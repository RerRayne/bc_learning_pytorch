import torch
import torchvision.transforms

from torchvision.transforms import TenCrop, Pad, RandomCrop, ToTensor, ToPILImage, Lambda
from .utils import IMG, LABEL, MAX_INT

# add path to scatnet_python module
from .path import PATH_TO_MODULE
import sys
sys.path.append(PATH_TO_MODULE)
from scatnet_python import *


class Scattering():
    def __init__(self, averaging_window = 2**10,
                 signal_length = 2**16,
                 precision=np.float32):

        self.signal_length = signal_length
        self.precision = precision
        
        # Set filter bank options
        self.averaging_window = averaging_window
        self.filt_opt_bank = [FiltOptions(Q = 8, T = self.averaging_window,
                            filter_type = 'morlet_1d',
                            boundary = 'symm',
                            precision = self.precision),
                         FiltOptions(Q = 1, T = self.averaging_window,
                            filter_type = 'morlet_1d',
                            boundary = 'symm',
                            precision = self.precision)]
        # Set scattering options
        self.scat_opt = ScatOptions(M=2)

        # Create filter bank

        self.Wop, _ = wavelet_factory_1d(self.signal_length,
                                         self.filt_opt_bank, self.scat_opt)
        
    def scattering_transform(self, batch):
        is_batch = False
        
        if len(batch.shape) > 3:
            is_batch = True
            z = (batch[:,-1,:,:])
        else:
            z = batch
        # z should have the shape (signal_length, 1, batch_size)
        # We do need reshape! 
        z = z.T

        S, _ = scat(z, self.Wop)
        
        #S_table, _ = format_scat(log_scat(renorm_scat(S)), format_type = 'table')
        
        # table with scatterin coefficients, sorted by scale lambda1
        S_table = stack_scat(concatenate_freq(log_scat(renorm_scat(S))))

        # S_table has the shape (scale_count, time_count, 1, batch_size)
        # We do need reshape it to (batch_size, scale_count, 1, time_count) if batch_size > 1
        # and to (scale_count, 1, time_count) otherwise

        S_table = np.moveaxis(S_table, [0, 1, 2, 3], [1, 3, 2, 0])

        if not is_batch:
            S_table = S_table[-1,:,:,:]

        batch = S_table
        
        return batch


class Centring(object):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return img / self.factor


class ConvertToTuple(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        img, label = sample[IMG], sample[LABEL]
        return {IMG: self.transform(img), LABEL: label}


def get_train_transform(length = None):
    transforms = [ToPILImage(),
                  Pad((length // 2, 0)),
                  RandomCrop((1, length)),
                  ToTensor(),
                  Centring(MAX_INT)]
    return torchvision.transforms.Compose([ConvertToTuple(default_transforms) for default_transforms in transforms])


def get_test_transform(length=None):
    transforms = [ToPILImage(),
                  Pad((length // 2, 0)),
                  TenCrop((1, length)),
#                   RandomCrop((1, length)),
#                   ToTensor(),
                  Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
                  Centring(MAX_INT)]
    return torchvision.transforms.Compose([ConvertToTuple(default_transforms) for default_transforms in transforms])