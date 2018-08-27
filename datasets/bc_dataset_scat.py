import os
import numpy as np
# import torchvision.transforms as transforms

from torch.utils import data

from .transforms import get_train_transform

from .utils import IMG, LABEL, numpy_one_hot, mix, tensor_to_numpy

# add path to scatnet_python module
from .path import PATH_TO_MODULE
import sys
sys.path.append(PATH_TO_MODULE)
from scatnet_python import *


class BCDatasets(data.Dataset):
    def __init__(self, data_path, dataset_name,
                 sr, exclude, transform=get_train_transform(),
                 scattering_time_transform = True, 
                 averaging_window = 2**10,
                 signal_length = 2**16,
                 mix=False, precision=np.float32):
        self.transform = transform
        self.scattering_time_transform = scattering_time_transform
        self.sr = sr
        self.mix = mix
        self.precision = precision
        data_set = np.load(os.path.join(data_path, dataset_name, 'wav{}.npz'.format(sr // 1000)))

        self.X = []
        self.y = []
        for fold_name in data_set.keys():
            if int(fold_name[4:]) in exclude:
                continue

            sounds = data_set[fold_name].item()['sounds']
            labels = data_set[fold_name].item()['labels']

            self.X.extend(sounds)
            self.y.extend(labels)

        self.n_classes = len(set(self.y))
        
#         print(np.max(list(map(len, self.X))))
#         print(np.mean(list(map(len, self.X))))
#         print(np.min(list(map(len, self.X))))
    
        if self.scattering_time_transform:
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
            self.signal_length = signal_length
            self.Wop, _ = wavelet_factory_1d(self.signal_length,
                                             self.filt_opt_bank, self.scat_opt)
    
    
    
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __do_transform(self, x, y):
        y = numpy_one_hot(y, num_classes=self.n_classes)

        x = x.astype(self.precision)
        sample = {IMG: x, LABEL: y}
        if self.transform:
            sample[IMG] = sample[IMG].reshape((1, -1, 1))
            sample = self.transform(sample)
            sample[IMG] = tensor_to_numpy(sample[IMG])
        
        # sample[IMG] has shape (1, 1, signal_length)
        # (batch_size, H, signal_length)
        self.signal_length = sample[IMG].shape[-1]
        if self.scattering_time_transform:
            # z should have the shape (signal_length, 1, batch_size)
            # We do need reshape!            
            z = sample[IMG]
            z = z.reshape((self.signal_length, 1, 1))
        
            S, _ = scat(z, self.Wop)
            S_table, _ = format_scat(log_scat(renorm_scat(S)), format_type = 'table')
            
            # S_table has the shape (scale_count, time_count, 1)
            # (scale_count, time_count, batch_size)
            # We do need reshape in (scale_count, 1, batch_size)
            S_table = S_table[:,:,-1,:]
            shape = S_table.shape
            S_table = np.reshape(S_table, (shape[0], shape[2], shape[1]))
            
            sample[IMG] = S_table
        return sample
    

    def __mix_samples(self, sample1, sample2):
        r = np.random.uniform()

        sound1 = sample1[IMG].reshape((-1))
        sound2 = sample2[IMG].reshape((-1))

        sound = mix(sound1, sound2, r, self.sr)
        label = r * sample1[LABEL] + (1.0 - r) * sample2[LABEL]

        return {IMG: sound, LABEL: label}

    def __getitem__(self, index):
        if self.mix*False:
            idx1, idx2 = np.random.choice(len(self), 2, replace=False)

            sound1, label1 = self.X[idx1], self.y[idx1]
            sound2, label2 = self.X[idx2], self.y[idx2]

            sample1 = self.__do_transform(sound1, label1)
            sample2 = self.__do_transform(sound2, label2)

            sample = self.__mix_samples(sample1, sample2)
        else:
            sample = self.__do_transform(self.X[index], self.y[index])

#         sample[IMG] = sample[IMG].reshape((1, 1, -1))

        return sample

# if __name__== "__main__":
#     data_path = "/home/julia/DeepVoice_data/ESC"
#     dataset_name="esc10"
#     sr=16000
#     exclude=[5]
#     dataset = BCDatasets(data_path, dataset_name, sr, exclude, scattering_time_transform = False)
#     print(dataset[0])
