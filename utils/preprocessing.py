# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import spsolve


class Normalisation(object):
    """
    Time series normalisation.
    """
    def __init__(self, mode="group_wise", groups=[3, 6, 12]) -> None:
        self.mode = mode
        self.groups = groups

    def __call__(self, sample) -> np.array:
        sample_dtype = sample.dtype

        if self.mode == "sample_wise":
            mean = np.mean(sample)
            var = np.var(sample)
        
        elif self.mode == "channel_wise":
            mean = np.mean(sample, axis=-1, keepdims=True)
            var = np.var(sample, axis=-1, keepdims=True)
        
        elif self.mode == "group_wise":
            mean = []
            var = []

            lower_bound = 0
            for idx in self.groups:
                mean_group = np.mean(sample[lower_bound:idx], axis=(0, 1), keepdims=True)
                mean_group = np.repeat(mean_group, repeats=int(idx-lower_bound), axis=0)
                var_group = np.var(sample[lower_bound:idx], axis=(0, 1), keepdims=True)
                var_group = np.repeat(var_group, repeats=int(idx-lower_bound), axis=0)
                lower_bound = idx

                mean.extend(mean_group)
                var.extend(var_group)

            mean = np.array(mean, dtype=sample_dtype)
            var = np.array(var, dtype=sample_dtype)

        normalised_sample = (sample - mean) / (var + 1.e-12)**.5

        return normalised_sample
    

def baseline_als(y, lam=1e8, p=1e-2, niter=10):
    """
    Asymmetric Least Squares Smoothing, i.e. asymmetric weighting of deviations to correct a baseline 
    while retaining the signal peak information.
    Refernce: Paul H. C. Eilers and Hans F.M. Boelens, Baseline Correction with Asymmetric Least Squares Smoothing (2005).
    """
    L = len(y)
    D = sparse.diags([1,-2,1], [0,-1,-2], shape=(L, L-2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


def process_ecg(sample):
    # remove nan
    sample = np.nan_to_num(sample)
    
    # clamp
    sample_std = sample.std()
    sample = np.clip(sample, a_min=-4*sample_std, a_max=4*sample_std)

    # remove baseline wander
    baselines = np.zeros_like(sample)
    for lead in range(sample.shape[0]):
        baselines[lead] = baseline_als(sample[lead], lam=1e7, p=0.3, niter=5)
    sample = sample - baselines

    # normalise 
    transform = Normalisation(mode="group_wise", groups=[3, 6, 12])
    sample = transform(sample)

    return sample


def main():
    data_path = "/home/ukbb/ecgs"   # path to the ecgs
    file_name = "ecg_train"         # ecg

    # process
    processed_sample = process_ecg(os.path.join(data_path, file_name, ".pt"))

    # save
    torch.save(torch.tensor(np.array(processed_sample), dtype=torch.float32), os.path.join(data_path, "processed", f'{file_name}_float32.pt'))


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

    main()
