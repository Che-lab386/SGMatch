"""Second-stage preprocessing for the DUO-FM complex-Q branch.

This file is intentionally separate from preprocess.py. It assumes the original
mesh normalization and standard/elastic preprocessing have already completed.
"""
import argparse
import os
from glob import glob
import numpy as np
import torch
from tqdm import tqdm

from utils.shape_util import read_shape
from utils.geometry_util import get_operators
from utils.connection_laplacian import compute_connection_cache


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess connection operators for complex-Q')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--n_eig', type=int, default=200)
    parser.add_argument('--n_conn', type=int, default=20)
    parser.add_argument('--standard_cache', default=None)
    parser.add_argument('--complex_cache', default=None)
    args = parser.parse_args()

    assert os.path.isdir(args.data_root), f'Invalid data root: {args.data_root}'
    off_files = sorted(glob(os.path.join(args.data_root, 'off', '*.off')))
    assert off_files, f'No OFF files under {args.data_root}/off'
    standard_cache = args.standard_cache or os.path.join(args.data_root, 'diffusion')
    complex_cache = args.complex_cache or os.path.join(args.data_root, 'complex')
    os.makedirs(complex_cache, exist_ok=True)

    for off_file in tqdm(off_files):
        verts, faces = read_shape(off_file)
        verts_t = torch.from_numpy(np.asarray(verts)).float()
        faces_t = torch.from_numpy(np.asarray(faces)).long()
        _, mass, _, _, _, _, _ = get_operators(
            verts_t, faces_t, k=args.n_eig, cache_dir=standard_cache)
        compute_connection_cache(
            verts_t, faces_t, mass, k=args.n_conn, cache_dir=complex_cache)

    print(f'Complex-Q preprocessing complete: {len(off_files)} meshes')
