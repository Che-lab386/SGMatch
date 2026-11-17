"""
离线预计算 connection Laplacian 缓存。

使用方法：
    python precompute_connection.py --data_dir data/SMAL_r/shapes_train --k 30 --out_dir data/SMAL_r/operators_cache/complex
    python precompute_connection.py --data_dir data/SMAL_r/shapes_test  --k 30 --out_dir data/SMAL_r/operators_cache/complex

这样训练/测试时 load_connection_cache() 直接读缓存，不需要在线计算。
"""

import os
import sys
import glob
import argparse
import numpy as np
import igl

# 加入项目根目录
sys.path.insert(0, os.path.dirname(__file__))
from utils.connection_laplacian import load_connection_cache


def precompute(data_dir, k, out_dir):
    """遍历目录下所有 .off/.ply/.obj，计算并缓存 connection Laplacian。"""
    patterns = ['*.off', '*.ply', '*.obj']
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(os.path.join(data_dir, p))))

    if not files:
        print(f'[ERROR] No mesh files found in {data_dir}')
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f'Found {len(files)} meshes. Output dir: {out_dir}')

    failed = []
    for i, f in enumerate(files):
        name = os.path.basename(f)
        try:
            verts, faces = igl.read_triangle_mesh(f)
            verts = verts.astype(np.float64)
            faces = faces.astype(np.int32)
            import torch
            verts_t = torch.tensor(verts, dtype=torch.float64)
            faces_t = torch.tensor(faces, dtype=torch.int64)
            load_connection_cache(verts_t, faces_t, k=k, cache_dir=out_dir)
            print(f'  [{i+1}/{len(files)}] done: {name}')
        except Exception as e:
            print(f'  [{i+1}/{len(files)}] FAILED: {name}: {e}')
            failed.append(name)

    print(f'\nComplete. {len(files)-len(failed)}/{len(files)} succeeded.')
    if failed:
        print('Failed meshes:')
        for f in failed:
            print(f'  {f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory with .off/.ply/.obj meshes')
    parser.add_argument('--k', type=int, default=30, help='Number of connection basis vectors')
    parser.add_argument('--out_dir', required=True, help='Output cache directory')
    args = parser.parse_args()
    precompute(args.data_dir, args.k, args.out_dir)
