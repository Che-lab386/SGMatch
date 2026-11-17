"""
Connection Laplacian 最小实现 - 基于 DUO-FM (Deep Orientation-Aware Functional Maps)

论文出处:
- Donati et al., 2022: Deep Orientation-Aware Functional Maps (CVPR)
- Sharp et al., 2019: The Vector Heat Method

connection Laplacian 用于处理切向量场。与标准 Laplacian 不同，它沿网格边考虑平行移动。

当前实现限制：仅支持闭合无边界网格。
"""

import os
import hashlib
import numpy as np
import scipy.sparse as sp
import torch
import igl


def _get_cache_key(verts_np, faces_np, k):
    """构造基于网格与参数的缓存键。"""
    verts_bytes = verts_np.astype(np.float32).tobytes()
    faces_bytes = faces_np.astype(np.int32).tobytes()
    combined = verts_bytes + faces_bytes + str(k).encode()
    return hashlib.sha1(combined).hexdigest()


def _build_halfedges(faces_np):
    """构建 halfedge 结构，兼容 DUO-FM 实现。"""
    faces_np = np.asarray(faces_np, dtype=np.int32)
    # 展开所有边
    edges = faces_np[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    
    # 构建边到 halfedge 索引的映射
    edge_to_halfedge = {}
    for index, (a, b) in enumerate(edges):
        edge_to_halfedge.setdefault((int(a), int(b)), index)
    
    # 查找对边（opposite halfedge）
    opposite = np.full(edges.shape[0], -1, dtype=np.int64)
    for index, (a, b) in enumerate(edges):
        opposite[index] = edge_to_halfedge.get((int(b), int(a)), -1)
    
    if np.any(opposite < 0):
        raise ValueError(
            'Connection Laplacian requires a closed manifold mesh; '
            'boundary halfedge found. Partial meshes are not supported.'
        )
    
    # 下一个 halfedge（在同一面内的循环）
    next_halfedge = np.arange(faces_np.shape[0] * 3, dtype=np.int64)
    next_halfedge = (next_halfedge // 3) * 3 + (next_halfedge % 3 + 1) % 3
    
    # 每个顶点的起始 halfedge（用于构建切标架）
    _, he_start = np.unique(edges[:, 0], return_index=True)
    
    return edges, opposite, next_halfedge, he_start


def compute_connection_laplacian(verts_np, faces_np):
    """
    计算 connection Laplacian 矩阵 Ψ。
    
    Args:
        verts_np: 顶点 [V, 3]
        faces_np: 面片 [F, 3]
    
    Returns:
        scipy.sparse matrix: connection Laplacian [V, V] with complex dtype
    """
    # 步骤 1：构建 halfedge 结构
    edges, op, nex, he_start = _build_halfedges(faces_np)
    
    # 步骤 2：计算内角和 Gaussian 曲率（角度缺陷）
    angles = igl.internal_angles(verts_np, faces_np)  # [F, 3]
    K = igl.gaussian_curvature(verts_np, faces_np)    # [V]
    
    # 步骤 3：构造 halfedge 角度数组
    he_angles = np.concatenate([angles[:, 1], angles[:, 2], angles[:, 0]])
    he_angles[he_start] = 0  # 基点角度设为 0
    
    vert_angle_sum = 2 * np.pi - K
    he_angles_norm = 2 * np.pi * he_angles / vert_angle_sum[edges[:, 0]]
    
    # 步骤 4：计算累积角度（绕每个顶点）
    rotate = np.zeros(verts_np.shape[0], dtype=int)
    he = np.copy(he_start)
    last_he = he
    i = 0
    while np.any(rotate == 0):
        he = op[he]
        he = nex[he]
        i += 1
        
        rot_mask = (he_start == he) * (rotate == 0)
        rotate[rot_mask] = i
        he_angles_norm[he[rotate == 0]] += he_angles_norm[last_he[rotate == 0]]
        last_he = he
    
    # 步骤 5：计算平行移动旋转角
    rho = (he_angles_norm[op] + np.pi) - he_angles_norm
    r = np.cos(rho) + np.sin(rho) * 1j  # 复数旋转
    r_op = r[op]
    
    # 步骤 6：构建 connection Laplacian 矩阵
    r = r.reshape(faces_np.shape[0], 3)
    r_op = r_op.reshape(faces_np.shape[0], 3)
    
    # cotangent 权重乘以复数旋转
    cot_ = 0.5 / np.tan(angles)
    cot = cot_ * r
    cot_op = cot_ * r_op
    
    S_ = np.concatenate([cot_[:, 2], cot_[:, 0], cot_[:, 1]])
    S = np.concatenate([cot[:, 2], cot[:, 0], cot[:, 1]])
    S_op = np.concatenate([cot_op[:, 2], cot_op[:, 0], cot_op[:, 1]])
    
    I = np.concatenate([faces_np[:, 0], faces_np[:, 1], faces_np[:, 2]])
    J = np.concatenate([faces_np[:, 1], faces_np[:, 2], faces_np[:, 0]])
    In = np.concatenate([I, J, I, J])
    Jn = np.concatenate([J, I, I, J])
    Sn = np.concatenate([-S_op, -S, S_, S_])
    
    A = sp.csr_matrix((Sn, (In, Jn)), shape=(verts_np.shape[0], verts_np.shape[0]))
    return A


def compute_grad_vert_op(verts_np, faces_np, edges, op, nex, he_start, he_angles_norm, K):
    """
    计算梯度算子 ∇_X 的 [2V, V] 稀疏表示（实部 x、虚部 y 堆叠）。
    
    返回 gradv，使得对标量函数 f，gradv @ f 的 [0::2] 是 x 分量，[1::2] 是 y 分量。
    """
    I, J, V = [], [], []
    
    # 步骤 1：收集邻域坐标
    Vjs = []
    rotate = np.zeros(verts_np.shape[0], dtype=int)
    he = np.copy(he_start)
    i = 0
    while np.any(rotate == 0):
        lij = np.linalg.norm(verts_np[edges[he, 1]] - verts_np[edges[he, 0]], axis=1)
        aij = he_angles_norm[he]
        vj = lij[:, None] * np.cos(np.stack([aij, np.pi/2 - aij], axis=-1))
        vj[rotate > 0] = 0
        Vjs.append(vj)
        
        he = op[he]
        he = nex[he]
        i += 1
        
        rot_mask = (he_start == he) * (rotate == 0)
        rotate[rot_mask] = i
    
    # 步骤 2：构建并反演局部系统
    Vjs = np.stack(Vjs, axis=1)
    Vjs_inv = np.linalg.pinv(Vjs)
    
    # 步骤 3：填充系数
    rotate = np.zeros(verts_np.shape[0], dtype=int)
    he = np.copy(he_start)
    i = 0
    while np.any(rotate == 0):
        jdv = edges[he, 1]
        idv = edges[he, 0]
        I.extend([2*idv, 2*idv, 2*idv+1, 2*idv+1])
        J.extend([idv, jdv, idv, jdv])
        V.extend([-Vjs_inv[:, 0, i], Vjs_inv[:, 0, i],
                  -Vjs_inv[:, 1, i], Vjs_inv[:, 1, i]])
        
        he = op[he]
        he = nex[he]
        i += 1
        
        rot_mask = (he_start == he) * (rotate == 0)
        rotate[rot_mask] = i
    
    I = np.concatenate(I)
    J = np.concatenate(J)
    V = np.concatenate(V)
    
    gradv = sp.csr_matrix((V, (I, J)), shape=(2 * verts_np.shape[0], verts_np.shape[0]))
    return gradv


def compute_spectral_gradient(verts_np, faces_np, ceig, m):
    """
    计算谱梯度：spec_grad = Ψ^H @ ∇
    
    Ψ^H 是 connection 基的共轭转置投影；∇ 是梯度算子。
    结果形状为 [K_c, V]，用于把逐点标量特征投影为 connection 谱系数。
    
    Args:
        verts_np: 顶点 [V, 3]
        faces_np: 面片 [F, 3]
        ceig: connection 基 [V, K_c] (complex)
        m: 质量矩阵 [V, V]
    
    Returns:
        spec_grad: torch.Tensor [K_c, V] (complex128)
    """
    # 构建 Hermitian 转置：Ψ^H = (M @ Ψ)^H
    ceig_trans = np.conj(m @ ceig).T  # [K_c, V]
    
    # 构建 halfedge 结构
    edges, op, nex, he_start = _build_halfedges(faces_np)
    
    # 计算角度和曲率
    angles = igl.internal_angles(verts_np, faces_np)
    he_angles = np.concatenate([angles[:, 1], angles[:, 2], angles[:, 0]])
    he_angles[he_start] = 0
    K = igl.gaussian_curvature(verts_np, faces_np)
    
    vert_angle_sum = 2 * np.pi - K
    he_angles_norm = 2 * np.pi * he_angles / vert_angle_sum[edges[:, 0]]
    
    # 累积角度
    rotate = np.zeros(verts_np.shape[0], dtype=int)
    he = np.copy(he_start)
    last_he = he
    i = 0
    while np.any(rotate == 0):
        he = op[he]
        he = nex[he]
        i += 1
        
        rot_mask = (he_start == he) * (rotate == 0)
        rotate[rot_mask] = i
        he_angles_norm[he[rotate == 0]] += he_angles_norm[last_he[rotate == 0]]
        last_he = he
    
    # 计算梯度算子
    gradv = compute_grad_vert_op(verts_np, faces_np, edges, op, nex, he_start, he_angles_norm, K)
    
    # 转换为复数梯度：[2V, V] → [V, V] complex
    gradv_complex = gradv[0::2, :].toarray() + 1j * gradv[1::2, :].toarray()
    
    # 投影到 connection 谱基：spec_grad = Ψ^H @ ∇_complex
    spec_grad = ceig_trans @ gradv_complex  # [K_c, V]
    
    return torch.tensor(spec_grad, dtype=torch.complex128)


def load_connection_cache(verts, faces, k=30, cache_dir=None):
    """
    加载或计算 connection Laplacian 的缓存。
    
    Args:
        verts: 顶点 torch.Tensor [V, 3] 或 numpy
        faces: 面片 torch.Tensor [F, 3] 或 numpy  
        k: connection 基的维数
        cache_dir: 缓存目录（可选）
    
    Returns:
        dict 包含：
            'conn_evals': connection 特征值 [K_c]
            'conn_evecs': connection 特征向量 [V, K_c] (complex128)
            'spec_grad': 谱梯度 [K_c, V] (complex128)
    """
    # 转换为 numpy
    if isinstance(verts, torch.Tensor):
        verts = verts.cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()
    
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    
    cache_key = _get_cache_key(verts, faces, k)
    cache_file = None
    
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'connection_{cache_key}_{k}.npz')
        
        if os.path.exists(cache_file):
            data = np.load(cache_file, allow_pickle=True)
            return {
                'conn_evals': torch.tensor(data['conn_evals'], dtype=torch.complex128),
                'conn_evecs': torch.tensor(data['conn_evecs'], dtype=torch.complex128),
                'spec_grad': torch.tensor(data['spec_grad'], dtype=torch.complex128),
            }
    
    # 计算 connection Laplacian
    print(f"[connection_laplacian] 计算 connection Laplacian （K={k}）...")
    cl = compute_connection_laplacian(verts, faces)
    m = igl.massmatrix(verts, faces, igl.MASSMATRIX_TYPE_VORONOI)
    
    # 特征分解
    cvals, ceig = sp.linalg.eigsh(cl, k, m, sigma=0, which="LM")
    cvals = np.asarray(cvals, dtype=np.complex128)
    ceig = np.asarray(ceig, dtype=np.complex128)
    
    # 计算谱梯度
    spec_grad = compute_spectral_gradient(verts, faces, ceig, m)
    
    # 缓存
    result = {
        'conn_evals': torch.tensor(cvals, dtype=torch.complex128),
        'conn_evecs': torch.tensor(ceig, dtype=torch.complex128),
        'spec_grad': spec_grad,
    }
    
    if cache_file is not None:
        np.savez(cache_file, 
                conn_evals=cvals, 
                conn_evecs=ceig, 
                spec_grad=spec_grad.cpu().numpy())
        print(f"[connection_laplacian] 缓存已保存至 {cache_file}")
    
    return result
