"""
Connection Laplacian 实现 - 基于 DUO-FM (Deep Orientation-Aware Functional Maps)

论文出处：
- Sharp et al., 2019: The Vector Heat Method
- Donati et al., 2022: Deep Orientation-Aware Functional Maps (CVPR)

原代码来自: https://github.com/nicolasdonati/DUO-FM/blob/main/Tools/mesh.py

connection Laplacian 是定义在切丛上的算子，用于处理切向量场的 Laplacian 滤波。
与标准 Laplacian 不同的是，它考虑了沿网格边的 parallel transport（平行移动）。
"""

import numpy as np
import scipy.sparse as sp
import torch
import igl
import hashlib
import os


def _build_halfedges(faces_np):
    """Build DUO-FM-compatible directed halfedges without libigl API variance."""
    faces_np = np.asarray(faces_np, dtype=np.int32)
    edges = faces_np[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    edge_to_halfedge = {}
    for index, (a, b) in enumerate(edges):
        edge_to_halfedge.setdefault((int(a), int(b)), index)
    opposite = np.full(edges.shape[0], -1, dtype=np.int64)
    for index, (a, b) in enumerate(edges):
        opposite[index] = edge_to_halfedge.get((int(b), int(a)), -1)
    if np.any(opposite < 0):
        raise ValueError('Connection Laplacian requires a closed manifold mesh; boundary halfedge found')
    next_halfedge = np.arange(faces_np.shape[0] * 3, dtype=np.int64)
    next_halfedge = (next_halfedge // 3) * 3 + (next_halfedge % 3 + 1) % 3
    _, he_start = np.unique(edges[:, 0], return_index=True)
    return edges, opposite, next_halfedge, he_start
def compute_connection_laplacian_numpy(verts_np, faces_np):
    """
    计算 connection Laplacian 矩阵（numpy 版本）

    基于 DUO-FM 的 halfedge 和 parallel transport 实现。
    """
    # ===================== Step 1: 构建 halfedge 结构 =====================
    edges, op, nex, he_start = _build_halfedges(faces_np)

    
    # 计算三角形内角
    angles = igl.internal_angles(verts_np, faces_np)  # [F, 3]
    
    # 将角度展开到 halfedge 数组
    he_angles = angles.reshape(-1).copy()
    he_angles[he_start] = 0  # 基点角度设为 0
    
    # 获取 Gaussian curvature（角度缺陷）
    K = igl.gaussian_curvature(verts_np, faces_np)  # [V]
    
    vert_angle_sum = 2 * np.pi - K
    he_angles_norm = 2 * np.pi * he_angles / vert_angle_sum[edges[:, 0]]
    
    # ===================== Step 3: 计算 cumulative angle sums =====================
    # 绕顶点循环计算累积角度
    rotate = np.zeros(verts_np.shape[0], dtype=int)
    he = np.copy(he_start)
    last_he = he
    i = 0
    while np.any(rotate == 0):
        # 顺时针循环
        he = op[he]
        he = nex[he]
        i += 1
        
        rot_mask = (he_start == he) * (rotate == 0)
        rotate[rot_mask] = i
        he_angles_norm[he[rotate == 0]] += he_angles_norm[last_he[rotate == 0]]
        last_he = he
    
    # ===================== Step 4: 计算 parallel transport 角度 =====================
    # rho 是沿 halfedge 的旋转角
    rho = (he_angles_norm[op] + np.pi) - he_angles_norm
    r = np.cos(rho) + np.sin(rho) * 1j  # complex rotation
    r_op = r[op]
    
    # ===================== Step 5: 构建 connection Laplacian 矩阵 =====================
    # 重塑为三角形索引
    r = r.reshape(faces_np.shape[0], 3)
    r_op = r_op.reshape(faces_np.shape[0], 3)
    
    # cotangent weights with complex rotation
    cot_ = 0.5 / np.tan(angles)
    cot = cot_ * r
    cot_op = cot_ * r_op
    
    # 提取系数
    S_ = np.concatenate([cot_[:, 2], cot_[:, 0], cot_[:, 1]])
    S = np.concatenate([cot[:, 2], cot[:, 0], cot[:, 1]])
    S_op = np.concatenate([cot_op[:, 2], cot_op[:, 0], cot_op[:, 1]])
    
    # 构建稀疏矩阵 (与标准 cotangent Laplacian 类似，但含 complex 系数)
    I = np.concatenate([faces_np[:, 0], faces_np[:, 1], faces_np[:, 2]])
    J = np.concatenate([faces_np[:, 1], faces_np[:, 2], faces_np[:, 0]])
    In = np.concatenate([I, J, I, J])
    Jn = np.concatenate([J, I, I, J])
    Sn = np.concatenate([-S_op, -S, S_, S_])
    
    # 创建 connection Laplacian
    CL = sp.csr_matrix((Sn, (In, Jn)), shape=(verts_np.shape[0], verts_np.shape[0]))
    
    return CL


def _connection_cache_key(verts_np, faces_np, k):
    hasher = hashlib.sha1()
    hasher.update(np.ascontiguousarray(verts_np, dtype=np.float64).tobytes())
    hasher.update(np.ascontiguousarray(faces_np, dtype=np.int32).tobytes())
    hasher.update(str(int(k)).encode('ascii'))
    return hasher.hexdigest()


def load_connection_cache(verts, faces, k=20, cache_dir=None):
    """Load the second-stage complex-Q cache; never compute during training."""
    if not cache_dir:
        raise ValueError('complex_cache_dir is required when return_complex=True')
    verts_np = verts.detach().cpu().numpy().astype(np.float64)
    faces_np = faces.detach().cpu().numpy().astype(np.int32)
    key = _connection_cache_key(verts_np, faces_np, k)
    path = os.path.join(cache_dir, f'connection_{key}_{int(k)}.npz')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'Missing complex-Q preprocessing cache: {path}. '
            'Run preprocess_complex.py first.')
    cached = np.load(path)
    device = verts.device
    return {
        'conn_evals': torch.from_numpy(cached['conn_evals']).to(device=device, dtype=verts.dtype),
        'conn_evecs': torch.from_numpy(cached['conn_evecs']).to(device=device),
        'spec_grad': torch.from_numpy(cached['spec_grad']).to(device=device),
    }


def compute_connection_cache(verts, faces, mass, k=20, cache_dir=None):
    """Load or compute DUO-FM connection basis and spectral gradient.

    The cache is content-addressed, so a new DataLoader dictionary does not
    trigger eigsh again for the same mesh. Files contain only CPU numpy arrays.
    """
    device = verts.device
    dtype = verts.dtype
    verts_np = verts.detach().cpu().numpy().astype(np.float64)
    faces_np = faces.detach().cpu().numpy().astype(np.int32)
    mass_np = mass.detach().cpu().numpy().reshape(-1)

    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = _connection_cache_key(verts_np, faces_np, k)
        cache_path = os.path.join(cache_dir, f'connection_{key}_{int(k)}.npz')
        if os.path.isfile(cache_path):
            cached = np.load(cache_path)
            return (
                torch.from_numpy(cached['conn_evals']).to(device=device, dtype=dtype),
                torch.from_numpy(cached['conn_evecs']).to(device=device),
                torch.from_numpy(cached['spec_grad']).to(device=device),
            )

    conn_evals, conn_evecs = compute_connection_laplacian_eigenbasis(
        verts, faces, k=k)
    mass_sparse = sp.diags(mass_np)
    spec_grad = compute_spectral_gradient(
        verts_np, faces_np, conn_evecs.detach().cpu().numpy(), mass_sparse)

    if cache_path is not None:
        tmp_path = cache_path + '.tmp.npz'
        np.savez(tmp_path,
                 conn_evals=conn_evals.detach().cpu().numpy(),
                 conn_evecs=conn_evecs.detach().cpu().numpy(),
                 spec_grad=spec_grad)
        os.replace(tmp_path, cache_path)

    return conn_evals, conn_evecs, torch.from_numpy(spec_grad).to(device=device)
def compute_connection_laplacian_eigenbasis(verts, faces, k=20):
    """
    计算 connection Laplacian 的特征基（用于 complex functional maps）
    
    Args:
        verts (torch.Tensor): 顶点 [V, 3]
        faces (torch.Tensor): 面片 [F, 3]
        k (int): 特征数量
        
    Returns:
        conn_evals (torch.Tensor): 特征值 [K]
        conn_evecs (torch.Tensor): 特征向量 [V, K]
    """
    device = verts.device
    dtype = verts.dtype
    
    # 转为 numpy
    verts_np = verts.detach().cpu().numpy().astype(np.float64)
    faces_np = faces.detach().cpu().numpy().astype(np.int32)
    
    # 计算 connection Laplacian
    CL = compute_connection_laplacian_numpy(verts_np, faces_np)
    
    # 计算质量矩阵（用于广义特征问题）
    M = igl.massmatrix(verts_np, faces_np, igl.MASSMATRIX_TYPE_VORONOI)
    
    # 求解广义特征问题: CL @ v = lambda * M @ v
    eps = 1e-8
    CL_eigsh = (CL + eps * sp.identity(CL.shape[0])).tocsc()
    Mmat = M + eps * sp.identity(M.shape[0])
    
    try:
        evals_np, evecs_np = sp.linalg.eigsh(CL_eigsh, k=k, M=Mmat, sigma=0, which="LM")
        # 剪裁负特征值（数值误差）
        evals_np = np.clip(evals_np, a_min=0., a_max=float('inf'))
    except Exception as e:
        # 不能回退到随机基：错误基会被写入缓存并污染后续所有训练
        raise RuntimeError(
            f'Connection Laplacian eigen-decomposition failed for k={k}: {e}') from e
    
    # 转回 torch
    conn_evals = torch.from_numpy(evals_np).to(device=device, dtype=dtype)
    conn_evecs = torch.from_numpy(evecs_np).to(device=device)
    
    return conn_evals, conn_evecs


def compute_grad_vert_op(verts_np, faces_np, edges, op, nex, he_start, he_angles_norm, K):
    """
    计算顶点梯度算子（基于 halfedge 局部坐标系）
    
    基于 DUO-FM mesh.py:393-461 的实现
    构建 [2V, V] 稀疏矩阵，将顶点函数映射到顶点切向量场
    
    Args:
        verts_np (np.ndarray): 顶点 [V, 3]
        faces_np (np.ndarray): 面片 [F, 3]
        edges (np.ndarray): halfedge 数组 [E, 2]
        op (np.ndarray): opposite halfedge 索引 [E]
        nex (np.ndarray): next halfedge 索引 [E]
        he_start (np.ndarray): 每个顶点的起始 halfedge [V]
        he_angles_norm (np.ndarray): 归一化角度 [E]
        K (np.ndarray): Gaussian curvature [V]
        
    Returns:
        gradv (scipy.sparse.csr_matrix): 梯度算子 [2V, V] (complex via [real_x; real_y])
    """
    I = []
    J = []
    V = []
    
    # 第一轮循环：收集每个顶点 1-ring 的局部坐标
    Vjs = []
    rotate = np.zeros(verts_np.shape[0], dtype=int)
    he = np.copy(he_start)
    i = 0
    while np.any(rotate == 0):
        # 计算 halfedge 向量在局部坐标系中的坐标
        lij = np.linalg.norm(verts_np[edges[he][:, 1]] - verts_np[edges[he][:, 0]], axis=1)
        aij = he_angles_norm[he]
        # [x, y] = lij * [cos(aij), cos(pi/2 - aij)]
        vj = lij[:, None] * np.cos(np.stack([aij, np.pi/2 - aij], axis=-1))
        vj[rotate > 0] = 0  # 循环完成的顶点置零
        Vjs.append(vj)
        
        # 顺时针循环
        he = op[he]
        he = nex[he]
        i += 1
        
        rot_mask = (he_start == he) * (rotate == 0)
        rotate[rot_mask] = i
    
    # 堆叠并求伪逆 [V, d_i, 2] -> [V, 2, d_i]
    Vjs = np.stack(Vjs, axis=1)  # [V, max_deg, 2]
    Vjs_inv = np.linalg.pinv(Vjs)  # [V, 2, max_deg]
    
    # 第二轮循环：填充稀疏矩阵系数
    rotate = np.zeros(verts_np.shape[0], dtype=int)
    he = np.copy(he_start)
    i = 0
    while np.any(rotate == 0):
        jdv = edges[he][:, 1]
        idv = edges[he][:, 0]
        
        # 梯度算子的 x 和 y 分量
        I += [2*idv, 2*idv, 2*idv+1, 2*idv+1]
        J += [idv, jdv, idv, jdv]
        V += [-Vjs_inv[:, 0, i], Vjs_inv[:, 0, i], 
              -Vjs_inv[:, 1, i], Vjs_inv[:, 1, i]]
        
        # 顺时针循环
        he = op[he]
        he = nex[he]
        i += 1
        
        rot_mask = (he_start == he) * (rotate == 0)
        rotate[rot_mask] = i
    
    # 构建稀疏矩阵
    I = np.concatenate(I)
    J = np.concatenate(J)
    V = np.concatenate(V)
    
    gradv = sp.csr_matrix((V, (I, J)), shape=(2 * verts_np.shape[0], verts_np.shape[0]))
    
    return gradv


def compute_spectral_gradient(verts_np, faces_np, ceig, m):
    """
    计算 spectral gradient: Ψ^† @ (gradX + i*gradY)
    
    这是 DUO-FM 中的关键：预计算投影到 connection Laplacian 特征基的梯度
    
    Args:
        verts_np (np.ndarray): 顶点 [V, 3]
        faces_np (np.ndarray): 面片 [F, 3]
        ceig (np.ndarray): connection Laplacian 特征向量 [V, K_c]
        m (scipy.sparse.matrix): 质量矩阵 [V, V]
        
    Returns:
        spec_grad (torch.Tensor): spectral gradient [K_c, V] (complex)
    """
    # 计算 Hermitian 转置的伪逆
    ceig_trans = np.conj(m @ ceig).T  # [K_c, V] (complex)
    
    # 构建 halfedge 结构（不依赖 libigl 返回值版本）
    edges, op, nex, he_start = _build_halfedges(faces_np)
    
    # 计算角度和曲率（复用 connection Laplacian 逻辑）
    angles = igl.internal_angles(verts_np, faces_np)
    he_angles = angles.reshape(-1).copy()
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
    
    # Convert stacked real/imaginary gradient rows [2V, V] to a complex
    # vertex operator [V, V], matching DUO-FM's tangent-field convention.
    gradv_complex = gradv[0::2, :].toarray() + 1j * gradv[1::2, :].toarray()
    spec_grad = ceig_trans @ gradv_complex  # [K_c, V] complex
    
    return spec_grad
