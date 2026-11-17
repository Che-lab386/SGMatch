"""
Complex Functional Map 工具函数。

支持将点映射转换为 Q（切向量场传输矩阵）。
当前实现仅支持 Q 正交正则，不包括 Π–Q 对齐。
"""

import torch


def _mm(operator, dense):
    """矩阵乘法，支持稀疏矩阵。"""
    if operator.layout == torch.strided:
        return operator @ dense
    return torch.sparse.mm(operator.coalesce(), dense)


def procrustes_q_from_functional_map(c_xy, df_x, df_y):
    """
    从标量泛函映射通过 Procrustes 求解复泛函映射。
    
    对于每个 LBO 基函数 φ_f，计算方向传输矩阵 Q，使得：
        C D_f^X ≈ D_{Cf}^Y Q
    
    这对应于 SVD 求解 Q = U V^H（Procrustes）。
    
    Args:
        c_xy: 标量泛函映射 [B, K_lb, K_lb]
        df_x: 源侧微分基 [B, K_lb, K_lb, 2*K_conn]  
        df_y: 目标侧微分基 [B, K_lb, K_lb, 2*K_conn]
    
    Returns:
        q_xy: 复泛函映射 [B, K_conn, K_conn] (complex)
    """
    # 左侧：C D_f^X，形状 [B, K_lb, K_lb, 2*K_conn]
    cd = torch.einsum('bij,bfjc->bfic', c_xy, df_x).flatten(1, 2)  # [B, K_lb^2, 2*K_conn]
    
    # 右侧：D_{Cf}^Y，形状 [B, K_lb, K_lb, 2*K_conn]
    dc = torch.einsum('bri,brjc->bijc', c_xy, df_y).flatten(1, 2)  # [B, K_lb^2, 2*K_conn]
    
    # 转为复数：[B, K_lb^2, K_conn]
    cd = torch.complex(cd[..., 0::2], cd[..., 1::2])
    dc = torch.complex(dc[..., 0::2], dc[..., 1::2])
    
    # Procrustes: min ||Q - dc^H @ cd||_F  =>  Q = U V^H from SVD(dc^H @ cd)
    u, _, vh = torch.linalg.svd(dc.mH @ cd, full_matrices=False)
    q_xy = u @ vh
    
    return q_xy


def point_map_to_q_hat(p_yx, evecs_x, evecs_trans_y, df_x, df_y):
    """
    从点对应矩阵 Π_YX 恢复 Q_XY。
    
    步骤：
    1. 通过点对应采样构造 C_XY：C_XY = Φ_Y^† Π_YX Φ_X
    2. 通过 Procrustes 从 C_XY 和微分基求解 Q_XY
    
    Args:
        p_yx: 点对应矩阵 Π_YX，形状 [V_y, V_x]（Y 点映射到 X 的概率）
        evecs_x: 源侧 LBO 特征向量 [V_x, K_lb]
        evecs_trans_y: 目标侧 LBO 特征向量的伪逆投影 [K_lb, V_y]
        df_x: 源侧微分基 [K_lb, K_lb, 2*K_conn]
        df_y: 目标侧微分基 [K_lb, K_lb, 2*K_conn]
    
    Returns:
        q_xy: 复泛函映射 [K_conn, K_conn] (complex)
    """
    # 添加 batch 维度（如果需要）
    unbatched = p_yx.ndim == 2
    if unbatched:
        p_yx = p_yx.unsqueeze(0)
        evecs_x = evecs_x.unsqueeze(0)
        evecs_trans_y = evecs_trans_y.unsqueeze(0)
        df_x = df_x.unsqueeze(0)
        df_y = df_y.unsqueeze(0)
    
    # C_XY = Φ_Y^† Π_YX Φ_X
    c_xy = evecs_trans_y @ p_yx @ evecs_x  # [B, K_lb, K_lb]
    
    # 通过 Procrustes 求解 Q_XY
    q_xy = procrustes_q_from_functional_map(c_xy, df_x, df_y)
    
    return q_xy.squeeze(0) if unbatched else q_xy
