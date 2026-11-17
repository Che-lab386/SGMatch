"""
Jacobian 约束损失：用于部分形状匹配的有界非共形约束。

约束局部 Jacobian 满足：
1. det(J) > 0 （保向，无翻转）
2. σ₁/σ₂ ≤ K_max （有界非共形）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JacobianLoss(nn.Module):
    """
    局部 Jacobian 约束损失。
    
    对 overlap 区域内的每个点，从其 1-ring 邻域估计局部切平面映射 J_p，
    并约束其行列式和条件数。
    """
    
    def __init__(self, k_max=3.0, det_eps=0.01, use_log_distortion=True):
        """
        Args:
            k_max: 最大条件数（σ_max / σ_min 的上界）
            det_eps: 行列式下界（小于此值视为翻转）
            use_log_distortion: 是否对条件数取对数（更稳定）
        """
        super().__init__()
        self.k_max = k_max
        self.det_eps = det_eps
        self.use_log_distortion = use_log_distortion
        self.log_k_max = torch.log(torch.tensor(k_max))
    
    def estimate_local_jacobian(self, verts_x, verts_y, p2p_map, faces_x, 
                               overlap_mask=None, robust=True):
        """
        从顶点坐标和对应估计局部 Jacobian。
        
        Args:
            verts_x: 源顶点 [B, Nx, 3]
            verts_y: 目标顶点 [B, Ny, 3]
            p2p_map: 点对应 [B, Nx]（硬匹配索引）或 [B, Nx, Ny]（软对应概率）
            faces_x: 源面片 [B, Fx, 3] 或 [Fx, 3]
            overlap_mask: overlap 掩码 [B, Nx]（可选）
            robust: 是否使用鲁棒拟合（加正则）
        
        Returns:
            J: 局部 Jacobian [B, Nx, 2, 2]
            valid_mask: 有效估计掩码 [B, Nx]（邻域退化的点标记为 False）
        """
        B, Nx, _ = verts_x.shape
        device = verts_x.device
        
        # 处理面片维度
        if faces_x.dim() == 2:
            faces_x = faces_x.unsqueeze(0).expand(B, -1, -1)
        
        # 构建 1-ring 邻接
        neigh_indices, neigh_counts = self._build_one_ring(faces_x, Nx)  # [B, Nx, max_deg]
        
        # 获取对应目标点
        if p2p_map.dim() == 2:
            # 硬匹配
            y_corr = torch.gather(verts_y, 1, p2p_map.unsqueeze(-1).expand(-1, -1, 3))
        else:
            # 软匹配（加权平均）
            y_corr = torch.bmm(p2p_map, verts_y)  # [B, Nx, 3]
        
        J_list = []
        valid_list = []
        
        for b in range(B):
            J_b, valid_b = self._estimate_jacobian_batch(
                verts_x[b], verts_y[b], y_corr[b], 
                neigh_indices[b], neigh_counts[b],
                robust=robust
            )
            J_list.append(J_b)
            valid_list.append(valid_b)
        
        J = torch.stack(J_list, dim=0)  # [B, Nx, 2, 2]
        valid_mask = torch.stack(valid_list, dim=0)  # [B, Nx]
        
        # 应用 overlap mask
        if overlap_mask is not None:
            valid_mask = valid_mask & overlap_mask
        
        return J, valid_mask
    
    def _build_one_ring(self, faces, n_verts):
        """构建 1-ring 邻接表。"""
        B, F, _ = faces.shape
        device = faces.device
        max_deg = 20  # 假设最大度数
        
        neigh_indices = torch.full((B, n_verts, max_deg), -1, dtype=torch.long, device=device)
        neigh_counts = torch.zeros((B, n_verts), dtype=torch.long, device=device)
        
        for b in range(B):
            for i in range(3):
                src = faces[b, :, i]
                dst = faces[b, :, (i+1) % 3]
                for s, d in zip(src, dst):
                    cnt = neigh_counts[b, s]
                    if cnt < max_deg:
                        neigh_indices[b, s, cnt] = d
                        neigh_counts[b, s] += 1
        
        return neigh_indices, neigh_counts
    
    def _estimate_jacobian_batch(self, verts_x, verts_y, y_corr, neigh_idx, neigh_cnt, robust=True):
        """
        对单个 batch 估计所有点的 Jacobian。
        
        Args:
            verts_x: [Nx, 3]
            verts_y: [Ny, 3]
            y_corr: [Nx, 3] 每个源点的对应目标点
            neigh_idx: [Nx, max_deg]
            neigh_cnt: [Nx]
        """
        Nx = verts_x.shape[0]
        device = verts_x.device
        
        J = torch.zeros(Nx, 2, 2, device=device)
        valid = torch.zeros(Nx, dtype=torch.bool, device=device)
        
        for i in range(Nx):
            cnt = neigh_cnt[i].item()
            if cnt < 2:
                continue  # 邻域太少，跳过
            
            neigh = neigh_idx[i, :cnt]
            neigh = neigh[neigh >= 0]  # 过滤无效索引
            
            if len(neigh) < 2:
                continue
            
            # 源侧邻域向量
            x_i = verts_x[i]
            x_neigh = verts_x[neigh]
            dx = x_neigh - x_i  # [K, 3]
            
            # 目标侧邻域向量
            y_i = y_corr[i]
            y_neigh_corr = y_corr[neigh]  # 使用邻居的对应点
            dy = y_neigh_corr - y_i  # [K, 3]
            
            # 投影到局部切平面
            normal_x = self._estimate_normal(dx)
            normal_y = self._estimate_normal(dy)
            
            if normal_x is None or normal_y is None:
                continue
            
            dx_tan = self._project_to_tangent(dx, normal_x)  # [K, 2]
            dy_tan = self._project_to_tangent(dy, normal_y)  # [K, 2]
            
            # 最小二乘拟合 J: dy_tan ≈ J @ dx_tan
            if robust:
                eps = 1e-4
                A = dx_tan.T @ dx_tan + eps * torch.eye(2, device=device)
                b = dx_tan.T @ dy_tan
                J[i] = torch.linalg.solve(A, b).T  # [2, 2]
            else:
                J[i] = torch.linalg.lstsq(dx_tan, dy_tan).solution.T
            
            valid[i] = True
        
        return J, valid
    
    def _estimate_normal(self, vecs):
        """从邻域向量估计法向量（PCA 最小方差方向）。"""
        if len(vecs) < 2:
            return None
        try:
            U, S, Vh = torch.linalg.svd(vecs, full_matrices=False)
            normal = Vh[-1]  # 最小奇异值对应方向
            return normal
        except:
            return None
    
    def _project_to_tangent(self, vecs, normal):
        """投影到切平面并生成 2D 坐标。"""
        # 选择两个正交切向量
        t1 = torch.tensor([1.0, 0.0, 0.0], device=vecs.device)
        if torch.abs(torch.dot(normal, t1)) > 0.9:
            t1 = torch.tensor([0.0, 1.0, 0.0], device=vecs.device)
        
        t1 = t1 - torch.dot(t1, normal) * normal
        t1 = t1 / (torch.norm(t1) + 1e-8)
        t2 = torch.cross(normal, t1)
        t2 = t2 / (torch.norm(t2) + 1e-8)
        
        coords = torch.stack([
            torch.sum(vecs * t1, dim=-1),
            torch.sum(vecs * t2, dim=-1)
        ], dim=-1)
        return coords
    
    def forward(self, verts_x, verts_y, p2p_map, faces_x, overlap_mask=None):
        """
        计算 Jacobian 约束损失。
        
        Args:
            verts_x: 源顶点 [B, Nx, 3]
            verts_y: 目标顶点 [B, Ny, 3]
            p2p_map: 点对应 [B, Nx] 或 [B, Nx, Ny]
            faces_x: 源面片 [B, Fx, 3] 或 [Fx, 3]
            overlap_mask: overlap 掩码 [B, Nx]（可选）
        
        Returns:
            loss_dict: {
                'l_det': 行列式约束损失,
                'l_distortion': 条件数约束损失,
                'flip_rate': 翻转率（统计）,
                'mean_distortion': 平均条件数（统计）
            }
        """
        J, valid_mask = self.estimate_local_jacobian(
            verts_x, verts_y, p2p_map, faces_x, overlap_mask
        )
        
        if not valid_mask.any():
            # 无有效点，返回零损失
            device = verts_x.device
            return {
                'l_det': torch.tensor(0.0, device=device),
                'l_distortion': torch.tensor(0.0, device=device),
                'flip_rate': torch.tensor(0.0, device=device),
                'mean_distortion': torch.tensor(0.0, device=device),
            }
        
        # 计算行列式
        det = J[:, :, 0, 0] * J[:, :, 1, 1] - J[:, :, 0, 1] * J[:, :, 1, 0]  # [B, Nx]
        
        # 行列式约束：det > det_eps
        l_det = torch.where(
            valid_mask,
            F.relu(self.det_eps - det) ** 2,
            torch.zeros_like(det)
        )
        l_det = l_det.sum() / (valid_mask.sum() + 1e-8)
        
        # 计算奇异值
        try:
            U, S, Vh = torch.linalg.svd(J[valid_mask])  # S: [N_valid, 2]
            sigma1 = S[:, 0]
            sigma2 = S[:, 1].clamp(min=1e-6)
            
            # 条件数约束
            if self.use_log_distortion:
                log_distortion = torch.log(sigma1 / sigma2 + 1e-8)
                l_distortion = F.relu(log_distortion - self.log_k_max.to(log_distortion.device)) ** 2
            else:
                distortion = sigma1 / sigma2
                l_distortion = F.relu(distortion - self.k_max) ** 2
            
            l_distortion = l_distortion.mean()
            
            # 统计指标
            flip_rate = (det[valid_mask] < 0).float().mean()
            mean_distortion = (sigma1 / sigma2).mean()
        
        except:
            # SVD 失败，返回零
            l_distortion = torch.tensor(0.0, device=verts_x.device)
            flip_rate = torch.tensor(0.0, device=verts_x.device)
            mean_distortion = torch.tensor(0.0, device=verts_x.device)
        
        return {
            'l_det': l_det,
            'l_distortion': l_distortion,
            'flip_rate': flip_rate,
            'mean_distortion': mean_distortion,
        }
