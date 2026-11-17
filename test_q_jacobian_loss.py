"""
验证 Q 酉性损失与 Jacobian 约束损失的协调。

用途：
1. 确认两个损失的计算正确性
2. 对比单独使用 Q、单独使用 Jacobian、联合使用三种情况
3. 检验 overlap 区域内的 Jacobian 指标（翻转率、畸变）
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from losses.jacobian_loss import JacobianLoss


def synthetic_test_case():
    """生成一个简单的合成测试用例。"""
    # 源形状：简单平面网格
    verts_x = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
    ], dtype=torch.float32).unsqueeze(0)  # [1, 4, 3]
    
    faces_x = torch.tensor([
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=torch.int32).unsqueeze(0)  # [1, 2, 3]
    
    # 目标形状：轻微缩放 + 剪切
    # 纯共形：只缩放
    # verts_y = verts_x * 1.2
    
    # 非共形：x 方向拉伸、y 方向压缩
    verts_y = verts_x.clone()
    verts_y[..., 0] *= 2.0    # x 拉伸 2 倍
    verts_y[..., 1] *= 0.5    # y 压缩 0.5 倍
    verts_y = verts_y.float()
    
    # 完美对应（恒等映射）
    p2p_map = torch.arange(4, dtype=torch.int64).unsqueeze(0)  # [1, 4]
    
    return verts_x, verts_y, p2p_map, faces_x


def test_jacobian_loss():
    """测试 Jacobian 损失计算。"""
    print("=" * 60)
    print("Test 1: Jacobian Loss Computation")
    print("=" * 60)
    
    verts_x, verts_y, p2p_map, faces_x = synthetic_test_case()
    
    jacobian_loss = JacobianLoss(k_max=3.0, det_eps=0.01, use_log_distortion=True)
    
    # 测试：完美非共形（条件数 = 4）
    print(f"\nInput: verts_x shape={verts_x.shape}, verts_y shape={verts_y.shape}")
    print(f"Expected condition number: 2.0 / 0.5 = 4.0")
    print(f"K_max constraint: 3.0")
    print(f"Expected: condition number 4.0 > 3.0, so l_distortion > 0")
    
    result = jacobian_loss(verts_x, verts_y, p2p_map, faces_x)
    
    print(f"\nResults:")
    print(f"  l_det: {result['l_det'].item():.6f} (should ≈ 0, no flips)")
    print(f"  l_distortion: {result['l_distortion'].item():.6f} (should > 0, violates K_max)")
    print(f"  flip_rate: {result['flip_rate'].item():.4f} (should ≈ 0)")
    print(f"  mean_distortion: {result['mean_distortion'].item():.4f} (should ≈ 4.0)")
    
    return result


def test_q_ortho_loss():
    """测试 Q 酉性损失计算。"""
    print("\n" + "=" * 60)
    print("Test 2: Q Orthogonal Loss")
    print("=" * 60)
    
    # 生成随机特征
    K_conn = 30
    batch_size = 1
    feat_dim = 256
    
    # 随机特征（模拟 spec_grad @ f_geo）
    feat_x = torch.randn(batch_size, K_conn, feat_dim, dtype=torch.complex128)
    feat_y = torch.randn(batch_size, K_conn, feat_dim, dtype=torch.complex128)
    
    # 计算 Q
    Qxy = feat_y @ torch.linalg.pinv(feat_x)  # [B, K_conn, K_conn]
    
    # Q 酉性损失
    I = torch.eye(K_conn, dtype=torch.complex128).unsqueeze(0)
    Q_ortho = Qxy.mH @ Qxy
    l_Q_ortho = torch.norm(Q_ortho - I, p='fro', dim=(-2, -1)) ** 2
    
    print(f"\nInput: feat_x shape={feat_x.shape}, feat_y shape={feat_y.shape}")
    print(f"Q shape: {Qxy.shape}")
    print(f"Q^H Q - I norm: {torch.norm(Q_ortho - I, p='fro').item():.6f}")
    print(f"\nResults:")
    print(f"  l_Q_ortho: {l_Q_ortho.item():.6f}")
    print(f"  Status: {'UNITARY' if l_Q_ortho.item() < 1e-4 else 'NOT UNITARY'}")
    
    return l_Q_ortho


def test_combined_loss():
    """测试 Q + Jacobian 联合约束。"""
    print("\n" + "=" * 60)
    print("Test 3: Combined Q + Jacobian Constraints")
    print("=" * 60)
    
    verts_x, verts_y, p2p_map, faces_x = synthetic_test_case()
    K_conn = 30
    
    # Jacobian 约束
    jacobian_loss = JacobianLoss(k_max=2.0, det_eps=0.01)  # 更严格的界
    jac_result = jacobian_loss(verts_x, verts_y, p2p_map, faces_x)
    
    # Q 约束（随机特征）
    feat_x = torch.randn(1, K_conn, 256, dtype=torch.complex128)
    feat_y = torch.randn(1, K_conn, 256, dtype=torch.complex128)
    Qxy = feat_y @ torch.linalg.pinv(feat_x)
    I = torch.eye(K_conn, dtype=torch.complex128).unsqueeze(0)
    Q_ortho = Qxy.mH @ Qxy
    l_Q_ortho = torch.norm(Q_ortho - I, p='fro', dim=(-2, -1)) ** 2
    
    # 联合损失
    w_q = 0.02
    w_j_det = 0.01
    w_j_dist = 0.01
    
    total_loss = (w_q * l_Q_ortho.mean() + 
                  w_j_det * jac_result['l_det'] + 
                  w_j_dist * jac_result['l_distortion'])
    
    print(f"\nLoss weights:")
    print(f"  w_q_orth: {w_q}")
    print(f"  w_jacobian_det: {w_j_det}")
    print(f"  w_jacobian_distortion: {w_j_dist}")
    
    print(f"\nIndividual losses:")
    print(f"  l_Q_ortho: {l_Q_ortho.mean().item():.6f}")
    print(f"  l_J_det: {jac_result['l_det'].item():.6f}")
    print(f"  l_J_distortion: {jac_result['l_distortion'].item():.6f}")
    
    print(f"\nWeighted losses:")
    print(f"  w_q × l_Q_ortho: {(w_q * l_Q_ortho.mean()).item():.6f}")
    print(f"  w_j_det × l_J_det: {(w_j_det * jac_result['l_det']).item():.6f}")
    print(f"  w_j_dist × l_J_distortion: {(w_j_dist * jac_result['l_distortion']).item():.6f}")
    
    print(f"\nTotal loss: {total_loss.item():.6f}")
    print(f"\nInterpretation:")
    print(f"  Q contributes {(w_q * l_Q_ortho.mean() / total_loss * 100).item():.1f}% of total loss")
    print(f"  Jacobian det contributes {(w_j_det * jac_result['l_det'] / total_loss * 100).item():.1f}%")
    print(f"  Jacobian distortion contributes {(w_j_dist * jac_result['l_distortion'] / total_loss * 100).item():.1f}%")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Testing Q Orthogonality + Jacobian Constraint Losses")
    print("=" * 60)
    
    test_jacobian_loss()
    test_q_ortho_loss()
    test_combined_loss()
    
    print("\n" + "=" * 60)
    print("All tests completed")
    print("=" * 60 + "\n")
