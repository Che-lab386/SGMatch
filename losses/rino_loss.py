import torch
import torch.nn as nn

from utils.registry import LOSS_REGISTRY


def _frobenius(a):
    return torch.sum(torch.abs(a) ** 2, dim=(-2, -1)).mean()


@LOSS_REGISTRY.register()
class FourBranchLoss(nn.Module):
    """RINO-style loss for SGMatch's P/C/Q/CFM architecture.

    P-to-C and self-map terms are computed here. P-to-Q is included only when
    the caller supplies Qhat, the CFMap induced by P via the SVD conversion.
    C and Q are never compared directly because they live in different bases.
    """

    def __init__(self, w_c_orth=1.0, w_q_orth=1.0, w_c_bij=1.0,
                 w_pc=1.0, w_pq=0.0, w_contr=0.0, contrastive_scale=6.0):
        super().__init__()
        self.w_c_orth = w_c_orth
        self.w_q_orth = w_q_orth
        self.w_c_bij = w_c_bij
        self.w_pc = w_pc
        self.w_pq = w_pq
        self.w_contr = w_contr
        self.contrastive_scale = contrastive_scale

    @staticmethod
    def _eye_like(matrix):
        return torch.eye(matrix.shape[-1], device=matrix.device,
                         dtype=matrix.dtype).expand(matrix.shape[0], -1, -1)

    @staticmethod
    def _real_orthogonal_loss(matrix):
        eye = torch.eye(matrix.shape[-1], device=matrix.device,
                        dtype=matrix.dtype).expand(matrix.shape[0], -1, -1)
        return _frobenius(matrix.transpose(-2, -1) @ matrix - eye)

    @staticmethod
    def _complex_orthogonal_loss(matrix):
        eye = torch.eye(matrix.shape[-1], device=matrix.device,
                        dtype=matrix.dtype).expand(matrix.shape[0], -1, -1)
        return _frobenius(matrix.transpose(-2, -1).conj() @ matrix - eye)

    @staticmethod
    def _orthogonal_loss(matrix):
        # Retained for callers that use the generic helper.
        return FourBranchLoss._real_orthogonal_loss(matrix)

    def forward(self, Cxy, Cyx, Qxy=None, Qyx=None, Pxy=None, Pyx=None,
                evecs_x=None, evecs_y=None, evecs_trans_x=None,
                evecs_trans_y=None, features_x=None, features_y=None,
                Qhat_xy=None, Qhat_yx=None):
        losses = {}

        if self.w_c_orth > 0:
            losses['l_c_orth'] = self.w_c_orth * (
                self._real_orthogonal_loss(Cxy) + self._real_orthogonal_loss(Cyx))

        if self.w_q_orth > 0 and Qxy is not None:
            q_loss = self._complex_orthogonal_loss(Qxy)
            if Qyx is not None:
                q_loss = q_loss + self._complex_orthogonal_loss(Qyx)
            losses['l_Q_ortho'] = self.w_q_orth * q_loss

        if self.w_c_bij > 0 and Cyx is not None:
            eye = self._eye_like(Cxy)
            losses['l_c_bij'] = self.w_c_bij * (
                _frobenius(Cxy @ Cyx - eye) +
                _frobenius(Cyx @ Cxy - self._eye_like(Cyx)))

        if (self.w_pc > 0 and Pxy is not None and Pyx is not None and
                evecs_x is not None and evecs_y is not None and
                evecs_trans_x is not None and evecs_trans_y is not None):
            c_hat_xy = evecs_trans_y @ Pyx @ evecs_x
            c_hat_yx = evecs_trans_x @ Pxy @ evecs_y
            losses['l_pc'] = self.w_pc * (
                _frobenius(Cxy - c_hat_xy) +
                _frobenius(Cyx - c_hat_yx))

        if self.w_pq > 0 and Qxy is not None and Qhat_xy is not None:
            pq_loss = _frobenius(Qxy - Qhat_xy)
            if Qyx is not None and Qhat_yx is not None:
                pq_loss = pq_loss + _frobenius(Qyx - Qhat_yx)
            losses['l_pq'] = self.w_pq * pq_loss

        if (self.w_contr > 0 and features_x is not None and
                features_y is not None and evecs_x is not None and
                evecs_y is not None and evecs_trans_x is not None and
                evecs_trans_y is not None):
            px = torch.softmax(features_x @ features_x.transpose(-2, -1), dim=-1)
            py = torch.softmax(features_y @ features_y.transpose(-2, -1), dim=-1)
            c_xx = evecs_trans_x @ px @ evecs_x
            c_yy = evecs_trans_y @ py @ evecs_y
            losses['l_contr'] = self.w_contr * self.contrastive_scale * (
                _frobenius(c_xx - self._eye_like(c_xx)) +
                _frobenius(c_yy - self._eye_like(c_yy)))

        return losses


@LOSS_REGISTRY.register()
class ComplexOrthogonalLoss(nn.Module):
    """Compatibility wrapper for the original SGMatch Q-only loss."""

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, Q):
        adjoint = Q.transpose(-2, -1).conj()
        eye = torch.eye(Q.shape[-1], device=Q.device, dtype=Q.dtype)
        eye = eye.expand(Q.shape[0], -1, -1)
        return self.loss_weight * _frobenius(Q @ adjoint - eye)
