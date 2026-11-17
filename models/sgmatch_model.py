import torch
import torch.nn.functional as F

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, trim_basis, hybrid_fmap2pointmap
from utils.geometry_util import get_all_operators
from utils.geometry_util import to_basis, from_basis
from utils.neigh_util import get_neigh
from losses.jacobian_loss import JacobianLoss

# Optional Q branch imports (only when enabled)
_Q_AVAILABLE = False
try:
    from utils.connection_laplacian import load_connection_cache
    from utils.complex_functional_map import point_map_to_q_hat
    _Q_AVAILABLE = True
except ImportError:
    pass

def cache_operators(data, cache_dir=None, complex_cache_dir=None,
                    use_complex_fmap=False, n_conn=30):
    """
    加载几何算子（标准 Laplacian + 可选 connection Laplacian）。
    
    use_complex_fmap=True 时加载 connection 算子并填入 operators 字典；
    失败时报错（不静默）。
    """
    data_x, data_y = data['first'], data['second']

    for side, d in [('first', data_x), ('second', data_y)]:
        if 'operators' not in d:
            cd = cache_dir or d.get('cache_dir', None)
            _, mass, L, evals, evecs, gradX, gradY = get_all_operators(
                d['verts'].cpu(), d['faces'].cpu(), k=128, cache_dir=cd)
            d['operators'] = {
                'mass': mass, 'L': L, 'evals': evals, 'evecs': evecs,
                'gradX': gradX, 'gradY': gradY,
            }

        if use_complex_fmap and 'spec_grad' not in d['operators']:
            if not _Q_AVAILABLE:
                raise RuntimeError(
                    'use_complex_fmap=True but utils.connection_laplacian '
                    'could not be imported. Check dependencies (igl, etc.).')
            # 若数据集已经加载了 connection 字段，直接搬入 operators
            if 'spec_grad' in d:
                d['operators'].update({
                    'conn_evals': d['conn_evals'],
                    'conn_evecs': d['conn_evecs'],
                    'spec_grad':  d['spec_grad'],
                })
            else:
                verts = d['verts'].squeeze(0) if d['verts'].dim() == 3 else d['verts']
                faces = d['faces'].squeeze(0) if d['faces'].dim() == 3 else d['faces']
                cached = load_connection_cache(
                    verts.cpu(), faces.cpu(), k=n_conn,
                    cache_dir=complex_cache_dir)
                d['operators'].update(cached)

@MODEL_REGISTRY.register()
class SGMatch_Model(BaseModel):
    def __init__(self, opt):
        self.with_refine = opt.get('refine', -1)
        self.partial = opt.get('partial', False)
        self.non_isometric = opt.get('non-isometric', False)
        self.heat_diffusion_tau = opt.get('heat_diffusion_tau', 0.0001)
        self.sm_feat = opt.get('sm_feat', False)
        self.neigh_k = opt.get('neigh_k', 32)
        # Q branch config
        self.use_complex_fmap = opt.get('use_complex_fmap', False)
        self.n_conn = opt.get('n_conn', 30)
        self.w_q_orth = opt.get('w_q_orth', 0.0)
        self.w_pq = opt.get('w_pq', 0.0)  # Π–Q alignment (keep 0 for scheme B)
        self.complex_cache_dir = opt.get('complex_cache_dir', None)
        self.q_verbose = opt.get('q_verbose', True)
        self._q_failed_once = False  # first failure raises; later warns
        # Jacobian (local distortion) loss config
        self.use_jacobian_loss = opt.get('use_jacobian_loss', False)
        self.w_jacobian_det = opt.get('w_jacobian_det', 0.0)
        self.w_jacobian_distortion = opt.get('w_jacobian_distortion', 0.0)
        self.jacobian_loss = None
        if self.use_jacobian_loss and (self.w_jacobian_det > 0 or self.w_jacobian_distortion > 0):
            self.jacobian_loss = JacobianLoss(k_max=opt.get('jacobian_k_max', 3.0))
        if self.with_refine > 0:
            opt['is_train'] = True
        super(SGMatch_Model, self).__init__(opt)

    def feed_data(self, data):
        cache_dir = self.opt['networks']['feature_extractor'].get('cache_dir', None)
        # Load Q connection operators if enabled
        cache_operators(
            data, cache_dir=cache_dir,
            complex_cache_dir=self.complex_cache_dir,
            use_complex_fmap=self.use_complex_fmap,
            n_conn=self.n_conn
        )
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # feature extractor for mesh
        f_geo_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'], data=data_x)  # [B, Nx, C]
        f_geo_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'], data=data_y)  # [B, Ny, C]

        # trim basis
        n_lb = self.opt.get('n_lb', 140)
        n_elas = self.opt.get('n_elas', 60)
        data_x = trim_basis(data_x, n_lb, n_elas)
        data_y = trim_basis(data_y, n_lb, n_elas)

        # feature fusion
        evals_x = data_x['evals']
        evals_y = data_y['evals']
        evecs_x = data_x['evecs']
        evecs_y = data_y['evecs']

        neigh_x = get_neigh(data_x['faces'], data_x['verts'].shape[1], K=self.neigh_k, add_self=True)  # [B, Nx, K]
        neigh_y = get_neigh(data_y['faces'], data_y['verts'].shape[1], K=self.neigh_k, add_self=True)  # [B, Ny, K]

        # Cross-attention fusion (geo features query semantic DINO)
        hx = self.networks["cross_attention"](f_geo_x, data_x["dino"], neigh_x, data_x['verts'])
        hy = self.networks["cross_attention"](f_geo_y, data_y["dino"], neigh_y, data_y['verts'])

        # Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)
        Pxy, Pyx = self.compute_permutation_matrix(hx, hy, bidirectional=True)

        # ------------------- LB Loss ------------------- #
        if n_lb > 0:
            # get spectral operators
            # evals_x = data_x['evals']
            # evals_y = data_y['evals']
            # evecs_x = data_x['evecs']
            # evecs_y = data_y['evecs']
            evecs_trans_x = data_x['evecs_trans']  # [B, K, Nx]
            evecs_trans_y = data_y['evecs_trans']  # [B, K, Ny]

            Cxy, Cyx = self.networks['fmap_net'](hx, hy, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

            self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, evals_x, evals_y)

            # compute C
            Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))

            self.loss_metrics['l_align'] = self.losses['align_loss'](Cxy, Cxy_est)
            if not self.partial:
                Cyx_est = torch.bmm(evecs_trans_x, torch.bmm(Pxy, evecs_y))
                self.loss_metrics['l_align'] += self.losses['align_loss'](Cyx, Cyx_est)

        # ------------------- Elas Loss ------------------- #
        if n_elas > 0:
            # get elas spectral operators
            elas_evals_x = torch.abs(data_x['elas_evals'])
            elas_evals_y = torch.abs(data_y['elas_evals'])
            elas_evecs_x = data_x['elas_evecs']
            elas_evecs_y = data_y['elas_evecs']
            elas_evecs_trans_x = data_x['elas_evecs_trans']  # [B, K, Nx]
            elas_evecs_trans_y = data_y['elas_evecs_trans']  # [B, K, Ny]
            elas_mass_x = data_x['elas_mass']
            elas_mass_y = data_y['elas_mass']
            elas_Mk_x = data_x['elas_Mk']
            elas_Mk_y = data_y['elas_Mk']
            elas_invsqrtMk_x = data_x['elas_invsqrtMk']
            elas_invsqrtMk_y = data_y['elas_invsqrtMk']
            elas_sqrtMk_x = data_x['elas_sqrtMk']
            elas_sqrtMk_y = data_y['elas_sqrtMk']
            
            elas_Cxy, elas_Cyx = self.networks['expanded_fmap_net'](hx, hy, elas_evals_x, elas_evals_y, elas_evecs_trans_x, elas_evecs_trans_y, elas_sqrtMk_x, elas_sqrtMk_y)

            hs_surfmnet_loss = self.losses['hs_surfmnet_loss'](elas_Cxy, elas_Cyx, elas_evals_x, elas_evals_y, elas_Mk_x, elas_Mk_y)
            self.loss_metrics['l_elas_bij'] = hs_surfmnet_loss['l_bij']
            self.loss_metrics['l_elas_orth'] = hs_surfmnet_loss['l_orth']

            # compute C
            elas_Cxy_est = torch.bmm(elas_evecs_trans_y, torch.bmm(Pyx, elas_evecs_x))

            self.loss_metrics['l_elas_align'] = self.losses['hs_align_loss'](elas_Cxy, elas_Cxy_est, elas_sqrtMk_x, elas_sqrtMk_y)
            if not self.partial:
                elas_Cyx_est = torch.bmm(elas_evecs_trans_x, torch.bmm(Pxy, elas_evecs_y))
                self.loss_metrics['l_elas_align'] += self.losses['hs_align_loss'](elas_Cyx, elas_Cyx_est, elas_sqrtMk_y, elas_sqrtMk_x)
        
        if 'dirichlet_loss' in self.losses:
            Lx, Ly = data_x['operators']['L'], data_y['operators']['L']
            verts_x, verts_y = data_x['verts'], data_y['verts']
            self.loss_metrics['l_d'] = self.losses['dirichlet_loss'](torch.bmm(Pxy, verts_y), Lx) + \
                                       self.losses['dirichlet_loss'](torch.bmm(Pyx, verts_x), Ly)

        # ------------------- Jacobian (local distortion) Loss ------------------- #
        if self.jacobian_loss is not None:
            jac = self.jacobian_loss(
                data_x['verts'], data_y['verts'], Pxy, data_x['faces'])
            self.loss_metrics['l_J_det'] = jac['l_det']
            self.loss_metrics['l_J_distortion'] = jac['l_distortion']
            # monitoring only, excluded from the total loss
            self.loss_metrics['J_flip_rate'] = jac['flip_rate']
            self.loss_metrics['J_mean_distortion'] = jac['mean_distortion']

        # ------------------- Q (Complex Functional Map) Section ------------------- #
        # Scheme B: Q uses geometric features (f_geo) before CrossAttention fusion
        if self.use_complex_fmap and self.w_q_orth > 0:
            try:
                if not _Q_AVAILABLE:
                    raise RuntimeError(
                        'use_complex_fmap=True but connection_laplacian not available')
                
                # Get Q spectral operators
                spec_grad_x = data_x['operators']['spec_grad']  # [K_conn, Nx]
                spec_grad_y = data_y['operators']['spec_grad']  # [K_conn, Ny]
                conn_evecs_x = data_x['operators']['conn_evecs']  # [Nx, K_conn]
                conn_evecs_y = data_y['operators']['conn_evecs']  # [Ny, K_conn]
                
                # Project geometric features to connection spectral basis
                # spec_grad @ f_geo -> [K_conn, Ny] matrix in connection coefficients
                # Note: f_geo shape is [B, Nx, C], project each batch sample individually
                B = f_geo_x.shape[0]
                feat_x_complex = spec_grad_x @ f_geo_x.squeeze(0).to(torch.complex128)  # [K_conn, Nx]
                feat_y_complex = spec_grad_y @ f_geo_y.squeeze(0).to(torch.complex128)  # [K_conn, Ny]
                
                # Compute Q_XY: from Y to X (inverse direction for Π_YX)
                Qxy = feat_y_complex @ torch.linalg.pinv(feat_x_complex)  # [K_conn, K_conn]
                
                # Q orthogonal loss: ||Q^H Q - I||_F^2
                I_conn = torch.eye(self.n_conn, dtype=torch.complex128, device=Qxy.device)
                Q_ortho = Qxy.mH @ Qxy  # [K_conn, K_conn]
                l_Q_orth = torch.norm(Q_ortho - I_conn, p='fro') ** 2
                
                self.loss_metrics['l_Q_ortho'] = l_Q_orth
                if self.q_verbose:
                    print(f"[Q] l_Q_ortho={l_Q_orth.item():.6f}")
                
            except Exception as e:
                msg = f'Complex-Q computation failed: {e}'
                if not self._q_failed_once:
                    self._q_failed_once = True
                    raise RuntimeError(msg)
                else:
                    print(f'[WARNING] {msg}')
                    self.loss_metrics['l_Q_ortho'] = torch.tensor(0.0, device=self.device)

        # ------------------- flow matching section ------------------- #
        B, _, C_geo = hx.shape
        mass_x, mass_y = data_x['mass'], data_y['mass']
        tau = torch.rand((B, C_geo), device=hx.device, dtype=hx.dtype) * self.heat_diffusion_tau
        hx = self.heat_diffusion_spectral(hx, mass_x, evals_x, evecs_x, tau)
        hy = self.heat_diffusion_spectral(hy, mass_y, evals_y, evecs_y, tau)

        # sample t_flow [B,1]
        t_flow = torch.rand((B, 1), device=hx.device, dtype=hx.dtype)

        self.loss_metrics['l_cfm'] = self.losses['flow_matching_loss'](self.networks['flow_field'], hx, hy, Pxy, t_flow)

    def validate_single(self, data, timer):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # get previous network state dict
        if self.with_refine > 0:
            state_dict = {'networks': self._get_networks_state_dict()}

        # start record
        timer.start()

        # test-time refinement
        if self.with_refine > 0:
            self.refine(data)
        
        # trim basis
        n_lb = self.opt.get('n_lb', 140)
        n_elas = self.opt.get('n_elas', 60)
        data_x = trim_basis(data_x, n_lb, n_elas)
        data_y = trim_basis(data_y, n_lb, n_elas)

        # feature extractor
        f_geo_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'], data=data_x)
        f_geo_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'], data=data_y)

        neigh_x = get_neigh(data_x['faces'], data_x['verts'].shape[1], K=self.neigh_k, add_self=True)  # [B, Nx, K]
        neigh_y = get_neigh(data_y['faces'], data_y['verts'].shape[1], K=self.neigh_k, add_self=True)  # [B, Ny, K]
        hx = self.networks["cross_attention"](f_geo_x, data_x["dino"], neigh_x, data_x['verts'])
        hy = self.networks["cross_attention"](f_geo_y, data_y["dino"], neigh_y, data_y['verts'])

        # get spectral operators
        evecs_x = data_x['evecs'].squeeze()
        evecs_y = data_y['evecs'].squeeze()
        evals_x = data_x['evals']
        evals_y = data_y['evals']
        evecs_trans_x = data_x['evecs_trans'].squeeze()
        evecs_trans_y = data_y['evecs_trans'].squeeze()
        elas_evecs_x = data_x['elas_evecs'].squeeze()
        elas_evecs_y = data_y['elas_evecs'].squeeze()
        elas_evals_x = data_x['elas_evals']
        elas_evals_y = data_y['elas_evals']
        elas_trans_x = data_x['elas_evecs_trans'].squeeze()
        elas_trans_y = data_y['elas_evecs_trans'].squeeze()
        elas_evecs_trans_x = data_x['elas_evecs_trans']
        elas_evecs_trans_y = data_y['elas_evecs_trans']
        elas_sqrtMk_x = data_x['elas_sqrtMk']
        elas_sqrtMk_y = data_y['elas_sqrtMk']
        elas_mass_x = data_x['elas_mass'].squeeze()
        elas_mass_y = data_y['elas_mass'].squeeze()

        if self.non_isometric:
            hx = F.normalize(hx, dim=-1, p=2)
            hy = F.normalize(hy, dim=-1, p=2)

            # nearest neighbour query
            p2p = nn_query(hx, hy).squeeze()
            # compute Pyx from functional map
            Cxy = evecs_trans_y @ evecs_x[p2p]
            Pyx = evecs_y @ Cxy @ evecs_trans_x
        else:
            # compute Pxy
            Pyx = self.compute_permutation_matrix(hy, hx, bidirectional=False).squeeze()
            Cxy = evecs_trans_y @ (Pyx @ evecs_x)
            elas_Cxy = elas_trans_y @ (Pyx @ elas_evecs_x)

            # convert functional map to point-to-point map
            p2p = hybrid_fmap2pointmap(Cxy, evecs_x, evecs_y, elas_Cxy, elas_evecs_x, elas_evecs_y, elas_mass_x, elas_mass_y)

            # compute Pyx from functional map
            Pyx = evecs_y @ Cxy @ evecs_trans_x

        # finish record
        timer.record()

        # resume previous network state dict
        if self.with_refine > 0:
            self.resume_model(state_dict, net_only=True, verbose=False)
        return p2p, Pyx, Cxy

    def compute_permutation_matrix(self, feat_x, feat_y, bidirectional=False, normalize=True):
        if normalize:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)
        similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

        # sinkhorn normalization
        Pxy = self.networks['permutation'](similarity)

        if bidirectional:
            Pyx = self.networks['permutation'](similarity.transpose(1, 2))
            return Pxy, Pyx
        else:
            return Pxy

    def refine(self, data):
        self.networks['permutation'].hard = False
        self.networks['fmap_net'].bidirectional = True

        with torch.set_grad_enabled(True):
            for _ in range(self.with_refine):
                self.curr_iter += 1
                self.feed_data(data)
                self.optimize_parameters()
        self.curr_iter = 0

        self.networks['permutation'].hard = True
        self.networks['fmap_net'].bidirectional = False

    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        # change permutation prediction status
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = True
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = False
        super(SGMatch_Model, self).validation(dataloader, tb_logger, update)
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = False
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = True
    
    def optimize_parameters(self):
        n_lb = self.opt.get('n_lb', 140)
        n_elas = self.opt.get('n_elas', 60)

        # Loss normalization and weight scheduling
        if n_lb > 0 and n_elas > 0:
            # Normalize LB and Elas Loss; 
            w_lb = 20000 / (n_lb * n_lb)
            w_elas = 20000 / (n_elas * n_elas)

            # early schduler for Elas Loss
            weight_schedule = self.opt['train'].get('weight_schedule', False)
            if weight_schedule:
                def linear_anneal(current_iter, total_iters):
                    return min(max(current_iter / total_iters, 0.0), 1.0)
                curr_iter = self.curr_iter
                anneal_weight = linear_anneal(curr_iter, weight_schedule)

                w_elas = w_elas * anneal_weight

            # applying loss weights
            self.loss_metrics['l_bij'] *= w_lb
            self.loss_metrics['l_orth'] *= w_lb
            self.loss_metrics['l_align'] *= w_lb

            self.loss_metrics['l_elas_bij'] *= w_elas
            self.loss_metrics['l_elas_orth'] *= w_elas
            self.loss_metrics['l_elas_align'] *= w_elas

        # Q branch weight (only acts as a switch in feed_data; real weight applies here)
        if 'l_Q_ortho' in self.loss_metrics and self.w_q_orth > 0:
            self.loss_metrics['l_Q_ortho'] = self.loss_metrics['l_Q_ortho'] * self.w_q_orth

        # Jacobian loss weights (flip rate / mean distortion are metrics only)
        if 'l_J_det' in self.loss_metrics:
            self.loss_metrics['l_J_det'] = self.loss_metrics['l_J_det'] * self.w_jacobian_det
            self.loss_metrics['l_J_distortion'] = self.loss_metrics['l_J_distortion'] * self.w_jacobian_distortion

        # compute total loss
        loss = 0.0
        for k, v in self.loss_metrics.items():
            if k in ('l_total', 'J_flip_rate', 'J_mean_distortion'):
                continue
            loss += v

        # update loss metrics
        self.loss_metrics['l_total'] = loss

        # zero grad
        for name in self.optimizers:
            self.optimizers[name].zero_grad()

        # backward pass
        loss.backward()

        # clip gradient for stability
        for key in self.networks:
            torch.nn.utils.clip_grad_norm_(self.networks[key].parameters(), 1.0)

        # update weight
        for name in self.optimizers:
            self.optimizers[name].step()

    def heat_diffusion_spectral(self, u, mass, evals, evecs, t, eps=1e-8):
        t = torch.clamp(t.to(device=u.device, dtype=u.dtype), min=eps)  # clamp t to positive
        u_spec = to_basis(u, evecs, mass) # [B,K,C] project to spectral: u_spec = Phi^T (M u)

        # diffuse: u_t_spec = exp(-lambda * t) * u_spec
        diffusion = torch.exp(-evals.unsqueeze(-1) * t.unsqueeze(1))  # [B,K,C] # evals: [B,K] -> [B,K,1], t: [B,C] -> [B,1,C]
        u_t_spec = diffusion * u_spec  # [B,K,C]
        u_t = from_basis(u_t_spec, evecs)   # back to vertices [B,N,C]
        return u_t
