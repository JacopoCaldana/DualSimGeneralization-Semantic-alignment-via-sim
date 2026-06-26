"""
Deep unrolling of Algorithm  (SCA).

Clean-room implementation. We unfold K layers of the projected-gradient update

    xi^{k+1} = Pi_[0,2pi)( S^k xi^k + W^k g^k ),   g^k = -grad_xi NMSE

with S^k = I (fixed) and lambda = 0, per the professor's simplification.

The innovation g^k is the *negative gradient* of the NMSE w.r.t. the SIM phases --
exactly the SCA innovation of eq. (36) up to the scale absorbed by W^k. It is computed
by autograd through the physical cascade Z = G_R H G_T, with the NMSE in its cosine
form (1 - |<Z,A>|^2/(||Z||^2 ||A||^2)): identical gradient to the closed-form-beta
residual (envelope theorem) but always bounded in [0,1], avoiding the beta blow-up when
||Z|| -> 0. The innovation is kept in the graph (create_graph) so the per-layer
parameters can be trained by backprop through the unrolled iterations.

Design notes (what actually makes the unrolled curve fall below the analog everywhere):
the pure scalar gradient step stalls at the symmetric
xi=0 saddle and oscillates, so -- following "add complexity only if the simple version
fails" -- we use: (i) `innovation_norm='rms'` to fix the (tiny) gradient scale, (ii)
`coupling='diagonal'` per-coordinate learned steps (a learned diagonal preconditioner),
and (iii) `momentum=True`, an Adam-style learnable EMA of the innovation. Training uses
deep supervision (every layer's NMSE) + best-curve restore.

Physics is reused verbatim from the baseline `DualSIMoptimizerTorch`
(`get_effective_cascade_functional_batched`) so the unrolled net and the analog
baseline (`optimize_alternating`) act on *identical* SIM hardware.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _split_sides(vals, n_left):
    return list(vals[:n_left]), list(vals[n_left:])


class DualSIMUnrolledSCA(nn.Module):
    def __init__(self, sim, K, init_w=0.2, coupling="scalar",
                 innovation_norm="rms", project=False, momentum=False, init_mu=0.9,
                 learn_S=False, first_order=False, analytic=False, checkpoint=False,
                 eps=1e-12):
        """
        sim          : a built dualsim.DualSIMoptimizerTorch (provides W buffers + cascade).
                       NOT registered as a submodule -> its phases are never trained.
        K            : number of unrolled layers.
        init_w       : initial per-layer step size.
        coupling     : 'scalar' (one step/side/layer) or 'diagonal' (per-phase step).
        innovation_norm : 'rms' (per-sample per-element RMS=1, default), 'sign'
                          (per-coordinate, Adam/RMSprop-like), 'unit' (L2=1), or 'none'
                          (raw gradient; tiny-magnitude -> needs huge/calibrated W).
        momentum     : if True, an Adam-style learnable EMA (per-layer mu) of the
                       innovation, with bias correction.
        project      : if True, wrap phases to [0,2pi) each layer (optional; phases enter
                       via exp(j xi), so it is mathematically inert -- default off).
        """
        super().__init__()
        object.__setattr__(self, "_sim", sim)  # keep sim out of .parameters()/.modules()
        self.K = int(K)
        self.coupling = coupling
        self.innovation_norm = innovation_norm
        self.project = project
        self.momentum = momentum
        self.learn_S = learn_S
        self.first_order = first_order  # detach innovation (no double-backprop) -> big configs
        self.analytic = analytic        # innovation by hand-coded backward (no autograd.grad)
        self.checkpoint = checkpoint    # gradient-checkpoint each layer (only with analytic)
        self.eps = eps

        self.sizes_T = [p.shape[0] for p in sim.xi_T]
        self.sizes_R = [p.shape[0] for p in sim.xi_R]
        self.n_T, self.n_R = sum(self.sizes_T), sum(self.sizes_R)

        shp_T = (self.K,) if coupling == "scalar" else (self.K, self.n_T)
        shp_R = (self.K,) if coupling == "scalar" else (self.K, self.n_R)
        if coupling not in ("scalar", "diagonal"):
            raise ValueError(f"unknown coupling {coupling!r}")
        self.w_T = nn.Parameter(torch.full(shp_T, float(init_w)))
        self.w_R = nn.Parameter(torch.full(shp_R, float(init_w)))
        if learn_S:
            # S^k acts on xi^k (paper eq. 38-39). Init = identity (=1, scalar/diagonal) so
            # the model starts identical to S=I; train it gently (low lr + ||S-I||^2 reg) --
            # prior sessions found S with a normal lr scrambles the phases and diverges.
            self.S_T = nn.Parameter(torch.ones(shp_T))
            self.S_R = nn.Parameter(torch.ones(shp_R))
        if momentum:
            # mu = sigmoid(mu_raw) in (0,1); heavy-ball/Adam-style first moment on the innovation
            mu0 = float(torch.logit(torch.tensor(init_mu)))
            self.mu_T = nn.Parameter(torch.full(shp_T, mu0))
            self.mu_R = nn.Parameter(torch.full(shp_R, mu0))

    # ---- physics + NMSE ----------------------------------------------------
    def _cascade(self, H, xT, xR):
        return self._sim.get_effective_cascade_functional_batched(H, xT, xR)

    def _nmse_per_sample(self, Z, A, A2):
        """Cosine form: NMSE = min_beta ||beta Z - A||^2 / ||A||^2 = 1 - |<Z,A>|^2/(||Z||^2 ||A||^2).

        Equivalent to the closed-form-beta NMSE (envelope theorem => identical gradient w.r.t.
        the phases), but always bounded in [0,1] -- avoids the beta-blowup when ||Z|| -> 0.
        """
        ip = (Z.conj() * A).sum((-2, -1)).abs() ** 2          # |<Z,A>|^2
        zz = (Z.conj() * Z).sum((-2, -1)).real                 # ||Z||^2
        cos2 = ip / (zz * A2 + self.eps)
        return 1.0 - cos2                                      # [B] in [0,1]

    def _innovation(self, H, A, A2, xT, xR, create_graph):
        """g = -grad_xi NMSE (descent direction), returned per physical layer."""
        xT = [x.requires_grad_(True) for x in xT]
        xR = [x.requires_grad_(True) for x in xR]
        Z = self._cascade(H, xT, xR)
        nmse = self._nmse_per_sample(Z, A, A2)  # [B], differentiable
        grads = torch.autograd.grad(nmse.sum(), xT + xR, create_graph=create_graph)
        gT, gR = _split_sides(grads, len(xT))
        gT = [-g for g in gT]
        gR = [-g for g in gR]
        return nmse, gT, gR

    # ---- analytic innovation (hand-coded backward through the phase-only cascade) ----
    @staticmethod
    def _build_G(W_list, phases):
        """Forward cascade G = prod_l diag(e^{j xi_l}) W_l, returning G and per-layer (U_l, v_l)
        with U_l = W_l G_{l-1}.  Batched: phases[l] is [B, out_l], W_list[l] is [out_l, in_l]."""
        B = phases[0].shape[0]
        input_dim = W_list[0].shape[1]
        G = torch.eye(input_dim, dtype=torch.complex64, device=phases[0].device)
        G = G.unsqueeze(0).expand(B, -1, -1)
        inter = []
        for l in range(len(W_list)):
            U = W_list[l] @ G                       # [B, out_l, input_dim]
            v = torch.exp(1j * phases[l])           # [B, out_l]
            inter.append((U, v))
            G = v.unsqueeze(-1) * U                  # diag(v) U
        return G, inter

    @staticmethod
    def _cascade_backward(W_list, inter, Gbar):
        """Reverse sweep: given cotangent Gbar on G_final, return g_l = -2 Im{v_l * vbar_l}
        per layer (vbar from (Gbar.conj()*U).sum) and propagate Gbar <- W_l^H (conj(v_l) Gbar)."""
        g_list = [None] * len(W_list)
        for l in reversed(range(len(W_list))):
            U, v = inter[l]
            vbar = (Gbar.conj() * U).sum(-1)                 # [B, out_l]
            g_list[l] = (-2.0) * (v * vbar).imag             # [B, out_l] (real)
            Ubar = v.conj().unsqueeze(-1) * Gbar
            Gbar = W_list[l].conj().transpose(-2, -1) @ Ubar  # [B, in_l, input_dim]
        return g_list

    def _innovation_analytic(self, H, A, A2, xT, xR):
        """Same (nmse, gT, gR) as _innovation but with the gradient computed by an explicit
        adjoint pass (no autograd.grad) -> the layer is a pure forward fn => checkpointable."""
        W_T = self._sim._get_W_list("W_T", self._sim.L_T)
        W_R = self._sim._get_W_list("W_R", self._sim.L_R)
        GT, interT = self._build_G(W_T, xT)
        GR, interR = self._build_G(W_R, xR)
        Z = GR @ H @ GT                                       # [B, omega, theta]
        # The cotangent involves beta = <Z,A>/||Z||^2; for deep cascades ||Z|| can be tiny so
        # beta and the 2nd-order terms (~1/||Z||^2) overflow in fp32. Compute this scalar/cotangent
        # part in fp64 (cheap: small tensors), then cast back to fp32 for the big matmuls.
        Zd, Ad, A2d = Z.to(torch.complex128), A.to(torch.complex128), A2.double()
        p = (Zd.conj() * Ad).sum((-2, -1))                    # <Z,A>  [B]
        s = (Zd.conj() * Zd).sum((-2, -1)).real               # ||Z||^2 [B]
        nmse = (1.0 - (p.abs() ** 2) / (s * A2d + self.eps)).to(Z.real.dtype)  # cosine NMSE
        beta = (p / (s + self.eps)).detach()                  # closed-form, detached [B]
        R = Ad - beta[:, None, None] * Zd                     # residual A - beta Z
        C = (beta.conj()[:, None, None] * R).to(Z.dtype)      # cotangent beta* R -> fp32
        Hh = H.conj().transpose(-2, -1)
        GT_bar = Hh @ (GR.conj().transpose(-2, -1) @ C)       # cotangent on G_T [B, N_T, theta]
        GR_bar = C @ (GT.conj().transpose(-2, -1) @ Hh)       # cotangent on G_R [B, omega, N_R]
        inv = 1.0 / A2                                        # NMSE normalisation
        gT = [g * inv for g in self._cascade_backward(W_T, interT, GT_bar)]
        gR = [g * inv for g in self._cascade_backward(W_R, interR, GR_bar)]
        return nmse, gT, gR

    def _innovation_ckpt(self, H, A, A2, xT, xR):
        """Gradient-checkpointed analytic innovation: each layer's cascade is recomputed in
        backward instead of stored -> deep-supervision 2nd-order at O(1)-in-K memory."""
        nT = len(xT)

        def fn(*phases):
            nmse, gT, gR = self._innovation_analytic(H, A, A2, list(phases[:nT]), list(phases[nT:]))
            return (nmse, *gT, *gR)

        out = checkpoint(fn, *xT, *xR, use_reentrant=False)
        return out[0], list(out[1:1 + nT]), list(out[1 + nT:])

    def _normalize(self, g_list):
        if self.innovation_norm == "none":
            return g_list
        if self.innovation_norm == "sign":  # per-coordinate (Adam/RMSprop-like), escapes saddles
            return [g / (g.abs().detach() + self.eps) for g in g_list]
        flat = torch.cat([g.reshape(g.shape[0], -1) for g in g_list], dim=1)  # [B, n]
        if self.innovation_norm == "rms":          # one scale per sample (whole side)
            s = flat.pow(2).mean(dim=1, keepdim=True).sqrt()
        elif self.innovation_norm == "unit":
            s = flat.norm(dim=1, keepdim=True)
        else:
            raise ValueError(self.innovation_norm)
        s = (s + self.eps).detach()
        return [g / s.view(-1, *([1] * (g.dim() - 1))) for g in g_list]

    def _steps(self, wk, sizes):
        """Return per-layer step tensors (scalar or [m_l]) for one side at layer k."""
        if self.coupling == "scalar":
            return [wk] * len(sizes)
        return [c[None, :] for c in torch.split(wk, sizes)]  # diagonal -> [1, m_l]

    # ---- unrolled forward --------------------------------------------------
    def forward(self, H, A, record=False, create_graph=None, supervision="final",
                gamma=0.9, return_phases=False):
        """
        H: [B, N_R, N_T] complex, A: [omega, theta] complex (fixed).
        Returns (loss, curve):
          loss  : scalar training loss (mean final NMSE, or deep-supervised sum).
          curve : list of mean NMSE per layer [xi^0, ..., xi^K] (len K+1) if record else [].
        If return_phases, also returns the final per-layer phase lists (xT, xR).
        """
        if create_graph is None:
            create_graph = self.training
        B, dev = H.shape[0], H.device
        A2 = torch.norm(A) ** 2
        xT = [torch.zeros(B, m, device=dev) for m in self.sizes_T]
        xR = [torch.zeros(B, m, device=dev) for m in self.sizes_R]
        vT = [torch.zeros(B, m, device=dev) for m in self.sizes_T] if self.momentum else None
        vR = [torch.zeros(B, m, device=dev) for m in self.sizes_R] if self.momentum else None

        curve = []
        loss = torch.zeros((), device=dev)  # real accumulator (deep supervision)
        wsum = 0.0
        inno_cg = create_graph and not self.first_order  # 2nd-order only if not first_order
        # first_order frees each layer's cascade graph -> per-layer nmse can't backprop;
        # fall back to final-layer supervision in that case. analytic keeps deep-sup at any scale.
        deep = (supervision == "all") and (self.analytic or not self.first_order)
        ckpt = self.analytic and self.checkpoint and create_graph and torch.is_grad_enabled()
        for k in range(self.K):
            if self.analytic:
                if ckpt:
                    nmse_k, gT, gR = self._innovation_ckpt(H, A, A2, xT, xR)
                else:
                    nmse_k, gT, gR = self._innovation_analytic(H, A, A2, xT, xR)
            else:
                nmse_k, gT, gR = self._innovation(H, A, A2, xT, xR, inno_cg)
            if record:
                curve.append(nmse_k.mean().item())
            if deep:
                wk = gamma ** (self.K - 1 - k)
                loss = loss + wk * nmse_k.mean()
                wsum += wk
            gT, gR = self._normalize(gT), self._normalize(gR)
            if self.momentum:  # Adam-style EMA of the innovation + bias correction
                muT = self._steps(torch.sigmoid(self.mu_T[k]), self.sizes_T)
                muR = self._steps(torch.sigmoid(self.mu_R[k]), self.sizes_R)
                vT = [mu * v + (1 - mu) * g for mu, v, g in zip(muT, vT, gT)]
                vR = [mu * v + (1 - mu) * g for mu, v, g in zip(muR, vR, gR)]
                gT = [v / (1 - mu ** (k + 1)) for v, mu in zip(vT, muT)]
                gR = [v / (1 - mu ** (k + 1)) for v, mu in zip(vR, muR)]
            sT, sR = self._steps(self.w_T[k], self.sizes_T), self._steps(self.w_R[k], self.sizes_R)
            if self.learn_S:  # xi^{k+1} = S^k xi^k + W^k g^k  (paper eq. 38-39)
                aT = self._steps(self.S_T[k], self.sizes_T)
                aR = self._steps(self.S_R[k], self.sizes_R)
                xT = [a * x + s * g for x, a, s, g in zip(xT, aT, sT, gT)]
                xR = [a * x + s * g for x, a, s, g in zip(xR, aR, sR, gR)]
            else:
                xT = [x + s * g for x, s, g in zip(xT, sT, gT)]
                xR = [x + s * g for x, s, g in zip(xR, sR, gR)]
            if self.project:
                two_pi = 2 * torch.pi
                xT = [x % two_pi for x in xT]
                xR = [x % two_pi for x in xR]
            if not create_graph:  # eval: detach to keep each layer's graph independent
                xT = [x.detach() for x in xT]
                xR = [x.detach() for x in xR]
                if self.momentum:
                    vT = [v.detach() for v in vT]
                    vR = [v.detach() for v in vR]

        Z = self._cascade(H, xT, xR)
        nmse_final = self._nmse_per_sample(Z, A, A2)
        if record:
            curve.append(nmse_final.mean().item())
        if deep:
            loss = (loss + nmse_final.mean()) / (wsum + 1.0)
        else:
            loss = nmse_final.mean()
        if return_phases:
            return loss, curve, xT, xR
        return loss, curve

    def infer_phases(self, H, A):
        """Single-shot inference: run the K-layer unroll and return the final per-layer
        phase lists (xi_T, xi_R) for channel(s) H and target A. The (autograd) innovation
        needs grad enabled even though no parameter is updated, so we do NOT use no_grad."""
        was_training = self.training
        self.eval()
        with torch.enable_grad():
            _, _, xT, xR = self.forward(H, A, record=False, create_graph=False,
                                        supervision="final", return_phases=True)
        if was_training:
            self.train()
        return [x.detach() for x in xT], [x.detach() for x in xR]

    def reg_S(self):
        """||S^k - I||_F^2 summed over layers (keeps the learnable S near identity)."""
        if not self.learn_S:
            return self.w_T.new_zeros(())
        return ((self.S_T - 1.0) ** 2).sum() + ((self.S_R - 1.0) ** 2).sum()

    def eval_curve(self, H, A):
        """NMSE-vs-layer curve (len K+1), averaged over the channels in H. No training."""
        was_training = self.training
        self.eval()
        _, curve = self.forward(H, A, record=True, create_graph=False, supervision="final")
        if was_training:
            self.train()
        return curve


def train_sca_unrolled(
    sim, A, K,
    N_T=None, N_R=None, epochs=30, iters_per_epoch=50, batch=32,
    lr_w=1e-2, init_w=0.1, coupling="scalar", innovation_norm="rms",
    supervision="all", gamma=1.0, project=False, grad_clip=1.0,
    momentum=False, init_mu=0.9, learn_S=False, s_lr_ratio=50.0, reg_S=1e-2,
    first_order=False, analytic=False, checkpoint=False,
    calibrate=False, target_step=0.1,
    device=None, seed=0, log_every=5, val_H=None,
):
    """
    Train the per-layer step sizes W^k on a FIXED A with random channels H~CN(0,1)/sqrt(2).
    Returns (model, history) where history is the list of (epoch, train_loss, val_nmse_final).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    A = A.to(device)
    if N_T is None:
        N_T = sim.get_buffer(f"W_T_{sim.L_T - 1}").shape[0]
    if N_R is None:
        N_R = sim.get_buffer("W_R_0").shape[1]

    model = DualSIMUnrolledSCA(sim, K, init_w=init_w, coupling=coupling,
                               innovation_norm=innovation_norm, project=project,
                               momentum=momentum, init_mu=init_mu, learn_S=learn_S,
                               first_order=first_order, analytic=analytic,
                               checkpoint=checkpoint).to(device)
    # safety: only the step sizes (+ momentum, + S) are trainable (sim phases excluded)
    n_params = sum(p.numel() for p in model.parameters())
    expected = model.w_T.numel() + model.w_R.numel()
    if momentum:
        expected += model.mu_T.numel() + model.mu_R.numel()
    if learn_S:
        expected += model.S_T.numel() + model.S_R.numel()
    assert n_params == expected, "unexpected trainable params"

    gen = torch.Generator(device=device).manual_seed(seed)

    def sample_H(b):
        H = torch.randn(b, N_R, N_T, dtype=torch.complex64, device=device, generator=gen)
        return H / (2 ** 0.5)

    # Calibrate the initial step so the first layer's RMS phase-step ~ target_step (rad).
    # With raw innovation the gradient magnitude is tiny/config-dependent; this puts W^0 in a
    # sensible basin (~ the analog step) so training starts from a comparable point.
    if calibrate and innovation_norm == "none":
        A2 = torch.norm(A.to(device)) ** 2
        H0 = sample_H(min(batch, 16))
        xT0 = [torch.zeros(H0.shape[0], m, device=device) for m in model.sizes_T]
        xR0 = [torch.zeros(H0.shape[0], m, device=device) for m in model.sizes_R]
        _, gT0, gR0 = model._innovation(H0, A.to(device), A2, xT0, xR0, create_graph=False)
        gflat = torch.cat([g.reshape(g.shape[0], -1) for g in gT0 + gR0], dim=1)
        typ = gflat.pow(2).mean().sqrt().item()  # RMS of raw gradient at init
        w0 = float(target_step) / (typ + 1e-12)
        with torch.no_grad():
            model.w_T.fill_(w0)
            model.w_R.fill_(w0)
        print(f"  calibrated init step: RMS|g|={typ:.2e} -> w0={w0:.2f}")

    if learn_S:  # S trained with a much smaller lr (prior lesson: high lr_S scrambles phases)
        s_params = [model.S_T, model.S_R]
        s_ids = {id(p) for p in s_params}
        base_params = [p for p in model.parameters() if id(p) not in s_ids]
        opt = torch.optim.Adam([
            {"params": base_params, "lr": lr_w},
            {"params": s_params, "lr": lr_w / s_lr_ratio},
        ])
        print(f"  S learnable: lr_S={lr_w / s_lr_ratio:.2e} (lr_W/{s_lr_ratio:g}), "
              f"reg ||S-I||^2 weight={reg_S:g}")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr_w)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * iters_per_epoch)

    def snapshot():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Best-model restore on the whole-curve AREA (mean NMSE over layers): protects against
    # the depth instability where training wanders to a worse-shaped curve. The untrained
    # init (momentum constant-step) is itself a candidate.
    history, best_area, best_state = [], float("inf"), None
    if val_H is not None:
        c0 = model.eval_curve(val_H.to(device), A)
        best_area, best_state = sum(c0) / len(c0), snapshot()

    model.train()
    for ep in range(epochs):
        running = 0.0
        for _ in range(iters_per_epoch):
            H = sample_H(batch)
            opt.zero_grad(set_to_none=True)
            loss, _ = model(H, A, supervision=supervision, gamma=gamma)
            if learn_S:
                loss = loss + reg_S * model.reg_S()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            if not torch.isfinite(loss):  # skip non-finite steps (numerically hard deep configs)
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
            opt.step()
            sched.step()
            running += loss.item()
        running /= iters_per_epoch
        val = float("nan")
        if val_H is not None:
            c = model.eval_curve(val_H.to(device), A)
            val, area = c[-1], sum(c) / len(c)
            if area < best_area:
                best_area, best_state = area, snapshot()
        history.append((ep, running, val))
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"  [unroll ep {ep:3d}/{epochs}] train_loss={running:.4f}"
                  + (f" | val_NMSE_K={val:.4f}" if val_H is not None else "")
                  + f" | lr={sched.get_last_lr()[0]:.2e}")
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  restored best model (val curve-area={best_area:.4f})")
    return model, history
