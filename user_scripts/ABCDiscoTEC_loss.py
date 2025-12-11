import torch.nn.functional as F
import torch
import torch.nn as nn

def _pairwise_centered_l1(x):
    # x: [N]
    # distance matrix (|xi-xj|), then double-center
    diff = torch.abs(x[:, None] - x[None, :])
    A = diff - diff.mean(dim=0, keepdim=True) - diff.mean(dim=1, keepdim=True) + diff.mean()
    return A

def distance_correlation(x, y, eps=1e-8):
    # x,y: [N] (use background subset)
    Ax = _pairwise_centered_l1(x)
    Ay = _pairwise_centered_l1(y)
    dcov2 = (Ax * Ay).mean()
    dvarx = (Ax * Ax).mean().clamp_min(eps)
    dvary = (Ay * Ay).mean().clamp_min(eps)
    return (dcov2 / torch.sqrt(dvarx * dvary)).clamp_min(0.0)  # dCorr in [0,1]


def abcdiscotec_loss(
    logit1, logit2, y_true, *,
    # ABCD plane hyperparams
    b1=0.6, b2=0.6, k=75.0,
    # weights
    w_cls=1.0, w_disco_bkg=0.5, w_disco_sig=0.2, w_closure=0.4,
    w_r=0.0,              # << weight for normalized contamination r
    w_sigCR=0.0,          # optional legacy term; can stay 0 when using r
    # numerics
    eps=1e-6, deltaA_min=1e-3
):
    """
    Implements BCE on both heads, DisCo on bkg (+tiny on sig), ABCD closure on bkg,
    and the paper's normalized signal contamination r = δ_A^{-1}(δ_B+δ_C-δ_D).
    """
    y = y_true.float()

    # --- classification (either keep both heads, or use logitA = logit1+logit2) ---
    # loss_cls = 0.5 * (
    #     F.binary_cross_entropy_with_logits(logit1, y) +
    #     F.binary_cross_entropy_with_logits(logit2, y)
    #     )
    loss_cls = torch.sqrt(
          (F.binary_cross_entropy_with_logits(logit1, y)**2 +
           F.binary_cross_entropy_with_logits(logit2, y)**2 )/2
          )

    # probabilities for ABCD/DisCo
    s1 = torch.sigmoid(logit1)
    s2 = torch.sigmoid(logit2)

    bkg = (y_true == 0)
    sig = ~bkg

    # --- DisCo ---
    if bkg.sum() >= 2:
        disco_bkg = distance_correlation(s1[bkg], s2[bkg])
    else:
        disco_bkg = torch.zeros((), device=logit1.device)

    if sig.sum() >= 2:
        disco_sig = distance_correlation(s1[sig], s2[sig])
    else:
        disco_sig = torch.zeros((), device=logit1.device)

    # --- differentiable ABCD regions ---
    H1 = torch.sigmoid(k * (s1 - b1))
    H2 = torch.sigmoid(k * (s2 - b2))

    A = H1 * H2
    B = (1 - H1) * H2
    C = H1 * (1 - H2)
    D = (1 - H1) * (1 - H2)

    # soft counts
    NA_b = A[bkg].sum(); NB_b = B[bkg].sum(); NC_b = C[bkg].sum(); ND_b = D[bkg].sum()
    NA_s = A[sig].sum(); NB_s = B[sig].sum(); NC_s = C[sig].sum(); ND_s = D[sig].sum()

    # --- closure on background ---                 (ABCDisCoTEC eq. 5)
    nom_closure     = NA_b * ND_b - NB_b * NC_b
    denom_closure   = NA_b * ND_b + NB_b * NC_b + eps
    closure         = (nom_closure / denom_closure)**2

    # --- normalized signal contamination r (paper Eq. 2.8) ---
    # δ_i = N_{i,s} / N_{i,b}
    deltaA = NA_s / (NA_b + eps)
    deltaB = NB_s / (NB_b + eps)
    deltaC = NC_s / (NC_b + eps)
    deltaD = ND_s / (ND_b + eps)

    denom = deltaA.clamp_min(deltaA_min)  # avoid exploding gradients if δ_A≈0
    r = (deltaB + deltaC - deltaD) / denom
    loss_r = r**2  # penalize |r| >> 0

    # --- (optional) legacy signal-in-CR penalty ---
    if w_sigCR > 0.0 and sig.any():
        SA = NA_s.clamp_min(eps)
        sigCR = (NB_s + NC_s + ND_s) / SA
    else:
        sigCR = torch.zeros((), device=logit1.device)

    disco = w_disco_bkg * disco_bkg + w_disco_sig * disco_sig
    total = w_cls * loss_cls + disco + w_closure * closure + w_r * loss_r + w_sigCR * sigCR

    return total, {
        'total':    float(total.detach()),
        'loss_cls': float(loss_cls.detach()),
        'disco':    float(disco.detach()),
        'closure':  float(closure.detach()),
        'r':        float(r.detach()),
        'loss_r':   float(loss_r.detach()),
        'NA_b': float(NA_b.detach()), 'NB_b': float(NB_b.detach()),
        'NC_b': float(NC_b.detach()), 'ND_b': float(ND_b.detach()),
        'NA_s': float(NA_s.detach()), 'NB_s': float(NB_s.detach()),
        'NC_s': float(NC_s.detach()), 'ND_s': float(ND_s.detach()),
        'deltaA': float(deltaA.detach()), 'deltaB': float(deltaB.detach()),
        'deltaC': float(deltaC.detach()), 'deltaD': float(deltaD.detach()),
    }


class ABCLagrangian(nn.Module):
  
  def __init__(self, eps_closure, eps_disco, k, alpha_lr=1e-3, numeric_eps=1e-6):
    super().__init__()
    self.eps_closure = float(eps_closure)
    self.eps_disco   = float(eps_disco)
    self.k = float(k)
    self.alpha_lr = float(alpha_lr)
    self.numeric_eps = float(numeric_eps)

    self.register_buffer("alpha_closure", torch.zeros((), dtype=torch.float32))
    self.register_buffer("alpha_disco",   torch.zeros((), dtype=torch.float32))
    self._delta_closure = None
    self._delta_disco   = None

  def forward(self, logit1, logit2, y_true, b1, b2):
    y = y_true.to(dtype=logit1.dtype)
    # loss_bce = 0.5 * (
    #   F.binary_cross_entropy_with_logits(logit1, y) +
    #   F.binary_cross_entropy_with_logits(logit2, y)
    # )

    loss_bce = torch.sqrt(
          (F.binary_cross_entropy_with_logits(logit1, y)**2 +
           F.binary_cross_entropy_with_logits(logit2, y)**2)/2
    )

    s1 = torch.sigmoid(logit1)
    s2 = torch.sigmoid(logit2)
    bkg = (y_true == 0)
    sig = ~bkg 

    if bkg.sum() >= 2:
      disco_bkg = distance_correlation(s1[bkg], s2[bkg])
    else:
      disco_bkg = torch.zeros((), device=logit1.device, dtype=logit1.dtype)

    if sig.sum() >= 2:
        disco_sig = distance_correlation(s1[sig], s2[sig])
    else:
        disco_sig = torch.zeros((), device=logit1.device)

    

    # allow scalar or per-sample thresholds; broadcasting will just work
    b1 = torch.as_tensor(b1, device=s1.device, dtype=s1.dtype)
    b2 = torch.as_tensor(b2, device=s2.device, dtype=s2.dtype)

    H1 = torch.sigmoid(self.k * (s1 - b1))
    H2 = torch.sigmoid(self.k * (s2 - b2))

    A = H1 * H2
    B = (1 - H1) * H2
    C = H1 * (1 - H2)
    D = (1 - H1) * (1 - H2)

    NA_b = A[bkg].sum()
    NB_b = B[bkg].sum()
    NC_b = C[bkg].sum()
    ND_b = D[bkg].sum()

    nom   = NA_b * ND_b - NB_b * NC_b
    denom = NA_b * ND_b + NB_b * NC_b + self.numeric_eps
    loss_closure = (nom / denom) ** 2

    delta_closure = loss_closure - self.eps_closure
    delta_disco_b   = disco_bkg    - self.eps_disco
    delta_disco_s   = disco_sig    - self.eps_disco # * 4
    delta_disco     = delta_disco_b + delta_disco_s

    loss = loss_bce + self.alpha_closure * delta_closure + self.alpha_disco * delta_disco

    self._delta_closure = delta_closure.detach()
    self._delta_disco   = delta_disco.detach()
    return loss, {
    'loss':             float(loss.detach()),
    'loss_bce':         float(loss_bce.detach()),
    'disco_bkg':        float(disco_bkg.detach()),
    'disco_sig':        float(disco_sig.detach()),
    'loss_closure':     float(loss_closure.detach()),
    'alpha_closure':    float(self.alpha_closure.detach()),
    'alpha_disco':      float(self.alpha_disco.detach()),
    }

  @torch.no_grad()
  def dual_ascent(self):
    if self._delta_closure is None or self._delta_disco is None:
      return
    self.alpha_closure.add_(self.alpha_lr * self._delta_closure).clamp_(min=0.0)
    self.alpha_disco.add_(self.alpha_lr * self._delta_disco).clamp_(min=0.0)


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------



def select_leading_vertices(logit1, logit2, event_idx, y_true, tol=1e-8):
    """
    Select exactly one leading vertex per event:
      - p1 = sigmoid(logit1), p2 = sigmoid(logit2)
      - score = p1 * p2
      - leading vertex = argmax(score) per event
        (ties broken by smallest global index)
    Works fully columnar via scatter_reduce_, no Python loop.
    """
    # Flatten to 1D
    logit1    = logit1.view(-1)
    logit2    = logit2.view(-1)
    y_true    = y_true.view(-1)
    ev_flat   = event_idx.view(-1).long()

    device = logit1.device
    dtype  = logit1.dtype

    # Per-vertex probabilities and scores
    p1 = torch.sigmoid(logit1)
    p2 = torch.sigmoid(logit2)
    scores = p1 * p2  # [N_vtx]

    # Map arbitrary event IDs -> [0 .. num_unique_events-1]
    _, inv = torch.unique(ev_flat, return_inverse=True)
    num_unique_events = int(inv.max().item()) + 1  # or len(unique_ids)

    # 1) Max score per event
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=scores.dtype)
    max_scores = torch.full((num_unique_events,), neg_inf,
                            device=device, dtype=scores.dtype)
    max_scores.scatter_reduce_(
        dim=0,
        index=inv,
        src=scores,
        reduce="amax",
        include_self=True,
    )

    # 2) Mark all vertices that are "close enough" to the max of their event
    max_scores_broad = max_scores[inv]              # [N_vtx], per-vertex event max
    is_candidate = scores >= (max_scores_broad - tol)

    # 3) Among candidates, pick the smallest global index per event
    N   = scores.numel()
    idx = torch.arange(N, device=device)

    big = torch.tensor(N, device=device, dtype=idx.dtype)  # sentinel
    idx_masked = torch.where(is_candidate, idx, big)

    lead_idx = torch.full((num_unique_events,), big,
                          device=device, dtype=idx.dtype)
    lead_idx.scatter_reduce_(
        dim=0,
        index=inv,
        src=idx_masked,
        reduce="amin",        # smallest index among candidates
        include_self=True,
    )

    # 4) Gather leading-vertex quantities (exactly one per event)
    logit1_lead = logit1[lead_idx]
    logit2_lead = logit2[lead_idx]
    y_lead      = y_true[lead_idx]

    return logit1_lead, logit2_lead, y_lead




class ABCLagrangian2_EventLevel(nn.Module):
    def __init__(self, eps_closure, eps_disco, k, alpha_lr=1e-3, numeric_eps=1e-6):
        super().__init__()
        self.eps_closure = float(eps_closure)
        self.eps_disco   = float(eps_disco)
        self.k = float(k)
        self.alpha_lr = float(alpha_lr)
        self.numeric_eps = float(numeric_eps)

        self.register_buffer("alpha_closure", 0.1 * torch.ones((), dtype=torch.float32))
        self.register_buffer("alpha_disco",   0.1 * torch.ones((), dtype=torch.float32))
        self._delta_closure = None
        self._delta_disco   = None

    def forward(self, logit1, logit2, y_true, event_idx, b1, b2):
        """
        Inputs:
            logit1, logit2: [N, 1]
            y_true:         [N, 1]
            event_idx:      [N, 1] 
        """
        # Ensure y matches logit dtype
        y = y_true.to(dtype=logit1.dtype)

        # --- Part A: Vertex-Level BCE Loss ---
        # Calculated on ALL vertices (shape [N, 1])
        loss_bce = torch.sqrt(
              (F.binary_cross_entropy_with_logits(logit1, y)**2 +
               F.binary_cross_entropy_with_logits(logit2, y)**2)/2
        )

        # --- Part B: Event-Level Selection ---
        # This reduces data from N vertices -> M events
        l1_lead, l2_lead, y_lead = select_leading_vertices(logit1, logit2, event_idx, y_true)

        # --- Part C: Event-Level Closure & Disco ---
        # Calculated ONLY on the leading vertices (shape [M, 1])
        
        s1_lead = torch.sigmoid(l1_lead)
        s2_lead = torch.sigmoid(l2_lead)
        # Flatten y_lead for boolean indexing: [M, 1] -> [M]
        bkg_mask_lead = (y_lead.view(-1) == 0) 

        # 1. Distance Correlation (Disco)
        if s1_lead.numel() >= 2:
            disco_all = distance_correlation(logit1, logit2)
        else:
            disco_all = logit1.new_zeros(())

        # 2. Closure Loss
        # Ensure thresholds match device/dtype
        b1 = torch.as_tensor(b1, device=s1_lead.device, dtype=s1_lead.dtype)
        b2 = torch.as_tensor(b2, device=s1_lead.device, dtype=s1_lead.dtype)
        
        # logit_b1 = torch.special.logit(b1)
        # logit_b2 = torch.special.logit(b2)

        # Soft binning (sigmoid approximation of step function)
        H1 = torch.sigmoid(self.k * (s1_lead - b1))
        H2 = torch.sigmoid(self.k * (s2_lead - b2))

        # Regions A, B, C, D
        A = H1 * H2
        B = (1 - H1) * H2
        C = H1 * (1 - H2)
        D = (1 - H1) * (1 - H2)

        # ORIGINAL IMPLEMENTATION
        # ------------------------
        # # Sum over BACKGROUND events only
        # # We use .view(-1) on regions to match bkg_mask_lead shape
        # NA_b = A.view(-1)[bkg_mask_lead].sum()
        # NB_b = B.view(-1)[bkg_mask_lead].sum()
        # NC_b = C.view(-1)[bkg_mask_lead].sum()
        # ND_b = D.view(-1)[bkg_mask_lead].sum()
        #
        # nom   = NA_b * ND_b - NB_b * NC_b
        # denom = NA_b * ND_b + NB_b * NC_b + self.numeric_eps
        # loss_closure = (nom / denom) ** 2

        # TEST -1 sigma closure loss
        # Sum over BACKGROUND events only
        NA_b = A.view(-1)[bkg_mask_lead].sum()
        NB_b = B.view(-1)[bkg_mask_lead].sum()
        NC_b = C.view(-1)[bkg_mask_lead].sum()
        ND_b = D.view(-1)[bkg_mask_lead].sum()

        # --- 1) Nominal closure f ----------------------------------------------
        num   = NA_b * ND_b - NB_b * NC_b
        den   = NA_b * ND_b + NB_b * NC_b + self.numeric_eps  # avoid 0
        f_nom = num / den

        # --- 2) Analytic derivatives df/dN_i -----------------------------------
        den2 = den * den + self.numeric_eps  # (NA ND + NB NC)^2

        df_dNA =  2.0 * NB_b * NC_b * ND_b / den2
        df_dNB = -2.0 * NA_b * NC_b * ND_b / den2
        df_dNC = -2.0 * NA_b * NB_b * ND_b / den2
        df_dND =  2.0 * NA_b * NB_b * NC_b / den2

        # --- 3) Poisson variances: Var(N_i) = N_i ------------------------------
        # clamp to avoid negative / zero issues
        NA_var = NA_b.clamp_min(self.numeric_eps)
        NB_var = NB_b.clamp_min(self.numeric_eps)
        NC_var = NC_b.clamp_min(self.numeric_eps)
        ND_var = ND_b.clamp_min(self.numeric_eps)

        sigma_f_sq = (
            (df_dNA * df_dNA) * NA_var +
            (df_dNB * df_dNB) * NB_var +
            (df_dNC * df_dNC) * NC_var +
            (df_dND * df_dND) * ND_var
        )

        sigma_f = torch.sqrt(sigma_f_sq + self.numeric_eps)

        

        # --- 4) -1σ closure and loss -------------------------------------------
        f_minus1sigma = torch.clamp(f_nom - sigma_f, min=0.0)
        # loss_closure  = f_minus1sigma * f_minus1sigma
        loss_closure  = f_nom * f_nom
        # print('-'*40)
        # print('NA_b: ', NA_b.item())
        # print('NB_b: ', NB_b.item())
        # print('NC_b: ', NC_b.item())
        # print('ND_b: ', ND_b.item())
        # print()

        # print('f: ', f_nom.item())
        # print('sigma_f: ', sigma_f.item())
        # print('f_minus1sigma: ', f_minus1sigma.item())
        # print('loss_closure: ', loss_closure.item())



        # 3. Lagrangian constraints
        delta_closure = loss_closure - self.eps_closure
        delta_disco   = disco_all    - self.eps_disco

        # Total Loss
        loss = loss_bce + self.alpha_closure * delta_closure + self.alpha_disco * delta_disco

        # Store for dual ascent
        self._delta_closure = delta_closure.detach()
        self._delta_disco   = delta_disco.detach()

        return loss, {
            'loss':             float(loss.detach()),
            'loss_bce':         float(loss_bce.detach()),
            'disco_all':        float(disco_all.detach()),
            'loss_closure':     float(loss_closure.detach()),
            'alpha_closure':    float(self.alpha_closure.detach()),
            'alpha_disco':      float(self.alpha_disco.detach()),
        }

    @torch.no_grad()
    def dual_ascent(self):
        if self._delta_closure is None or self._delta_disco is None:
            return
        self.alpha_closure.add_(self.alpha_lr * self._delta_closure).clamp_(min=0.0)
        self.alpha_disco.add_(self.alpha_lr * self._delta_disco).clamp_(min=0.0)



class ABCLagrangian2(nn.Module):
  def __init__(self, eps_closure, eps_disco, k, alpha_lr=1e-3, numeric_eps=1e-6):
    super().__init__()
    self.eps_closure = float(eps_closure)
    self.eps_disco   = float(eps_disco)
    self.k = float(k)
    self.alpha_lr = float(alpha_lr)
    self.numeric_eps = float(numeric_eps)

    self.register_buffer("alpha_closure", torch.zeros((), dtype=torch.float32))
    self.register_buffer("alpha_disco",   torch.zeros((), dtype=torch.float32))
    self._delta_closure = None
    self._delta_disco   = None

  def forward(self, logit1, logit2, y_true, b1, b2):
    y = y_true.to(dtype=logit1.dtype)

    loss_bce = torch.sqrt(
          (F.binary_cross_entropy_with_logits(logit1, y)**2 +
           F.binary_cross_entropy_with_logits(logit2, y)**2)/2
    )

    s1 = torch.sigmoid(logit1)
    s2 = torch.sigmoid(logit2)
    bkg = (y_true == 0)

    if s1.numel() >= 2:
        # disco_all = distance_correlation(s1, s2)
        disco_all = distance_correlation(logit1, logit2)
    else:
        disco_all = logit1.new_zeros(())

    

    # allow scalar or per-sample thresholds; broadcasting will just work
    b1 = torch.as_tensor(b1, device=s1.device, dtype=s1.dtype)
    b2 = torch.as_tensor(b2, device=s2.device, dtype=s2.dtype)

    logit_b1 = torch.special.logit(b1)
    logit_b2 = torch.special.logit(b2)

    H1 = torch.sigmoid(self.k * (logit1 - logit_b1))
    H2 = torch.sigmoid(self.k * (logit2 - logit_b2))

    A = H1 * H2
    B = (1 - H1) * H2
    C = H1 * (1 - H2)
    D = (1 - H1) * (1 - H2)

    NA_b = A[bkg].sum()
    NB_b = B[bkg].sum()
    NC_b = C[bkg].sum()
    ND_b = D[bkg].sum()

    nom   = NA_b * ND_b - NB_b * NC_b
    denom = NA_b * ND_b + NB_b * NC_b + self.numeric_eps
    loss_closure = (nom / denom) ** 2

    delta_closure = loss_closure - self.eps_closure
    delta_disco   = disco_all    - self.eps_disco   # single constraint, no special factor

    loss = loss_bce + self.alpha_closure * delta_closure + self.alpha_disco * delta_disco

    self._delta_closure = delta_closure.detach()
    self._delta_disco   = delta_disco.detach()
    return loss, {
    'loss':             float(loss.detach()),
    'loss_bce':         float(loss_bce.detach()),
    'disco_all':        float(disco_all.detach()),
    'loss_closure':     float(loss_closure.detach()),
    'alpha_closure':    float(self.alpha_closure.detach()),
    'alpha_disco':      float(self.alpha_disco.detach()),
    }

  @torch.no_grad()
  def dual_ascent(self):
    if self._delta_closure is None or self._delta_disco is None:
      return
    self.alpha_closure.add_(self.alpha_lr * self._delta_closure).clamp_(min=0.0)
    self.alpha_disco.add_(self.alpha_lr * self._delta_disco).clamp_(min=0.0)





# ------------ no closure version ------------


class ABCLagrangian_nocl(nn.Module):
  def __init__(self, eps_disco, k, alpha_lr=1e-3, numeric_eps=1e-6):
    super().__init__()
    self.eps_disco   = float(eps_disco)
    self.k = float(k)
    self.alpha_lr = float(alpha_lr)
    self.numeric_eps = float(numeric_eps)

    self.register_buffer("alpha_closure", torch.zeros((), dtype=torch.float32))
    self.register_buffer("alpha_disco",   torch.zeros((), dtype=torch.float32))
    self._delta_disco   = None

  def forward(self, logit1, logit2, y_true, b1, b2):
    y = y_true.to(dtype=logit1.dtype)

    loss_bce = torch.sqrt(
          (F.binary_cross_entropy_with_logits(logit1, y)**2 +
           F.binary_cross_entropy_with_logits(logit2, y)**2)/2
    )

    s1 = torch.sigmoid(logit1)
    s2 = torch.sigmoid(logit2)
    bkg = (y_true == 0)

    if s1.numel() >= 2:
        # disco_all = distance_correlation(s1, s2)
        disco_all = distance_correlation(logit1, logit2)
    else:
        disco_all = logit1.new_zeros(())

    

    # allow scalar or per-sample thresholds; broadcasting will just work
    b1 = torch.as_tensor(b1, device=s1.device, dtype=s1.dtype)
    b2 = torch.as_tensor(b2, device=s2.device, dtype=s2.dtype)

    logit_b1 = torch.special.logit(b1)
    logit_b2 = torch.special.logit(b2)

    H1 = torch.sigmoid(self.k * (logit1 - logit_b1))
    H2 = torch.sigmoid(self.k * (logit2 - logit_b2))

    A = H1 * H2
    B = (1 - H1) * H2
    C = H1 * (1 - H2)
    D = (1 - H1) * (1 - H2)

    NA_b = A[bkg].sum()
    NB_b = B[bkg].sum()
    NC_b = C[bkg].sum()
    ND_b = D[bkg].sum()

    nom   = NA_b * ND_b - NB_b * NC_b
    denom = NA_b * ND_b + NB_b * NC_b + self.numeric_eps
    loss_closure = (nom / denom) ** 2

    delta_disco   = disco_all    - self.eps_disco   # single constraint, no special factor

    loss = loss_bce + self.alpha_disco * delta_disco

    self._delta_disco   = delta_disco.detach()
    return loss, {
    'loss':             float(loss.detach()),
    'loss_bce':         float(loss_bce.detach()),
    'disco_all':        float(disco_all.detach()),
    'loss_closure':     float(loss_closure.detach()),
    'alpha_disco':      float(self.alpha_disco.detach()),
    }

  @torch.no_grad()
  def dual_ascent(self):
    if self._delta_disco is None: return
    self.alpha_disco.add_(self.alpha_lr * self._delta_disco).clamp_(min=0.0)