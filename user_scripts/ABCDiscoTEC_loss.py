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
    loss_cls = 0.5 * (
        F.binary_cross_entropy_with_logits(logit1, y) +
        F.binary_cross_entropy_with_logits(logit2, y)
    # )
    # loss_cls = torch.sqrt(
    #       (F.binary_cross_entropy_with_logits(logit1, y) +
    #       F.binary_cross_entropy_with_logits(logit2, y))/2
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
           F.binary_cross_entropy_with_logits(logit2, y)**2)/2)

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
    delta_disco_s   = disco_sig    - self.eps_disco * 4
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