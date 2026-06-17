import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    # Shape-based dispatch (NOT conditioning-based -- both paths are exact for
    # all inputs). torch.geqrf serializes over the batch, so it only wins when
    # there are few matrices or they are tiny; the batched Householder path wins
    # whenever the batch is large enough to hide its per-column launch overhead.
    Bsz, n, _ = data.shape
    if n <= 64 or Bsz <= 16:
        return torch.geqrf(data)
    return _batched_householder(data)


def _batched_householder(data: input_t) -> output_t:
    # Batched unblocked Householder QR, vectorized across the batch so all
    # matrices advance together (vs torch.geqrf, which serializes per-matrix).
    # Matches LAPACK geqrf / dlarfg conventions so triu(H)=R and
    # householder_product(H, tau)=Q reconstruct the factorization exactly.
    A = data.clone()                      # (B, n, n); factored in place
    Bsz, n, _ = A.shape
    dev, dt = A.device, A.dtype
    tau = torch.zeros((Bsz, n), device=dev, dtype=dt)
    one = torch.ones((), device=dev, dtype=dt)

    for j in range(n):
        x = A[:, j:, j]                                  # (B, m), m = n-j
        alpha = x[:, 0]                                  # (B,)
        xnorm_sq = (x * x).sum(dim=1)                    # ||x||^2
        normx = torch.sqrt(xnorm_sq)
        beta = torch.copysign(normx, -alpha)             # R diagonal (no cancellation)

        active = normx > 0.0                             # zero column => identity reflector
        denom = torch.where(active, alpha - beta, one)   # safe divisor
        tau_j = torch.where(active, -denom / beta, alpha)

        v = x / denom.unsqueeze(1)
        v[:, 0] = 1.0                                     # unit Householder head

        A[:, j, j] = beta                                # R[j,j]
        tau[:, j] = tau_j
        if j + 1 < n:
            A[:, j + 1:, j] = v[:, 1:]                    # store v below diagonal
            sub = A[:, j:, j + 1:]                        # (B, m, n-j-1) trailing block
            w = (v.unsqueeze(1) @ sub) * tau_j.view(Bsz, 1, 1)   # (B, 1, n-j-1)
            sub.baddbmm_(v.unsqueeze(2), w, beta=1.0, alpha=-1.0)

    return A, tau
