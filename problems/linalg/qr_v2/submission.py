import torch
from task import input_t, output_t

# ---------------------------------------------------------------------------
# Force true IEEE FP32 matmuls. On Blackwell, allowing TF32 for the trailing
# WY GEMMs would silently drop ~10 mantissa bits and blow the orthogonality
# gate at batch=640 (the check is .amax() over the whole batch). Keep this.
# ---------------------------------------------------------------------------
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    pass
try:  # newer PyTorch precision API; ignore if absent
    torch.backends.cuda.matmul.fp32_precision = "ieee"
except Exception:
    pass

# ---------------------------------------------------------------------------
# Tunables (iterate on these via Modal, NOT via leaderboard submissions).
# ---------------------------------------------------------------------------
_PANEL_NB = 16          # panel width (must be a power of 2 for the Triton tile)
_PANEL_WARPS = 8        # num_warps for the panel kernel
_BLOCKED_MIN_N = 176    # only route n >= this to the blocked Triton path
_TRAIL_FP64 = False     # True => do the trailing WY GEMMs in FP64 (safest,
                        # slower); False => FP32 trailing with an FP64-formed T
                        # (recommended: fast + accurate). Flip to True if the
                        # benchmark (batch=640) ever fails orthogonality.

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # pragma: no cover - Triton always present on the eval box
    _HAVE_TRITON = False


def custom_kernel(data: input_t) -> output_t:
    # Shape-based dispatch (NOT conditioning-based -- every path is exact for
    # all inputs). torch.geqrf serializes over the batch, so it only wins when
    # there are few matrices or they are tiny. For large launch-bound shapes the
    # blocked Householder path collapses ~15*n tiny kernel launches into
    # ~8*(n/nb) launches (one fused Triton panel kernel + a handful of batched
    # GEMMs per panel), which is the whole point of this submission.
    Bsz, n, _ = data.shape
    if n <= 64 or Bsz <= 16:
        return torch.geqrf(data)
    if _HAVE_TRITON and n >= _BLOCKED_MIN_N:
        try:
            return _blocked_householder(data, _PANEL_NB)
        except Exception:
            # Never let a kernel hiccup take down a submission: fall back to the
            # proven unblocked path. Numerical errors do NOT raise -- they show
            # up as residual failures in the checker -- so this only catches
            # genuine launch/compile faults, not silent wrong answers.
            pass
    return _batched_householder(data)


# ===========================================================================
# Blocked (WY) Householder QR: fused Triton panel kernel + batched-GEMM update.
# ===========================================================================

if _HAVE_TRITON:

    @triton.jit
    def _panel_kernel(
        A_ptr, tau_ptr,
        n, k, kb,
        stride_ab, stride_ar, stride_ac,
        stride_tb,
        BLOCK_M: tl.constexpr, BLOCK_KB: tl.constexpr,
    ):
        # One program per matrix. Factor the panel columns [k, k+kb) with the
        # unblocked Householder algorithm, updating ONLY the within-panel
        # trailing columns. The big trailing update (columns >= k+kb) is done
        # outside in PyTorch via the WY representation. This is exactly
        # `panel_factor` from the validated numpy reference, vectorized.
        pid = tl.program_id(0)
        A_b = A_ptr + pid * stride_ab

        r = tl.arange(0, BLOCK_M)            # submatrix row index (abs row = k+r)
        c = tl.arange(0, BLOCK_KB)           # panel col index   (abs col = k+c)
        row_abs = k + r
        col_abs = k + c
        rmask = row_abs < n
        cmask = c < kb

        ptrs = A_b + row_abs[:, None] * stride_ar + col_abs[None, :] * stride_ac
        P = tl.load(ptrs, mask=rmask[:, None] & cmask[None, :], other=0.0)

        tau_acc = tl.zeros((BLOCK_KB,), dtype=tl.float32)

        for i in range(BLOCK_KB):
            colsel = c == i                                  # (BLOCK_KB,)
            xi = tl.sum(tl.where(colsel[None, :], P, 0.0), axis=1)   # (BLOCK_M,)
            actr = (r >= i) & rmask
            xa = tl.where(actr, xi, 0.0)                     # active part of col i

            xnorm_sq = tl.sum(xa * xa, axis=0)
            alpha = tl.sum(tl.where(r == i, xi, 0.0), axis=0)
            normx = tl.sqrt(xnorm_sq)
            reflect = normx > 0.0
            # beta = -sign(alpha) * normx  (== copysign(normx, -alpha))
            s = tl.where(alpha >= 0.0, 1.0, -1.0)
            beta = -s * normx
            denom = tl.where(reflect, alpha - beta, 1.0)
            safe_beta = tl.where(reflect, beta, 1.0)
            tau_i = tl.where(reflect, (beta - alpha) / safe_beta, 0.0)

            v = xa / denom
            v = tl.where(reflect & actr, v, 0.0)
            v = tl.where(r == i, 1.0, v)                     # unit Householder head

            # within-panel rank-1 update of columns j > i:  P[:,j] -= v*(tau*(v.P[:,j]))
            vP = tl.sum(v[:, None] * P, axis=0)              # (BLOCK_KB,)
            w = tau_i * vP
            updmask = (c > i)[None, :]
            P = P - (v[:, None] * w[None, :]) * updmask

            # write column i: keep R above the head, beta on the diagonal,
            # Householder entries below it.
            diag_i = tl.where(reflect, beta, alpha)
            vbelow = tl.where(actr & (r > i), v, 0.0)
            newcol = tl.where(r == i, diag_i, vbelow)
            newcol = tl.where(r < i, xi, newcol)             # preserve R rows
            P = tl.where(colsel[None, :], newcol[:, None], P)

            tau_acc = tl.where(c == i, tau_i, tau_acc)

        tl.store(ptrs, P, mask=rmask[:, None] & cmask[None, :])
        tl.store(tau_ptr + pid * stride_tb + col_abs, tau_acc, mask=cmask)


def _blocked_householder(data: input_t, nb: int) -> output_t:
    A = data.clone()                                  # (B, n, n) FP32, in place
    Bsz, n, _ = A.shape
    dev = A.device
    tau = torch.zeros((Bsz, n), device=dev, dtype=torch.float32)

    BLOCK_M = triton.next_power_of_2(n)
    BLOCK_KB = nb
    eye = torch.eye(nb, device=dev, dtype=torch.float32)

    for k in range(0, n, nb):
        kb = min(nb, n - k)
        _panel_kernel[(Bsz,)](
            A, tau,
            n, k, kb,
            A.stride(0), A.stride(1), A.stride(2),
            tau.stride(0),
            BLOCK_M=BLOCK_M, BLOCK_KB=BLOCK_KB,
            num_warps=_PANEL_WARPS,
        )

        if k + kb < n:
            # ---- build V: unit lower-trapezoidal, identity-reflector cols zeroed
            Vraw = A[:, k:, k:k + kb]                  # (B, m, kb) FP32 view
            V = Vraw.clone()
            V[:, :kb, :] = torch.tril(Vraw[:, :kb, :], -1) + eye[:kb, :kb]
            tk = tau[:, k:k + kb]                       # (B, kb)
            nz = tk != 0.0
            V = V * nz[:, None, :].to(V.dtype)

            # ---- closed-form block reflector T = inv(triu(VᵀV,1) + diag(1/tau))
            # formed in FP64 (tiny kb×kb): FP32 inv(G) is what blew up before.
            V64 = V.double()
            S = V64.transpose(1, 2) @ V64              # (B, kb, kb)
            tk64 = tk.double()
            inv_tau = torch.where(nz, 1.0 / torch.where(nz, tk64, torch.ones_like(tk64)),
                                  torch.ones_like(tk64))
            G = torch.triu(S, 1) + torch.diag_embed(inv_tau)
            T = torch.linalg.inv(G)                     # (B, kb, kb) FP64

            # ---- trailing update:  C <- C - V @ (Tᵀ @ (Vᵀ @ C))
            if _TRAIL_FP64:
                C = A[:, k:, k + kb:].double()
                W = T.transpose(1, 2) @ (V64.transpose(1, 2) @ C)
                A[:, k:, k + kb:] = (C - V64 @ W).to(torch.float32)
            else:
                Vf = V                                  # FP32 (already zeroed)
                Tf = T.to(torch.float32)
                C = A[:, k:, k + kb:]                   # FP32 view
                W = Tf.transpose(1, 2) @ (Vf.transpose(1, 2) @ C)
                A[:, k:, k + kb:] = C - Vf @ W

    return A, tau


# ===========================================================================
# Proven unblocked path (fallback + small-n route). Identical to commit c340815.
# ===========================================================================

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
