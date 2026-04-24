from scipy.stats import kendalltau as kt
from scipy.stats import circmean as sscircmean
from scipy.stats import circstd as sscircstd
from scipy.stats import rankdata as _rankdata
import numpy as np

try:
    import numba as _numba
    _NUMBA = True
except ImportError:
    _NUMBA = False

if _NUMBA:
    @_numba.njit(cache=True)
    def _batch_tau_nb(kkey_arr, ref_ranks):
        """Tau-b of kkey_arr (T,) against every row of ref_ranks (N, T) → (N,)."""
        N = ref_ranks.shape[0]
        T = ref_ranks.shape[1]
        nx = 0
        for j in range(T):
            for k in range(j + 1, T):
                if kkey_arr[j] != kkey_arr[k]:
                    nx += 1
        taus = np.empty(N, dtype=np.float64)
        for i in range(N):
            c = 0
            d = 0
            ny_i = 0
            for j in range(T):
                for k in range(j + 1, T):
                    dy = ref_ranks[i, j] - ref_ranks[i, k]
                    if dy != 0:
                        ny_i += 1
                    dx = kkey_arr[j] - kkey_arr[k]
                    if dx != 0 and dy != 0:
                        if (dx > 0) == (dy > 0):
                            c += 1
                        else:
                            d += 1
            denom = nx * ny_i
            taus[i] = (c - d) / (denom ** 0.5) if denom > 0 else 0.0
        return taus


def _batch_tau_numpy(kkey_arr, ref_ranks):
    """Vectorized tau-b fallback: kkey_arr (T,) vs ref_ranks (N, T) → (N,)."""
    sx = np.sign(kkey_arr[:, None] - kkey_arr[None, :])              # (T, T)
    sy = np.sign(ref_ranks[:, :, None] - ref_ranks[:, None, :])       # (N, T, T)
    num = np.einsum('ij,nij->n', sx, sy) / 2.0                        # (N,)
    nx = float(np.count_nonzero(sx)) / 2.0
    ny = np.count_nonzero(sy.reshape(len(ref_ranks), -1), axis=1).astype(float) / 2.0
    denom = np.sqrt(nx * ny)
    return np.where(denom > 0.0, num / denom, 0.0)


def rank_references(dref, triples):
    """Pre-rank all reference waveforms as integer arrays for batch tau."""
    N = len(triples)
    T = len(next(iter(dref.values())))
    ref_ranks = np.empty((N, T), dtype=np.int64)
    for i, triple in enumerate(triples):
        key = (float(triple[0]), float(triple[1]), float(triple[2]))
        ref_ranks[i] = _rankdata(dref[key]).astype(np.int64)
    return ref_ranks


def get_stat_probs(dorder, new_header, triples, dref, ref_ranks, size):
    new_header_arr = np.array(new_header, dtype=float)
    periods = triples[:, 0]
    phases = triples[:, 1]
    widths = triples[:, 2]
    nadirs = (phases + widths) % periods

    _batch_tau = _batch_tau_nb if _NUMBA else _batch_tau_numpy

    d_taugene, d_pergene, d_phgene, d_nagene = {}, {}, {}, {}
    rs = []
    for kkey in dorder:
        kkey_arr = np.array(kkey, dtype=np.int64)
        maxloc = new_header_arr[np.argmax(kkey_arr)]
        minloc = new_header_arr[np.argmin(kkey_arr)]

        raw_taus = _batch_tau(kkey_arr, ref_ranks)
        taus = np.arctanh(np.clip(raw_taus, -0.99, 0.99))
        abs_taus = np.abs(taus)

        neg_mask = taus < 0
        phase_col = np.where(neg_mask, nadirs, phases)
        nadir_col = np.where(neg_mask, phases, nadirs)

        res = np.column_stack([
            abs_taus,
            np.zeros(len(triples)),
            periods,
            phase_col,
            nadir_col,
            np.full(len(triples), maxloc),
            np.full(len(triples), minloc),
        ])

        r = pick_best_match(res)
        d_taugene[r[0]] = d_taugene.get(r[0], 0) + dorder[kkey]
        d_pergene[r[2]] = d_pergene.get(r[2], 0) + dorder[kkey]
        d_phgene[r[3]] = d_phgene.get(r[3], 0) + dorder[kkey]
        d_nagene[r[4]] = d_nagene.get(r[4], 0) + dorder[kkey]
        count = int(np.round(size * dorder[kkey]))
        rs.extend([r] * count)
    rs = np.array(rs)
    m_tau = np.mean(rs[:, 0])
    s_tau = np.std(rs[:, 0])
    m_per = np.mean(rs[:, 2])
    s_per = np.std(rs[:, 2])
    m_ph = sscircmean(rs[:, 3], high=24, low=0)
    s_ph = sscircstd(rs[:, 3], high=24, low=0)
    m_na = sscircmean(rs[:, 4], high=24, low=0)
    s_na = sscircstd(rs[:, 4], high=24, low=0)
    return [m_per, s_per, m_ph, s_ph, m_na, s_na], [m_tau, s_tau], d_taugene, d_pergene, d_phgene, d_nagene


def generate_base_reference(header, waveform="cosine", period=24., phase=0., width=12.):
    ZTs = np.array(header, dtype=float)
    coef = 2.0 * np.pi / period
    w = (width * coef) % (2.0 * np.pi)
    tpoints = ((ZTs - phase) * coef) % (2.0 * np.pi)
    if waveform == 'cosine':
        return np.where(
            tpoints <= w,
            np.cos(tpoints / (w / np.pi)),
            np.cos((tpoints + 2.0 * (np.pi - w)) * np.pi / (2.0 * np.pi - w))
        )
    elif waveform == 'trough':
        return np.where(
            tpoints <= w,
            1.0 - tpoints / w,
            (tpoints - w) / (2.0 * np.pi - w)
        )
    elif waveform == 'impulse':
        d = np.minimum(tpoints, np.abs(2.0 * np.pi - tpoints))
        return np.maximum(-2.0 * d / (3.0 * np.pi / 4.0) + 1.0, 0.0)
    elif waveform == 'step':
        return np.where(tpoints < np.pi, 1.0, 0.0)


def farctanh(x):
    return float(np.arctanh(np.clip(x, -0.99, 0.99)))


def periodic(x):
    x = float(x)
    while x > 12:
        x -= 24.
    while x <= -12:
        x += 24.
    return x


def pick_best_match(res):
    res = np.array(res)
    taus = res[:, 0]
    tau_mask = (max(taus) == taus)
    if np.sum(tau_mask) == 1:
        return res[int(np.argmax(tau_mask))]

    res = res[tau_mask]
    phases = np.abs(res[:, 3] - res[:, 5])
    phasemask = (min(phases) == phases)
    if np.sum(phasemask) == 1:
        return res[int(np.argmax(phasemask))]

    res = res[phasemask]
    diffs = np.abs(res[:, 4] - res[:, 6])
    diffmask = (min(diffs) == diffs)
    if np.sum(diffmask) == 1:
        return res[int(np.argmax(diffmask))]

    return res[np.random.randint(len(res))]


def get_waveform_list(periods, phases, widths):
    triples = []
    for period in periods:
        seen = set()
        for phase in phases:
            for width in widths:
                nadir = (phase + width) % period
                # deduplicate waveforms where peak and nadir positions are swapped
                key = (float(nadir), float(phase))
                if key not in seen:
                    seen.add(key)
                    triples.append([float(period), float(phase), float(width)])
    return np.array(triples, dtype=float)


def make_references(new_header, triples, waveform='cosine'):
    dref = {}
    for triple in triples:
        period, phase, width = triple
        reference = generate_base_reference(new_header, waveform, period, phase, width)
        dref[(period, phase, width)] = reference
    return dref


def get_matches(kkey, triple, d_ref, new_header):
    reference = d_ref[tuple(triple)]
    period, phase, width = triple
    nadir = (phase + width) % period
    tau, p = kt(reference, kkey)
    p = p / 2.0
    tau = farctanh(tau)
    maxloc = new_header[kkey.index(max(kkey))]
    minloc = new_header[kkey.index(min(kkey))]
    r = [tau, p, period, phase, nadir, maxloc, minloc]
    if tau < 0:
        r = [abs(tau), p, period, nadir, phase, maxloc, minloc]
    return r
