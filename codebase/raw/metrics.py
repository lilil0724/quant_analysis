"""Full-test-set feature and prediction metrics; no N-by-N sample matrices."""
import numpy as np


def _chunks(n, size):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def centered_linear_cka(x, y, chunk_size=1024):
    """Centered linear CKA via 768-by-768 feature cross-products."""
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] < 2:
        raise ValueError('CKA requires matched [N,D] arrays with N >= 2')
    n, d = x.shape
    sum_x = np.zeros(d, np.float64)
    sum_y = np.zeros(d, np.float64)
    xx = np.zeros((d, d), np.float64)
    yy = np.zeros((d, d), np.float64)
    xy = np.zeros((d, d), np.float64)
    for sl in _chunks(n, chunk_size):
        a = np.asarray(x[sl], dtype=np.float64)
        b = np.asarray(y[sl], dtype=np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('CKA input has non-finite features')
        sum_x += a.sum(axis=0)
        sum_y += b.sum(axis=0)
        xx += a.T @ a
        yy += b.T @ b
        xy += a.T @ b
    xx -= np.outer(sum_x, sum_x) / n
    yy -= np.outer(sum_y, sum_y) / n
    xy -= np.outer(sum_x, sum_y) / n
    denom = np.linalg.norm(xx) * np.linalg.norm(yy)
    if denom <= 0:
        return None
    return float(np.sum(xy * xy) / denom)


def class_equal_cosine_distances(x, labels, chunk_size=2048):
    """Pair-equal class distances, including distinct-pair intra-class mean."""
    labels = np.asarray(labels)
    if x.ndim != 2 or len(x) != len(labels):
        raise ValueError('Feature/label shape mismatch')
    classes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.zeros((len(classes), x.shape[1]), np.float64)
    for sl in _chunks(len(x), chunk_size):
        a = np.asarray(x[sl], dtype=np.float64)
        norms = np.linalg.norm(a, axis=1)
        if not np.isfinite(a).all():
            raise ValueError('Cosine distance input has nonfinite feature')
        if np.any(norms <= 0):
            return {'d_intra': None, 'd_inter': None, 'separability': None,
                    'n_intra_classes': 0, 'n_inter_class_pairs': 0}
        np.add.at(sums, inverse[sl], a / norms[:, None])
    valid = counts >= 2
    intra = None
    if valid.any():
        cos = (np.sum(sums[valid] ** 2, axis=1) - counts[valid]) / (
            counts[valid] * (counts[valid] - 1))
        intra = float(np.mean(1 - cos))
    inter = None
    if len(classes) >= 2:
        means = sums / counts[:, None]
        pair_cos = means @ means.T
        inter = float(np.mean(1 - pair_cos[np.triu_indices(len(classes), k=1)]))
    return {'d_intra': intra, 'd_inter': inter,
            'separability': None if intra is None or inter is None else inter - intra,
            'n_intra_classes': int(valid.sum()), 'n_inter_class_pairs': len(classes) * (len(classes) - 1) // 2}


def norm_quantiles(x, chunk_size=2048):
    values = np.empty(len(x), np.float64)
    for sl in _chunks(len(x), chunk_size):
        values[sl] = np.linalg.norm(np.asarray(x[sl], dtype=np.float64), axis=1)
    p50, p95, p99 = np.quantile(values, (0.5, 0.95, 0.99))
    return {'norm_p50': float(p50), 'norm_p95': float(p95), 'norm_p99': float(p99),
            'norm_p99_p50': None if p50 <= 0 else float(p99 / p50)}


def regularized_h_score(x, labels, ridge_fraction=0.01):
    """tr(S_b (S_w + lambda I)^-1), with lambda tied to within scatter."""
    a = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    if a.ndim != 2 or len(a) != len(labels) or len(np.unique(labels)) < 2:
        return None
    global_mean = a.mean(axis=0)
    between = np.zeros((a.shape[1], a.shape[1]), np.float64)
    within = np.zeros_like(between)
    for label in np.unique(labels):
        part = a[labels == label]
        center = part.mean(axis=0)
        delta = center - global_mean
        between += len(part) * np.outer(delta, delta)
        residual = part - center
        within += residual.T @ residual
    between /= len(a)
    within /= len(a)
    ridge = ridge_fraction * np.trace(within) / a.shape[1]
    if ridge <= 0:
        return None
    try:
        return float(np.trace(np.linalg.solve(within + ridge * np.eye(a.shape[1]), between)))
    except np.linalg.LinAlgError:
        return None


def prediction_metrics(fp, quant):
    if len(fp) != len(quant):
        raise ValueError('Prediction lengths differ')
    for left, right in zip(fp, quant):
        if left['sample_id'] != right['sample_id'] or left['target'] != right['target']:
            raise ValueError('Sample pairing mismatch')
    n = len(fp)
    fpc = np.fromiter((int(row['correct']) for row in fp), dtype=bool, count=n)
    qc = np.fromiter((int(row['correct']) for row in quant), dtype=bool, count=n)
    fpp = np.fromiter((int(row['prediction']) for row in fp), dtype=np.int64, count=n)
    qp = np.fromiter((int(row['prediction']) for row in quant), dtype=np.int64, count=n)
    fp_accuracy = float(fpc.mean())
    accuracy = float(qc.mean())
    swap = fpp != qp
    damage = int(np.count_nonzero(fpc & ~qc))
    rescue = int(np.count_nonzero(~fpc & qc))
    swaps = int(swap.sum())
    return {'n': n, 'fp_accuracy': fp_accuracy, 'accuracy': accuracy,
            'relative_loss': None if fp_accuracy == 0 else 1 - accuracy / fp_accuracy,
            'accuracy_drop': fp_accuracy - accuracy, 'swap_rate': swaps / n,
            'damage_count': damage, 'rescue_count': rescue,
            'damage_rate': damage / n, 'rescue_rate': rescue / n,
            'net_damage_r': None if swaps == 0 else (damage - rescue) / swaps}


def pca_fit(x, chunk_size=2048):
    n, d = x.shape
    if n < 2:
        raise ValueError('PCA requires two samples')
    total = np.zeros(d, np.float64)
    gram = np.zeros((d, d), np.float64)
    for sl in _chunks(n, chunk_size):
        a = np.asarray(x[sl], dtype=np.float64)
        total += a.sum(axis=0)
        gram += a.T @ a
    mean = total / n
    cov = (gram - n * np.outer(mean, mean)) / (n - 1)
    eig, vectors = np.linalg.eigh(cov)
    order = np.argsort(eig)[::-1][:2]
    components = vectors[:, order]
    explained = np.maximum(eig[order], 0) / max(float(np.maximum(eig, 0).sum()), 1e-30)
    return mean, components, explained


def pca_project(x, mean, components, chunk_size=2048):
    result = np.empty((len(x), 2), np.float32)
    for sl in _chunks(len(x), chunk_size):
        result[sl] = (np.asarray(x[sl], dtype=np.float64) - mean) @ components
    return result


def rank_average(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='stable')
    rank = np.empty(len(values), np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank[order[start:end]] = (start + end - 1) / 2
        start = end
    return rank


def spearman_description(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    return float(np.corrcoef(rank_average(x), rank_average(y))[0, 1])
