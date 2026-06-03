"""Integer share reconciliation helpers for schedule builders."""

import numpy as np


def reconcile_integer_shares(raw_shares: np.ndarray, target_total: int) -> np.ndarray:
    """Adjust rounded share allocations so they sum exactly to ``target_total``.

    Parameters
    ----------
    raw_shares
        Integer share allocations before final reconciliation.
    target_total
        Required parent-order share count.

    Returns
    -------
    np.ndarray
        Share array whose elements sum to ``target_total``. Any rounding residual
        is applied to the final slice so earlier buckets stay unchanged.
    """
    shares = np.array(raw_shares, dtype=int, copy=True)
    share_difference = int(target_total) - int(shares.sum())

    if share_difference != 0:
        shares[-1] = int(shares[-1]) + share_difference

    return shares
