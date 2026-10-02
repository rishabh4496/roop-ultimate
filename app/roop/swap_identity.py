"""The one gate every identity vector passes before it steers a swap.

Every embedding-based swapper (inswapper family, hyperswap, the crossface-converted models)
consumes the 512-d buffalo_l / w600k_r50 ArcFace identity that insightface stores on
``face.embedding``. That contract is deliberately separate from the recognition backend a user
can pick for TRACKING (``face_analyser.set_recognition_model`` / ``IdentityBank``): tracking
may use any registered model, the swapper never does.

Why a shape check alone cannot enforce it. AdaFace and Glint-R100 are ALSO 512-d, so a vector
from the wrong model passes any ``(1, 512)`` test and quietly steers the swap toward a point
in a different embedding space. What this module can and does reject:

  * any vector that is not exactly 512 values (a 128-d SFace vector, a stacked batch);
  * non-finite values;
  * a zero vector -- which used to become a zero latent that the cache then kept, a swap
    toward nobody that reported success.

The 512-d-but-wrong-model case is closed STRUCTURALLY instead: the recognition API is not
imported by any swap or production path, and tests/test_swap_identity_isolation.py fails if
that changes. Wiring it in must be a decision, not an import.
"""

import numpy as np

SWAP_EMBEDDING_DIM = 512
_MIN_NORM = 1e-6


def validate_identity_embedding(embedding, source: str = "source face") -> np.ndarray:
    """Return `embedding` as a C-contiguous float32 (1, 512) array, or raise ValueError.

    `source` names the caller in the message so a bad vector can be traced to its face.
    """
    if embedding is None:
        raise ValueError("%s: no identity embedding (None)" % source)
    try:
        arr = np.asarray(embedding, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s: identity embedding is not numeric (%s)" % (source, exc)) from None
    if arr.size != SWAP_EMBEDDING_DIM:
        raise ValueError(
            "%s: identity embedding has %d values; the swappers consume the %d-d buffalo_l/w600k_r50 "
            "vector. A different recogniser's output (e.g. 128-d SFace) must never reach the swapper."
            % (source, arr.size, SWAP_EMBEDDING_DIM))
    if not np.isfinite(arr).all():
        raise ValueError("%s: identity embedding contains NaN/Inf" % source)
    if float(np.linalg.norm(arr)) <= _MIN_NORM:
        raise ValueError(
            "%s: identity embedding is a zero vector; swapping toward it would swap toward nobody "
            "while reporting success" % source)
    return np.ascontiguousarray(arr.reshape(1, SWAP_EMBEDDING_DIM))
