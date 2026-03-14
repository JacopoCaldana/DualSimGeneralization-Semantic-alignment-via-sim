""""""

import torch
from sklearn.cluster import KMeans


def ppfe(
    X: torch.tensor,
    Y: torch.tensor,
    output_real: torch.tensor,
    n_clusters: int,
    n_proto: int,
    seed: int = None,
) -> torch.tensor:
    """ """
    X = X.H
    Y = Y.H

    data = output_real.detach().cpu().resolve_conj().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    labels = torch.tensor(kmeans.fit_predict(data))

    n_input = []
    n_output = []

    for label in labels.unique():
        mask = labels == label

        # Retrieve the obs for each label
        inps = X[mask]
        outs = Y[mask]

        # Get only some obs to calculate the proto
        iidx = torch.randperm(inps.size(0))[:n_proto]

        # Calculate the mean
        n_input.append(inps[iidx].mean(dim=0))
        n_output.append(outs[iidx].mean(dim=0))

    # Stack all of them in a tensor
    X = torch.stack(n_input)
    Y = torch.stack(n_output)

    # Shuffle
    idx = torch.randperm(X.size(0))
    X = X[idx]
    Y = Y[idx]

    # Encoder Frame
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    F = U @ Vt

    # Decoder Frame
    U, S, Vt = torch.linalg.svd(Y, full_matrices=False)
    G = (U @ Vt).H

    return G @ F


def ridge_regression(
    X: torch.tensor,
    Y: torch.tensor,
    weights: torch.Tensor | None = None,
    lmb: float = 0.0,
) -> torch.tensor:
    """A function to solve the ridge regression.

    Args:
        X : torch.tensor
            A complex matrix.
        Y : torch.tensor
            A complex matrix.
        weights : torch.tensor | None
            Non-negative sample weights.
            If None, uses uniform weights (identity matrix).
        lmb : float
            The multiplier. Default 0.0.

    Returns:
        A : torch.tensor
            The solution of the ridge regression.
    """

    d, n = X.shape
    reg = lmb * torch.eye(d, dtype=X.dtype, device=X.device)

    if weights is not None:
        W = torch.diag(weights.to(X.device, dtype=X.dtype))

        # A @ X @ X.H + lmb * torch.linalg.inv(W) @ A = Y @ X.H
        A = torch.linalg.solve(X @ W @ X.H + reg, Y @ W @ X.H, left=False)
    else:
        # A = Y @ X.H @ torch.linalg.inv(X @ X.H + reg)
        A = torch.linalg.solve(X @ X.H + reg, Y @ X.H, left=False)

    return A


if __name__ == '__main__':
    pass
