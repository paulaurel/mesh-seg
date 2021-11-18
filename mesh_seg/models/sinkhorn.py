import torch


def log_sinkhorn_iterations(Z: torch.tensor, log_mu, log_nu, num_iterations: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(num_iterations):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def add_bins_to_score_matrix(score_matrix: torch.tensor, bin_score: torch.tensor) -> torch.tensor:
    batch_size, ref_size, query_size = score_matrix.shape
    row_bins = bin_score.expand(batch_size, ref_size, 1)
    col_bins = bin_score.expand(batch_size, 1, query_size)
    single_bin = bin_score.expand(batch_size, 1, 1)
    return torch.cat(
        [torch.cat([score_matrix, row_bins], dim=-1), torch.cat([col_bins, single_bin], dim=-1)],
        dim=1,
    )


def log_optimal_transport(score_matrix: torch.tensor, bin_score, num_iterations: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    batch_size, ref_size, query_size = score_matrix.shape
    scalar_one = torch.tensor(1, dtype=score_matrix.dtype)
    ms, ns = (ref_size * scalar_one).to(score_matrix), (query_size * scalar_one).to(score_matrix)

    assignment_matrix = add_bins_to_score_matrix(score_matrix, bin_score)
    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(ref_size), ns.log().expand(1) + norm]).unsqueeze(0)
    log_nu = torch.cat([norm.expand(query_size), ms.log().expand(1) + norm]).unsqueeze(0)
    log_mu, log_nu = log_mu.expand(batch_size, -1), log_nu.expand(batch_size, -1)

    Z = log_sinkhorn_iterations(assignment_matrix, log_mu, log_nu, num_iterations)
    Z = Z - norm  # multiply probabilities by M+N
    return Z