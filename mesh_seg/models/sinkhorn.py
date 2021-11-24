import torch


def perform_log_sinkhorn_iterations(log_assignment_matrix, log_mu, log_nu, num_iterations: int):
    """Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(num_iterations):
        u = log_mu - torch.logsumexp(log_assignment_matrix + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(log_assignment_matrix + u.unsqueeze(2), dim=1)
    return log_assignment_matrix + u.unsqueeze(2) + v.unsqueeze(1)


def solve_log_optimal_transport(assignment_matrix, num_iterations: int):
    """Perform Differentiable Optimal Transport in Log-space for stability"""
    m, n = assignment_matrix.shape
    scalar_one = torch.tensor(1, dtype=assignment_matrix.dtype)
    ms, ns = (m * scalar_one).to(assignment_matrix), (n * scalar_one).to(assignment_matrix)

    norm = - (ms + ns).log()
    log_mu = norm.expand(m)
    log_nu = norm.expand(n)
    log_mu, log_nu = log_mu[None].expand(1, -1), log_nu[None].expand(1, -1)

    log_assignment_matrix = perform_log_sinkhorn_iterations(
        assignment_matrix.unsqueeze(0),
        log_mu,
        log_nu,
        num_iterations,
    )
    log_assignment_matrix = log_assignment_matrix - norm
    return log_assignment_matrix.squeeze(0)
