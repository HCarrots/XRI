"""Joint nonnegative least squares and anchored relative analyzer gain fitting.
Optional TV regularization belongs to the inverse problem; no geometric postprocessing.
"""
from __future__ import annotations
import numpy as np

def _joint_terms(responses, measurements, *, gains=None, voxel_width_mm,
                 foil_thickness_mm):
    """Build reduced data terms without stacking a large detector operator."""
    if len(responses) != len(measurements) or not responses:
        raise ValueError("One response and data vector/volume per analyzer is required")
    gains = np.ones(len(responses)) if gains is None else np.asarray(gains, float)
    if gains.shape != (len(responses),) or not np.isfinite(gains).all() or np.any(gains <= 0):
        raise ValueError("Gains must be finite and positive")
    if voxel_width_mm <= 0 or foil_thickness_mm <= 0:
        raise ValueError("Physical widths must be positive")
    nx = np.asarray(responses[0]).shape[1]
    shape = np.asarray(measurements[0]).shape[1:]
    hessian = np.zeros((nx, nx))
    rhs = np.zeros((nx, *shape))
    operators, data = [], []
    for response, measured, gain in zip(responses, measurements, gains):
        a = np.asarray(response, float)*gain*voxel_width_mm/foil_thickness_mm
        d = np.asarray(measured, float)
        if a.ndim != 2 or a.shape[1] != nx or d.shape != (a.shape[0], *shape):
            raise ValueError("Expected response[p,x], data[p,...] with common transverse grid")
        if not np.isfinite(a).all() or not np.isfinite(d).all() or np.any(a < 0):
            raise ValueError("Operator must be finite/nonnegative; data finite")
        hessian += a.T@a
        rhs += (a.T@d.reshape(a.shape[0], -1)).reshape(rhs.shape)
        operators.append(a)
        data.append(d)
    return operators, data, hessian, rhs


def reconstruct_nnls(responses, measurements, **kwargs):
    """Baseline nonnegative joint reconstruction, independently for each column.

    Same energy processing, dose units, detector pixels and geometry are required
    for data and calibration. Result is relative Kapton scattering density;
    different composition/q/attenuation require an explicit physical correction.
    """
    operators, data, hessian, rhs = _joint_terms(responses, measurements, **kwargs)
    density = solve_gram_nnls(hessian, rhs)
    predicted = [(a@density.reshape(density.shape[0], -1)).reshape(d.shape) for a, d in zip(operators, data)]
    return density, predicted, [p-d for p, d in zip(predicted, data)]


def solve_gram_nnls(hessian, rhs, *, max_iterations=20000, tolerance=1e-7,
                    initial=None, return_info=False):
    """Batched accelerated projected-gradient NNLS with an explicit KKT check.

    Near-null detector columns make active-set NNLS unstable on the full X
    domain. Starting at zero (or the previous gain iteration) and using a
    projected-gradient stopping rule avoids division by tiny column norms.
    No column or voxel is removed and no additional penalty is introduced.
    Numerical convergence does not establish observability of weak columns.
    """
    if max_iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Invalid NNLS stopping parameters")
    if not np.isfinite(hessian).all() or not np.isfinite(rhs).all():
        raise ValueError("NNLS sufficient statistics must be finite")
    values, vectors = np.linalg.eigh(hessian)
    lipschitz = float(values[-1])
    if lipschitz <= 0:
        raise ValueError("Zero response operator")
    keep = values > lipschitz*1e-14
    basis, spectrum = vectors[:, keep], values[keep]/lipschitz
    reduced = int(keep.sum())*2 < hessian.shape[0]
    matrix = (basis*spectrum)@basis.T
    target = basis@(basis.T@rhs.reshape(rhs.shape[0], -1))/lipschitz
    current = np.zeros_like(target) if initial is None else np.array(initial, float, copy=True).reshape(target.shape)
    if not np.isfinite(current).all() or np.any(current < 0):
        raise ValueError("Initial NNLS density must be finite and nonnegative")
    extrapolated, acceleration = current.copy(), 1.
    converged = False
    for iteration in range(1, max_iterations+1):
        gradient = (basis@(spectrum[:, None]*(basis.T@extrapolated)) if reduced else matrix@extrapolated)-target
        updated = np.maximum(extrapolated-gradient, 0.)
        if iteration % 20 == 0 or iteration == 1 or iteration == max_iterations:
            g = (basis@(spectrum[:, None]*(basis.T@updated)) if reduced else matrix@updated)-target
            projected = updated-np.maximum(updated-g, 0.)
            kkt = float(np.linalg.norm(projected)/max(np.linalg.norm(updated), np.linalg.norm(target), 1e-15))
            if kkt < tolerance:
                current, converged = updated, True
                break
        next_acceleration = (1+np.sqrt(1+4*acceleration**2))/2
        next_extrapolated = updated+(acceleration-1)/next_acceleration*(updated-current)
        # Adaptive restart suppresses momentum oscillation near the active set.
        if np.sum((extrapolated-updated)*(updated-current)) > 0:
            next_acceleration, next_extrapolated = 1., updated.copy()
        current, extrapolated, acceleration = updated, next_extrapolated, next_acceleration
    result = current.reshape(rhs.shape)
    info = dict(converged=converged, iterations=iteration, projected_kkt_relative=kkt)
    return (result, info) if return_info else result


def fit_crystal_gains(responses, measurements, *, anchor_index, max_iterations=100,
                      tolerance=1e-5, gain_bounds=(0.05, 20.0),
                      voxel_width_mm, foil_thickness_mm, progress=None,
                      nnls_max_iterations=20000, nnls_tolerance=1e-7):
    """Alternating nonnegative LS with bounded gains and one fixed f_anchor=1.

    Uses equation (8)'s closed-form gain update, clipped to explicit bounds.
    Conditional convex subproblems do not establish a global joint optimum.
    Background/geometry cannot generally be corrected by scalar gains.
    """
    if max_iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Invalid gain fit stopping parameters")
    if len(gain_bounds) != 2 or not np.isfinite(gain_bounds).all() or not 0 < gain_bounds[0] <= 1 <= gain_bounds[1]:
        raise ValueError("Gain bounds must be positive and include the fixed anchor 1")
    operators, data, _, rhs = _joint_terms(responses, measurements,
                                         voxel_width_mm=voxel_width_mm,
                                         foil_thickness_mm=foil_thickness_mm)
    if not 0 <= anchor_index < len(operators):
        raise ValueError("Specify a valid scale anchor")
    grams = np.asarray([a.T@a for a in operators])
    back = np.asarray([(a.T@d.reshape(a.shape[0], -1)).reshape(rhs.shape)
                       for a, d in zip(operators, data)])
    gains = np.ones(len(operators))
    history = []
    rho = None
    converged = False
    for iteration in range(1, max_iterations+1):
        hessian = np.einsum("a,aij->ij", gains**2, grams)
        rhs = np.einsum("a,ai...->i...", gains, back)
        rho, inner = solve_gram_nnls(hessian, rhs, initial=rho, return_info=True,
            max_iterations=nnls_max_iterations, tolerance=nnls_tolerance)
        flat = rho.reshape(rho.shape[0], -1)
        numerator = np.sum(back.reshape(len(gains), flat.shape[0], -1)*flat[None], axis=(1, 2))
        denominator = np.einsum("ic,aij,jc->a", flat, grams, flat)
        updated = np.divide(numerator, denominator, out=gains.copy(), where=denominator > 0)
        updated = np.clip(updated, *gain_bounds)
        updated[anchor_index] = 1
        history.append(float(np.linalg.norm(updated-gains)/max(np.linalg.norm(gains), 1e-15)))
        gains = updated
        if progress is not None:
            progress(iteration, history[-1])
        if history[-1] < tolerance:
            converged = True
            break
    # Refit density with the final coefficients rather than the previous block.
    rho, inner = solve_gram_nnls(np.einsum("a,aij->ij", gains**2, grams),
                         np.einsum("a,ai...->i...", gains, back), initial=rho, return_info=True,
                         max_iterations=nnls_max_iterations, tolerance=nnls_tolerance)
    prediction = [(gain*a@rho.reshape(rho.shape[0], -1)).reshape(d.shape)
                  for gain, a, d in zip(gains, operators, data)]
    return dict(density=rho, gains=gains, predicted=prediction,
                residuals=[p-d for p, d in zip(prediction, data)],
                converged=converged and inner["converged"], iterations=iteration, relative_gain_change=history[-1],
                gain_change_history=history, nnls_info=inner)


def spatial_gradient(volume, spacing):
    """Forward finite differences with zero derivative on the last grid cell."""
    grad = np.zeros((volume.ndim, *volume.shape))
    for axis, step in enumerate(spacing):
        left = [slice(None)]*volume.ndim
        right = left.copy()
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        grad[(axis, *left)] = (volume[tuple(right)]-volume[tuple(left)])/step
    return grad


def gradient_adjoint(dual, spacing):
    """Exact transpose of spatial_gradient, including boundary treatment."""
    result = np.zeros(dual.shape[1:])
    for axis, step in enumerate(spacing):
        left = [slice(None)]*result.ndim
        right = left.copy()
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        values = dual[(axis, *left)]/step
        result[tuple(left)] -= values
        result[tuple(right)] += values
    return result


def reconstruct_tv(responses, measurements, *, spacing_mm, lambda_tv,
                   max_iterations=5000, tolerance=1e-6, initial_density=None, progress=None, **kwargs):
    """Nonnegative joint reconstruction with isotropic TV using Condat--Vu.

    Objective: .5*sum_a ||A_a rho-D_a||^2 + lambda_tv * sum_voxel ||grad rho||.
    TV includes physical 1/mm derivatives; cell volume is absorbed into lambda.
    Gains are FIXED, avoiding the density/gain scale ambiguity. Returns a
    convergence flag; the iteration limit does not imply convergence.
    """
    operators, data, hessian, rhs = _joint_terms(responses, measurements, **kwargs)
    spacing = np.asarray(spacing_mm, float)
    if spacing.shape != (rhs.ndim,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Provide positive spacing for every volume axis (x,y,z)")
    if not np.isfinite(lambda_tv) or lambda_tv < 0 or max_iterations < 1 or tolerance <= 0:
        raise ValueError("Invalid solver parameters")
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    lipschitz = eigenvalues[-1]
    if lipschitz <= 0:
        raise ValueError("Zero response operator")
    # Scale the entire objective by L for numerical stability; same minimizer.
    hessian, rhs, lam = hessian/lipschitz, rhs/lipschitz, lambda_tv/lipschitz
    resolved = eigenvalues > lipschitz*1e-14
    basis = eigenvectors[:, resolved]
    spectrum = eigenvalues[resolved]/lipschitz
    use_reduced = int(resolved.sum())*2 < hessian.shape[0]
    # Project both terms onto the same numerically resolved data modes.
    rhs = (basis@(basis.T@rhs.reshape(rhs.shape[0], -1))).reshape(rhs.shape)
    hessian = (basis*spectrum)@basis.T
    norm_k2 = sum(4/s**2 for s, n in zip(spacing, rhs.shape) if n > 1)
    # Balance the data and dual steps: tau*(L/2+sigma*||K||^2) < 1
    # after scaling L to 1. A tiny spatial pixel need not shrink the data step.
    tau, sigma = 0.9, 0.45/norm_k2 if norm_k2 > 0 else 1.0
    rho = np.zeros_like(rhs) if initial_density is None else np.array(initial_density, dtype=float, copy=True)
    if rho.shape != rhs.shape or not np.isfinite(rho).all() or np.any(rho < 0):
        raise ValueError("Initial density must be finite, nonnegative and match the grid")
    dual = np.zeros((rho.ndim, *rho.shape))
    converged = False
    for iteration in range(1, max_iterations+1):
        flat = rho.reshape(rho.shape[0], -1)
        product = basis@(spectrum[:, None]*(basis.T@flat)) if use_reduced else hessian@flat
        grad_data = product.reshape(rho.shape)-rhs
        new = np.maximum(rho-tau*(grad_data+gradient_adjoint(dual, spacing)), 0)
        new_dual = dual+sigma*spatial_gradient(2*new-rho, spacing)
        if lam == 0:
            new_dual.fill(0)
        else:
            norm = np.linalg.norm(new_dual, axis=0)
            new_dual /= np.maximum(1, norm/lam)[None, ...]
        primal_change = np.linalg.norm(new-rho)/max(np.linalg.norm(new), 1e-15)
        dual_change = np.linalg.norm(new_dual-dual)/max(np.linalg.norm(new_dual), 1e-15)
        rho, dual = new, new_dual
        if progress is not None and iteration % 1000 == 0:
            progress(iteration, float(primal_change), float(dual_change))
        if iteration > 10 and max(primal_change, dual_change) < tolerance:
            converged = True
            break
    predicted = [(a@rho.reshape(rho.shape[0], -1)).reshape(d.shape) for a, d in zip(operators, data)]
    residuals = [p-d for p, d in zip(predicted, data)]
    objective = .5*sum(np.sum(r*r) for r in residuals)+lambda_tv*np.linalg.norm(spatial_gradient(rho, spacing), axis=0).sum()
    return dict(density=rho, predicted=predicted, residuals=residuals,
                converged=converged, iterations=iteration, objective=float(objective),
                primal_relative_change=float(primal_change), dual_relative_change=float(dual_change))
