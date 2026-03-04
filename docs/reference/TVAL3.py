#  TVAL3 deconvolusion
from __future__ import annotations

import importlib
import numpy as np
import os
from typing import Optional, Literal
from Shared.algo.Utilities import Padding, CropImage, AsNumpy

from Shared import use_cupy

# Try to import CuPy; fall back to NumPy if it is absent or if the caller
# explicitly disables the GPU via an environment variable.
_cupy_spec = importlib.util.find_spec("cupy")
_use_gpu    = (_cupy_spec is not None) and use_cupy()

if _use_gpu:
    import cupy as cp    
    xp    = cp                       # array module alias used everywhere
    _fft  = cp.fft                   # CuPy’s FFT sub-module
    try:                             # v12+-only: enlarge plan cache ⇒ no first-call penalty
        cp.fft.config.set_plan_cache_size(64)   # 64 plans ≈ 16–32 MiB, tune to taste
    except AttributeError:
        pass                         # older CuPy, safe to ignore
else:
    xp   = np
    _fft = np.fft

import logging
logger = logging.getLogger(__name__)

def fft2(a, *args, **kwargs):
    """Backend-agnostic 2-D FFT; chooses cupy or numpy once at import."""
    return _fft.fft2(a, *args, **kwargs)

def ifft2(a, *args, **kwargs):
    """Backend-agnostic inverse 2-D FFT."""
    return _fft.ifft2(a, *args, **kwargs)

fftshift  = _fft.fftshift
ifftshift = _fft.ifftshift
fftfreq = _fft.fftfreq

def _freeze(a):
    try:
        a.flags.writeable = False
    except AttributeError:
        pass
    return a

EPSILON = 1e-6
PHI = 1.618033988749895
PHI_INV = 1.0 / PHI

class TVAL3:
    """
    Implementation of TVAL3 (Total Variation Augmented Lagrangian ALternating Direction Algorithm)
    for image deconvolution/deblurring with known Point Spread Function (PSF).
    
    This algorithm solves the following optimization problem:
    min lambda_TV*|∇u|₁ + 0.5*|Au - b|²
    
    where:
    - u is the image we want to recover
    - A is the convolution operator with the PSF
    - b is the blurred image
    - |∇u|₁ is the L1-norm of the gradient (total variation)
    """

    def __init__(self,
                 psf: np.ndarray,
                 image: np.ndarray,
                 mu: float = 2**5,
                 mu_max: float = 2**10,
                 mu_min: float = 2**-5,
                 mu_factor: float = 1.2,
                 lambda_tv: float = 0.01,
                 tol: float =  1e-4,
                 maxiter: int =  300,
                 TVnorm: int = 2,
                 nonneg: bool = True,
                 psf_normalize: bool = True,
                 padding_scale: float = 2.0,
                 initialEstimate: Optional[np.ndarray] = None) -> None:                 
        """
        Initialize the TVAL3 deconvolution algorithm.        
        """

        self._channels_last = image.ndim == 3
        self._n_channels = image.shape[-1] if self._channels_last else 1
        if self._channels_last:
            gray = image.mean(-1)
        else:
            gray = image
        
        self.h, self.w = gray.shape
        self.full_shape = (int(padding_scale*self.h), int(padding_scale*self.w))

        self.psf = xp.array(psf).astype(xp.float64)
        if psf_normalize:
            s = self.psf.sum()
            if s != 0:
                self.psf = self.psf / s

        # The blurred image
        self.image = (Padding(xp.array(gray), self.full_shape, "LinearRamp")).astype(xp.float64)
          
        self.H = _freeze(fft2(ifftshift(Padding(self.psf, self.full_shape, "Zero"))))
        self.H_conj = _freeze(self.H.conj())
        self.H_H_conj = _freeze(xp.real(self.H_conj * self.H))
        self.ATb = _freeze(xp.real(ifft2(self.H_conj * fft2(self.image))))

        # Laplacian in frequency domain
        M, N = self.full_shape
        fy = fftfreq(M).reshape(-1, 1)
        fx = fftfreq(N).reshape(1, -1)
        #self.lap_fft = 4 - 2 * xp.cos(2 * xp.pi * fx) - 2 * xp.cos(2 * xp.pi * fy)
        self.lap_fft = (4 - 2*xp.cos(2*xp.pi*fy) - 2*xp.cos(2*xp.pi*fx))

        # Initial guess
        if initialEstimate is not None:            
            self.estimated_image = (Padding(xp.array(initialEstimate), self.full_shape, "LinearRamp")).astype(xp.float64)
        else:
            self.estimated_image = (Padding(xp.array(gray), self.full_shape, "LinearRamp")).astype(xp.float64)        

        self.costs = []         

        # Parameters
        self.mu = mu
        self.mu_max = mu_max
        self.mu_min = mu_min
        self.mu_factor = mu_factor
        self.lambda_tv = lambda_tv
        self.tol = tol
        self.maxiter = maxiter
        self.TVnorm = TVnorm
        self.nonneg = nonneg

    def _grad(self, u):
        """
        Compute the discrete gradients in x and y directions.
        Uses forward differences with circular boundary conditions.
        
        Parameters:
        -----------
        u : ndarray
            Input image
            
        Returns:
        --------
        dx, dy : ndarray, ndarray
            Gradient in x and y directions
        """        
        dx = xp.zeros_like(u)
        dy = xp.zeros_like(u)                
        dx[:, :-1] = u[:, 1:] - u[:, :-1]
        dx[:, -1] = u[:, 0] - u[:, -1]  # Wrap around
        dy[:-1, :] = u[1:, :] - u[:-1, :]
        dy[-1, :] = u[0, :] - u[-1, :]
        return dx, dy

    def _div(self, px, py):
        """
        Compute the discrete divergence (adjoint of gradient operator).
        Uses backward differences with circular boundary conditions.
        
        Parameters:
        -----------
        px, py : ndarray, ndarray
            Components of the vector field
            
        Returns:
        --------
        div : ndarray
            Divergence of the vector field
        """                
        dx = xp.zeros_like(px)
        dy = xp.zeros_like(py)
        dx[:, 1:] = px[:, 1:] - px[:, :-1]
        dy[1:, :] = py[1:, :] - py[:-1, :]
        return dx + dy

    def _shrink(self, x, y, thresh, eps, adaptive_thresh=None):
        """
        Shrinkage operator for TV norm.

        Parameters:
        -----------
        x, y : ndarray, ndarray
            Components of the vector field
        thresh : float
            Threshold value
        eps : float
            Small value to avoid division by zero
        adaptive_thresh : ndarray, optional
            Spatially-varying threshold (if None, use uniform thresh)            

        Returns:
        --------
        x, y : ndarray, ndarray
            Shrinked components
        """
        # Use adaptive threshold if provided, otherwise uniform
        if adaptive_thresh is not None:
            effective_thresh = adaptive_thresh
        else:
            effective_thresh = thresh        

        if self.TVnorm == 1:
            return xp.sign(x) * xp.maximum(xp.abs(x) - effective_thresh, 0), xp.sign(y) * xp.maximum(xp.abs(y) - effective_thresh, 0)
        else:
            mag = xp.sqrt(x**2 + y**2)
            scale = xp.maximum(mag - effective_thresh, 0) / (mag + eps)
            return scale * x, scale * y

    def _update_u_fft(self, rhs_fft, n_inner=3):
        """
        Update the image u in the frequency domain using the FFT.

        Parameters:
        -----------
        rhs_fft : ndarray
            Right-hand side of the linear system
        n_inner : int, optional
            Number of inner iterations for Richardson refinement (default: 3)

        Returns:
        --------
        u : ndarray
            Updated image
        """   
        denom = self.H_H_conj + self.mu * self.lap_fft

        # Initial solution
        u_fft = rhs_fft / (denom + EPSILON)

        # Richardson refinement
        for _ in range(n_inner):
            residual = rhs_fft - (self.H_H_conj + self.mu * self.lap_fft) * u_fft
            u_fft += residual / (denom + EPSILON)        
        
        return xp.real(ifft2(u_fft))
        
    def _compute_edge_map(self, u):
        """
        Compute edge strength for adaptive regularization

        Parameters:
        -----------
        u : ndarray
            Image to compute edge strength for

        Returns:
        --------
        adaptive_weight : ndarray
            Adaptive weight for TV regularization based on edge strength
        """
        dx, dy = self._grad(u)
        edge_strength = xp.sqrt(dx**2 + dy**2 + 1e-8)
        
        # Normalize edge strength to [0, 1]
        e_min = edge_strength.min()
        e_max = edge_strength.max()
        if e_max > e_min:
            edge_strength = (edge_strength - e_min) / (e_max - e_min)
        else:
            edge_strength = xp.zeros_like(edge_strength)
        
        # Create adaptive weights: 
        # - At edges (edge_strength ≈ 1): weight ≈ 0.2 * lambda_tv (less smoothing)
        # - In smooth regions (edge_strength ≈ 0): weight ≈ lambda_tv (normal smoothing)
        adaptive_weight = self.lambda_tv * (1 - 0.8 * edge_strength)

        return adaptive_weight

    def _compute_cost(self, u, w1, w2, dx, dy):
            """
            Compute the cost function of the TVAL3 algorithm.

            Parameters:
            -----------
            u : ndarray
                Image to be deblurred
            w1, w2 : ndarray, ndarray
                Components of the vector field
            dx, dy : ndarray, ndarray
                Gradients of the image

            Returns:
            --------
            cost : float
                Value of the cost function
            """
            Au = xp.real(ifft2(self.H * fft2(u)))
            data_term = 0.5 * xp.sum((Au - self.image)**2)
            if self.TVnorm == 1:
                tv_term = xp.sum(xp.abs(w1)) + xp.sum(xp.abs(w2))
            else:
                tv_term = xp.sum(xp.sqrt(w1**2 + w2**2 + EPSILON))
            aug_term = 0.5 * self.mu * (xp.sum((dx - w1)**2) + xp.sum((dy - w2)**2))
            
            return data_term + self.lambda_tv * tv_term + aug_term 

    def _check_convergence(self, dx, dy, w1, w1_old, w2, w2_old, u):
        """
        Check convergence using both primal and dual residuals.
        This is more robust than checking just one metric.
        
        The primal residual measures how well constraints are satisfied.
        The dual residual measures how much the dual variables are changing.
        
        Parameters:
        -----------
        dx, dy : ndarray
            Current gradients
        w1, w2 : ndarray
            Current auxiliary variables
        w1_old, w2_old : ndarray
            Previous auxiliary variables
        u : ndarray
            Current image estimate
            
        Returns:
        --------
        converged : bool
            Whether convergence criteria are met
        rel_change : float
            Combined relative change metric
        r_primal : float
            Primal residual norm
        r_dual : float
            Dual residual norm
        """
        # Primal residuals: how well dx ≈ w1 and dy ≈ w2
        r_primal_x = xp.linalg.norm(dx - w1)
        r_primal_y = xp.linalg.norm(dy - w2)
        r_primal = xp.sqrt(r_primal_x**2 + r_primal_y**2)
        
        # Dual residuals: how much w is changing
        r_dual_x = self.mu * xp.linalg.norm(w1 - w1_old)
        r_dual_y = self.mu * xp.linalg.norm(w2 - w2_old)
        r_dual = xp.sqrt(r_dual_x**2 + r_dual_y**2)
        
        # Compute relative changes for scale-invariance
        norm_primal = xp.sqrt(xp.linalg.norm(dx)**2 + xp.linalg.norm(dy)**2) + EPSILON
        norm_dual = xp.sqrt(xp.linalg.norm(w1)**2 + xp.linalg.norm(w2)**2) + EPSILON
        
        rel_primal = r_primal / norm_primal
        rel_dual = r_dual / norm_dual
        
        # Combined relative change (you can adjust the weighting)
        rel_change = max(rel_primal, rel_dual)
        
        # Check if converged
        converged = rel_change < self.tol
        
        return converged, rel_change, r_primal, r_dual         

    def get_cost_history(self):
        """
        Get the history of cost function values during the iterations.
        
        Returns:
        --------
        costs : list
            List of cost function values at each iteration
        """
        return self.costs

    def deblur(self, verbose=True):
        """
        Run the TVAL3 algorithm to recover the deblurred image.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print the iteration details (default: True)
            
        Returns:
        --------
        u : ndarray
            Recovered sharp image
        """ 
        
        # Initialize u with the initial estimate
        u = self.estimated_image.copy()

        # Adaptive epsilon based on data type
        EPSILON = 1e-6 if u.dtype == np.float64 else 1e-4
        eps_grad = 1e-4 if u.dtype == np.float32 else 1e-6
        eps_shrink = 1e-8 if u.dtype == np.float64 else 1e-6

        # Calculate initial gradients
        dx, dy = self._grad(u)

        # Initialize dual variables
        w1, w2 = dx.copy(), dy.copy()
        dual1 = xp.zeros_like(u)
        dual2 = xp.zeros_like(u)

        # Initialize variables
        self.costs = [] 
        prev_cost  = self._compute_cost(u, w1, w2, dx, dy)  
        self.costs.append(self._compute_cost(u, w1, w2, dx, dy))

        # main loop
        for iter in range(1, self.maxiter + 1):
            
            # u-subproblem
            rhs = self.ATb + self.mu * self._div(w1 - dual1, w2 - dual2)
            rhs_fft = fft2(rhs)
            u = self._update_u_fft(rhs_fft)
            if self.nonneg:
                u = xp.maximum(0, u)

            # Gradient
            dx, dy = self._grad(u)
            
            # w-subproblem
            w1_old = w1.copy()
            w2_old = w2.copy()
            adaptive_weights = self._compute_edge_map(u)
            if adaptive_weights is not None:    
                w1, w2 = self._shrink(dx + dual1, dy + dual2, self.lambda_tv / self.mu, eps_grad, adaptive_thresh=adaptive_weights / self.mu)
            else:
                w1, w2 = self._shrink(dx + dual1, dy + dual2, self.lambda_tv / self.mu, eps_grad)
            
            # Dual update
            dual1 += dx - w1
            dual2 += dy - w2

            # Compute the relative change
            rel_change = xp.linalg.norm(dx - w1) / (xp.linalg.norm(dx) + EPSILON)

            # Check convergence
            converged, rel_change, r_primal, r_dual = self._check_convergence(dx, dy, w1, w1_old, w2, w2_old, u)           

            # Compute cost
            TVAL3_cost = self._compute_cost(u, w1, w2, dx, dy)
            self.costs.append(TVAL3_cost)

            # Print cost and relative change
            if verbose:
                print(f"iter: {iter}, cost: {TVAL3_cost}, rel_change: {rel_change}, r_primal: {r_primal:.6e}, r_dual: {r_dual:.6e}, mu: {self.mu:.2e}")   

            # Check if cost is exploding or going NaN
            if np.isnan(TVAL3_cost) or np.isinf(TVAL3_cost) or TVAL3_cost > 1e20:
                if verbose:
                    print("Warning: Cost is exploding. Stopping early.")
                break

            cost_change = abs(TVAL3_cost - prev_cost) / (abs(prev_cost) + EPSILON)
            prev_cost = TVAL3_cost
            if cost_change < 1e-8 and iter > 10:
                if verbose:
                    print(f"Cost stagnation detected (change: {cost_change:.2e}). Stopping.")
                break

            # Check convergence
            if (rel_change < self.tol or converged) and iter > 5:
                if verbose:
                    print(f"Converged! Final residuals - Primal: {r_primal:.6e}, Dual: {r_dual:.6e}")
                break

            if not xp.isfinite(u).all():
                raise FloatingPointError("NaN encountered; check λ or μ")

            # Adaptively update mu - Update mu based on primal and dual residuals
            residual_ratio = r_primal / (r_dual + EPSILON)
            if residual_ratio > 10:
                # Primal residual too large, increase mu
                self.mu = min(self.mu * self.mu_factor, self.mu_max)
                if verbose and iter % 10 == 0:
                    print(f"  -> Increasing mu to {self.mu:.2e} (primal dominant)")
            elif residual_ratio < 0.1:
                # Dual residual too large, decrease mu:
                self.mu = max(self.mu / self.mu_factor, self.mu_min)
                if verbose and iter % 10 == 0:
                    print(f"  -> Decreasing mu to {self.mu:.2e} (dual dominant)")

        # Finalize the image
        estimated_image = CropImage(u, (self.h, self.w))

        n_channels = getattr(self, "_n_channels", 1) 
        if self._channels_last and n_channels > 1:
            estimated_image = xp.repeat(estimated_image[..., None], n_channels, axis=-1)

        return AsNumpy(estimated_image)
    
def TVAL3_deblur(image, psf, iters, **kwargs):
    return TVAL3(psf, image, maxiter=iters, **kwargs).deblur(verbose=False)    

