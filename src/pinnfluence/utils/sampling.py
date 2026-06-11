import os
import tempfile
import time
from collections.abc import Iterable
from functools import partial

import captum
import deepxde as dde
import numpy as np
import torch
from tqdm import tqdm

from .dataset import DummyDataset
from .models import ModelWrapper, PINNLoss

# Experimental setup summary (see full comments in original code)
# 1. Replace by N training points with different strategies
# 2. Add N points to training (ONCE and finetune)
# 3. Add N points iteratively (and finetune only short amount of time)


class Sampler:
    """Handles sampling strategies like top-k or distribution-based selection"""

    def __init__(
        self,
        strategy: str,
        num_samples: int = 1,
        k=2,  # exponent for emphasizing high scores
        c=0,  # constant for flattening the distribution
        data: dde.data.Data = None,
        sampling_strategy: str = "add",
    ):
        assert strategy in [
            "top_k",
            "distribution",
        ], f"Strategy {strategy} not implemented"

        self.strategy = strategy
        self.num_samples = num_samples
        self.distribution_k = k
        self.distribution_c = c
        self.data = data
        self.sampling_strategy = sampling_strategy
        # Set up the appropriate sampling function
        if self.strategy == "top_k":
            self.sample_points = partial(sample_top_k, k=num_samples)
        elif self.strategy == "distribution":
            self.sample_points = partial(sample_from_distribution, n_samples=num_samples, k=k, c=c)
        else:
            raise NotImplementedError(
                f"Strategy {strategy} not implemented (and how did you get past the assert?)"
            )

    def __call__(self, candidate_points, scores):
        sampled_points, sampled_idx = self.sample_points(candidate_points, scores)
        return sampled_points


def sample_top_k(
    candidate_points,
    scores,
    k: int,  # = n_samples
):
    """Select top-k points based on scores"""
    scores = torch.tensor(scores)
    topk_idx = torch.topk(scores, k, dim=0)[1].numpy()
    sample = candidate_points[topk_idx]

    return sample, topk_idx


def sample_from_distribution(
    candidate_points,
    scores,
    n_samples: int,  # = k
    k=2,  # exponent for emphasizing high scores
    c=0,  # constant for flattening the distribution
):
    """Sample points based on the distribution of scores"""
    if (scores < 0).any():
        scores = scores - scores.min()
    # prevent numerical errors
    scores /= scores.max()
    scores = np.power(scores, k) / np.power(scores, k).mean() + c
    scores = scores / scores.sum()
    # still catch underflows
    scores[np.isnan(scores)] = 0
    sample = np.random.choice(a=len(candidate_points), size=n_samples, p=scores, replace=False)

    return candidate_points[sample], sample


class Scorer:
    """Assigns scores to points based on different strategies"""

    def __init__(
        self,
        strategy: str,
        model: dde.Model = None,
        n_candidate_points: int = 100_000,
        summation_sign: str = "abs",
        potential_precalculated: str = None,  # cache for influence scores
        seed: int = 42,
        verbose=False,
    ):
        assert strategy in [
            "random",
            "RAR",
            "PINNfluence",
            "steepest_loss_gradient",
            "steepest_prediction_gradient",
        ], f"Strategy {strategy} not implemented"

        dde.config.set_random_seed(seed)

        self.strategy = strategy
        self.model = model
        self.n_candidate_points = n_candidate_points
        self.sign = summation_sign
        self.sample_candidates()

        # flag to track if sampler is called multiple times
        self.multiple_execution = False

        # Set up the appropriate scoring function
        if self.strategy == "random":
            self.score_points = score_random
        elif self.strategy == "RAR":
            self.score_points = partial(score_RAR, model=self.model)
        elif self.strategy == "PINNfluence":
            self.score_points = partial(
                score_TDA,
                model=self.model,
                potential_precalculated=potential_precalculated,
                summation_sign=summation_sign,
                show_progress=verbose,
            )
        elif self.strategy == "steepest_loss_gradient":
            self.score_points = partial(
                score_steepest_loss_gradient, model=self.model, show_progress=verbose
            )
        elif self.strategy == "steepest_prediction_gradient":
            self.score_points = partial(score_steepest_prediction_gradient, model=self.model)
        else:
            raise NotImplementedError(
                f"Strategy {strategy} not implemented (and how did you get past the assert?)"
            )

        if self.strategy != "random":
            assert self.model is not None, "Model must be provided for non-random strategies"

    def __call__(self, *args, **kwargs):
        """Score candidate points"""
        if self.multiple_execution:
            self.sample_candidates()

            if self.strategy == "PINNfluence":
                self.reinit_IF()

        scores = self.score_points(candidate_points=self.candidate_points)
        candidate_points = self.candidate_points

        self.multiple_execution = True

        return candidate_points, scores

    def sample_candidates(self):
        """Sample random points from the geometry"""
        self.candidate_points = sample_random_points(
            geometry=self.model.data.geom,
            num_points=self.n_candidate_points,
        )

    def reinit_IF(self):
        """Reinitialize influence function scoring"""
        self.score_points = partial(
            score_TDA,
            model=self.model,
            potential_precalculated=None,
            summation_sign="abs",
        )


def score_random(candidate_points):
    """Assign random scores to points"""
    return np.random.rand(len(candidate_points))


def score_residual(model, candidate_points):
    """Score by residual (absolute PDE residual)"""
    pde = model.data.pde
    residuals = np.abs(model.predict(candidate_points, operator=pde))

    if len(residuals.shape) == 3:
        residuals = residuals.sum(axis=(0, 2))
    elif len(residuals.shape) == 2:
        residuals = residuals[:, 0]
    else:
        raise ValueError(f"Unexpected shape of residuals: {residuals.shape}")

    return residuals


def score_RAR(model, candidate_points):
    """Score by residual-adaptive refinement (absolute PDE residual)"""

    residual_scores = score_residual(model, candidate_points)

    return residual_scores


def score_TDA(
    model,
    candidate_points,
    potential_precalculated: str = None,
    summation_sign="abs",
    show_progress=False,
):
    """Score using PINNfluence."""
    assert summation_sign in [
        "abs",
        "pos",
        "neg",
    ], "Please choose summation_sign from ['abs', 'pos', 'neg']"

    if potential_precalculated is not None:
        print(f"Precalculated exists: {os.path.exists(potential_precalculated)}")

    # Use precalculated scores if available
    if potential_precalculated is not None and os.path.exists(potential_precalculated):
        OG_candidate_points = candidate_points
        influences = np.load(potential_precalculated)
        influence_scores = influences[f"scores_{summation_sign}"]
        candidate_points = influences["candidate_points"]

        assert np.array_equal(candidate_points, OG_candidate_points), (
            "Candidate points mismatch in precomputed influences."
        )

    else:
        batch_size = 1024
        tda_instance = instantiate_IF(model, model_name=None, show_progress=show_progress)
        influence_scores = calculate_influence_scores(
            candidate_points, tda_instance, show_progress=show_progress, batch_size=batch_size
        )

        influence_scores = apply_sign(influence_scores, summation_sign).sum(axis=0)

    return influence_scores


def wrap_points_in_dataloader(arr, batch_size=1024):
    """Convert numpy array to PyTorch DataLoader"""
    x = torch.tensor(arr, requires_grad=True)
    return torch.utils.data.DataLoader(x, batch_size=batch_size)


def score_steepest_loss_gradient(model, candidate_points, show_progress=False):
    """Score by L2 norm of gradients of loss w.r.t. parameters"""
    candidate_loader = wrap_points_in_dataloader(candidate_points, batch_size=1024)

    wrapped_model = ModelWrapper(
        net=model.net,
        pde=model.data.pde,
        bcs=model.data.bcs,  # bcs are None
    )
    grads_l2_norms = []

    loss_fn = PINNLoss()

    for batch in tqdm(candidate_loader, disable=not show_progress):
        batch = batch.requires_grad_(True)
        grads = captum._utils.gradient._compute_jacobian_wrt_params(
            wrapped_model,
            [batch],
            labels=torch.zeros_like(batch[:, 0]),
            loss_fn=loss_fn,
        )
        grads_l2_norm = compute_input_wise_l2_norm(grads)
        grads_l2_norms.append(grads_l2_norm)

    return np.concatenate(grads_l2_norms)


def score_steepest_prediction_gradient(model, candidate_points):
    """Score by L2 norm of gradients of network output w.r.t. input"""
    x = torch.tensor(candidate_points, requires_grad=True)
    out = model.net(x)

    grad = torch.autograd.grad(out, x, grad_outputs=torch.ones_like(out))[0]
    grad_l2_norm = torch.square(grad).sum(axis=1).sqrt()
    grad_l2_norm = grad_l2_norm.detach().numpy()

    return grad_l2_norm


def apply_sign(scores, sign):
    """Apply transformation based on sign parameter"""
    if sign == "abs":
        scores = np.abs(scores)
    elif sign == "neg":
        scores = -scores
    elif sign == "pos":
        pass
    else:
        raise NotImplementedError(f"Sign {sign} not implemented")

    return scores


def instantiate_IF(
    model,
    model_name: str,
    batch_size: int = None,
    show_progress: bool = False,
    seed: int = 0,
    use_train_set: bool = True,
    tmp_dir: str | None = "tmp/r_cache/",
    prefer_load_R: bool = True,
    projection_dim: int = 50,
):
    """Create an instance of Influence Function estimator"""
    data = model.data
    net = model.net

    # Sample points for hessian approximation
    if use_train_set:
        influences_set = data.train_x_all
    else:
        influences_set = data.geom.random_points(1_000, random="pseudo")

    print(f"Influences set shape (right side of equation): {influences_set.shape}")
    # Add dummy zero targets for DataLoader
    influences_set = DummyDataset(influences_set, return_zeroes=True)

    if batch_size is None:
        batch_size = min(1024, len(influences_set))

    pde_net = ModelWrapper(
        net=net,
        pde=data.pde,
        bcs=data.bcs,
    )

    print("Approximating hessian")
    start = time.time()
    # Approximate Hessian using Arnoldi method
    if_instance = captum.influence.ArnoldiInfluenceFunctionPrecomputed(
        pde_net,
        train_dataset=influences_set,
        loss_fn=PINNLoss(),
        show_progress=show_progress,
        checkpoint="dummy",
        checkpoints_load_func=lambda x, y: 0,  # net assumed to be loaded
        batch_size=batch_size,
        seed=seed,
        r_cache_path=f"{tmp_dir}/{model_name}_r_cache.pt",
        prefer_load_R=prefer_load_R,
        projection_dim=projection_dim,
    )
    end = time.time()
    print(f"Approximation took: {end - start}")

    return if_instance


def instantiate_IF_individual_loss_term(
    model,
    loss_idx: Iterable[int],
    batch_size: int | None = None,
    show_progress: bool = False,
    seed: int = 0,
):
    data = model.data
    net = model.net

    influences_set = data.train_x_all

    pde_net = ModelWrapper(
        net=net,
        pde=data.pde,
        bcs=data.bcs,
    )

    loss_fn = PINNLoss(include_all_losses=False, include_specific_ids=loss_idx)

    train_x = torch.tensor(influences_set, requires_grad=True)
    out = pde_net(train_x)

    # reduce to points where loss terms are defined
    defined_mask = torch.isnan(out).any(dim=1) == False
    influences_set = influences_set[defined_mask]

    print(
        f"Number of training points considered for loss_idx {loss_idx}: {influences_set.shape[0]}"
    )
    # Add dummy zero targets for DataLoader
    influences_set = DummyDataset(influences_set, return_zeroes=True)

    if batch_size is None:
        batch_size = len(influences_set)

    tmp_dir = tempfile.TemporaryDirectory()

    start = time.time()
    if_instance = captum.influence.ArnoldiInfluenceFunctionPrecomputed(
        pde_net,
        train_dataset=influences_set,
        loss_fn=loss_fn,
        show_progress=show_progress,
        checkpoint="dummy",
        checkpoints_load_func=lambda x, y: 0,  # net assumed to be loaded
        batch_size=batch_size,
        seed=seed,
        r_cache_path=tmp_dir.name + "/r_cache.pt",
        trainset_gradients_cache_path=tmp_dir.name + "/trainset_grads_cache.pt",
    )
    end = time.time()
    print(f"Approximation for loss terms {loss_idx} took: {end - start}")

    return if_instance, defined_mask


def calculate_influence_scores(
    candidate_points,
    tda_instance: captum.influence.ArnoldiInfluenceFunction,
    show_progress: bool = False,
    batch_size: int = None,
):
    """Calculate influence scores for candidate points"""
    # Use appropriate batch size
    if batch_size is None:
        batch_size = min(1024, len(candidate_points))
    else:
        batch_size = min(batch_size, len(candidate_points))

    candidate_set = DummyDataset(candidate_points, return_zeroes=True)
    candidate_loader = torch.utils.data.DataLoader(
        candidate_set,
        batch_size=batch_size,
    )

    test_samples = candidate_loader

    print("Calculating influences")
    start = time.time()
    influences = tda_instance.influence(test_samples, show_progress=show_progress)
    end = time.time()
    print(f"Influence calculation took: {end - start}")

    return influences.detach().numpy().astype(np.float32)


def sample_random_points(
    geometry: dde.geometry.Geometry,
    num_points: int = 1,
    num_bcs: int | None = None,
):
    """Sample random points from the geometry maintaining the same ratios as model data.

    Args:
        geometry: The geometry to sample from
        num_points: Total number of points to sample
    """

    if hasattr(geometry, "random_boundary_points"):
        if hasattr(geometry, "random_initial_points") and num_bcs is not None:
            num_domain = int(9 * num_points / 10)
            num_remaining = num_points - num_domain
            num_initial = int(num_remaining / num_bcs)
            num_boundary = num_points - num_domain - num_initial

            points = geometry.random_points(num_domain, random="pseudo")
            points_initial = geometry.random_initial_points(num_initial, random="pseudo")
            points_boundary = geometry.random_boundary_points(num_boundary, random="pseudo")
            points = np.vstack([points, points_initial, points_boundary])
        else:
            points = geometry.random_points((9 * num_points // 10), random="pseudo")
            points_boundary = geometry.random_boundary_points((num_points // 10), random="pseudo")
            points = np.vstack([points, points_boundary])
    else:
        points = geometry.random_points(num_points, random="pseudo")

    return points


def compute_input_wise_l2_norm(grads):
    """
    Compute the input-wise L2 norm of gradients.
    """
    grad_norms_per_param = []

    for grad in grads:
        # Flatten while keeping the input dimension
        n_input = grad.shape[0]
        grad_flat = grad.view(n_input, -1)

        # Compute squared L2 norm per input
        grad_squared = grad_flat.pow(2)
        grad_norms = grad_squared.sum(dim=1)
        grad_norms_per_param.append(grad_norms)

    # Sum squared norms across parameters
    total_grad_norms_squared = sum(grad_norms_per_param)

    # Take square root for L2 norm
    total_grad_norms = total_grad_norms_squared.sqrt()

    return total_grad_norms
