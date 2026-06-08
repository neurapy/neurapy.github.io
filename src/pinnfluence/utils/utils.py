import json
import os
import sys
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import CenteredNorm, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy import stats

from pinnfluence.problem_factory import construct_problem
from pinnfluence.utils.defaults import BAD_PROBLEMS, PROBLEMS
from pinnfluence.utils.models import ModelWrapper, PINNLoss

loss_term_names = {
    "allen_cahn": {
        "total_loss": "Total Loss",
        "pde_loss": "PDE Loss",
        "pde_0": "PDE Loss",
        "bc_loss": "BC Loss",
        "bc_0": "IC Loss",
        "bc_1": "Dirichlet BC ($x=1$)",
        "bc_2": "Dirichlet BC ($x=-1$)",
        "output_0": "$\\hat u$",
    },
    "burgers": {
        "total_loss": "Total Loss",
        "pde_loss": "PDE Loss",
        "pde_0": "PDE Loss",
        "bc_loss": "BC Loss",
        "bc_0": "IC Loss",
        "bc_1": "Dirichlet BC ($x=-1$ and $x=1$)",
        "output_0": "$\\hat u$",
    },
    "diffusion": {
        "total_loss": "Total Loss",
        "pde_loss": "PDE Loss",
        "pde_0": "PDE Loss",
        "bc_loss": "BC Loss",
        "bc_0": "IC Loss",
        "bc_1": "Dirichlet BC ($x=-1$ and $x=1$)",
        "output_0": "$\\hat u$",
    },
    "drift_diffusion": {
        "total_loss": "Total Loss",
        "pde_loss": "PDE Loss",
        "pde_0": "PDE Loss",
        "bc_loss": "BC Loss",
        "bc_0": "IC Loss",
        "bc_1": "Periodic BC ($x=0$)",
        "bc_2": "Periodic BC ($x=2\\pi$)",
        "output_0": "$\\hat u$",
    },
    "navier_stokes_nd": {
        "total_loss": "Total Loss",
        "pde_loss": "PDE Loss",
        "pde_0": "PDE (continuity)",
        "pde_1": "PDE (x-momentum)",
        "pde_2": "PDE (y-momentum)",
        "bc_loss": "BC Loss",
        "bc_0": "No-slip $u$ BC (x-direction)",
        "bc_1": "No-slip $v$ BC (y-direction)",
        "bc_2": "Inflow $u$ BC (x-direction)",
        "bc_3": "Inflow $v$ BC (y-direction)",
        "bc_4": "Outflow $u$ BC (x-direction)",
        "bc_5": "Outflow $v$ BC (y-direction)",
        "output_0": "$\\hat u$",
        "output_1": "$\\hat v$",
        "output_2": "$\\hat p$",
    },
    "poisson_disk": {
        "total_loss": "Total Loss",
        "pde_loss": "PDE Loss",
        "pde_0": "PDE Loss",
        "bc_loss": "BC Loss",
        "bc_0": "Dirichlet BC ($u(x) = 0$ at $||{x}||=1$)",
        "output_0": "$\\hat u$",
    },
    "wave": {
        "total_loss": "Total Loss",
        "pde_loss": "PDE Loss",
        "pde_0": "PDE Loss",
        "bc_loss": "BC Loss",
        "bc_0": "IC Loss",
        "bc_1": "Dirichlet BC ($u(0,t) = 0$)",
        "bc_2": "Dirichlet BC ($u(1,t) = 0$)",
        "bc_3": "Operator BC ($\\frac{\\partial u}{\\partial t}$ at $t=0$)",
        "output_0": "$\\hat u$",
    },
}


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def set_default_device(device: str = "cpu"):
    if device == "cpu":
        torch.set_default_device("cpu")
        print("Using CPU")
    elif device == "cuda":
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
            print("Using CUDA")
        else:
            print("CUDA not available. Using CPU")
            torch.set_default_device("cpu")
    elif device == "mps":
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            torch._dynamo.disable()
            torch.set_default_device("mps")
            torch._dynamo.reset()
            print("Using mps")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            print("Using MPS")
        else:
            print("MPS not available. Using CPU")
            torch.set_default_device("cpu")
    else:
        print("Invalid device. Using CPU")
        torch.set_default_device("cpu")


def generate_experiment_df(experiment_path: str):
    experiment_path = Path(experiment_path)
    df = pd.DataFrame(
        columns=[
            "problem",
            "model_name",
            "strategy",
            "n_iterations_finetune",
            "n_iterations_lbfgs_finetune",
            "lr",
            "n_candidate_points",
            "n_samples",
            "distribution_k",
            "distribution_c",
            "scoring_method",
            "scoring_sign",
            "pertubation_strategy",
            "epoch",
            "criterion",
            "train_loss",
            "valid_loss",
            "test_loss",
            "l2re",
            "mse",
        ]
    )

    for dir in experiment_path.iterdir():
        config = dir / "config.json"

        if not config.exists():
            continue

        with open(config) as f:
            config = json.load(f)

        model_name = config["model_name"]
        config["seed"] = int(model_name[model_name.rfind("_") + 1 :])

        csvs = list(dir.glob("*.csv"))
        if len(csvs) == 0:
            continue

        csv = csvs[0]
        df_ = pd.read_csv(csv)

        for criterion in ["train", "valid"]:
            best_row = df_.iloc[df_[f"{criterion}_loss"].idxmin()]

            config.update(
                {
                    "epoch": best_row["epoch"],
                    "criterion": criterion,
                    "train_loss": best_row["train_loss"],
                    "valid_loss": best_row["valid_loss"],
                    "test_loss": best_row["test_loss"],
                    "l2re": best_row["l2_relative_error"],
                    "mse": best_row["mse"],
                }
            )

            df = pd.concat([df, pd.DataFrame([config])])

    return df


def plot_prediction_heatmap(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    title: str,
    cmap: str = "jet",
):
    if y_true.shape[1] == 1:
        fig, ax = plt.subplots(figsize=(15, 5), ncols=3)

        sc = ax[0].scatter(X[:, 0], X[:, 1], c=y_pred, cmap=cmap)
        ax[0].set_title("Predicted")
        fig.colorbar(sc, ax=ax[0])
        sc = ax[1].scatter(X[:, 0], X[:, 1], c=y_true, cmap=cmap)
        ax[1].set_title("True")
        fig.colorbar(sc, ax=ax[1])
        sc = ax[2].scatter(X[:, 0], X[:, 1], c=residuals, cmap=cmap)
        ax[2].set_title("Residuals")
        fig.colorbar(sc, ax=ax[2])

    else:
        fig, ax = plt.subplots(
            figsize=(15, 5 * y_true.shape[1]),
            ncols=3,
            nrows=y_true.shape[1],
            sharex=True,
            sharey=True,
        )

        ax[0, 0].set_title("Predicted")
        ax[0, 1].set_title("True")
        ax[0, 2].set_title("Residuals")

        for i in range(y_true.shape[1]):
            sc = ax[i, 0].scatter(X[:, 0], X[:, 1], c=y_pred[:, i], cmap=cmap)
            fig.colorbar(sc, ax=ax[i, 0])
            sc = ax[i, 1].scatter(X[:, 0], X[:, 1], c=y_true[:, i], cmap=cmap)
            fig.colorbar(sc, ax=ax[i, 1])
            sc = ax[i, 2].scatter(X[:, 0], X[:, 1], c=residuals[i], cmap=cmap)
            fig.colorbar(sc, ax=ax[i, 2])

    fig.suptitle(title)

    return fig


def plot_heatmap(
    X: np.ndarray,
    y: np.ndarray,
    title: str = None,
    cmap: str = "bwr",
    use_norm: bool = False,
    x_label: str = "x1",
    y_label: str = "x2",
    figsize: tuple = (10, 10),
):
    if use_norm:
        norm = TwoSlopeNorm(vmin=-np.max(np.abs(y)), vcenter=0, vmax=np.max(np.abs(y)))
    else:
        norm = None

    fig, ax = plt.subplots(figsize=figsize)

    sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, norm=norm)
    fig.colorbar(sc, ax=ax)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)

    return fig, ax


def get_min_max_from_geom(geom):
    if hasattr(geom, "timedomain"):
        x1_min, x1_max = geom.geometry.bbox
        x2_min, x2_max = geom.timedomain.bbox
        x_min = np.concatenate([x1_min, x2_min])
        x_max = np.concatenate([x1_max, x2_max])
    else:
        x_min, x_max = geom.bbox
    return x_min, x_max


def scale_x(x, x_min=0, x_max=1):
    return (x - x_min) / (x_max - x_min)


def scaled_rbf_kernel(X, Y, sigma=1, scale_fn=scale_x):
    return np.exp(-np.linalg.norm(scale_fn(X) - scale_fn(Y), ord=2, axis=1) / (2 * sigma**2))


def load_problem(
    problem_name: str,
    params_dict: dict,
    broken: bool = False,
    soft_constrained: bool = True,
    float64: bool = True,
    seed: int = 0,
    drop_single_point_type: str = "none",
    load_path: str = None,
    force_reinitialize: bool = False,
):
    return construct_problem(
        problem_name=problem_name,
        layers=params_dict["layers"],
        num_domain=params_dict["num_domain"],
        num_boundary=params_dict["num_boundary"],
        num_initial=params_dict["num_initial"],
        n_iterations=params_dict["n_iterations"],
        n_iterations_lbfgs=params_dict["n_iterations_lbfgs"],
        optimizer=params_dict.get("optimizer", "adam"),
        soft_constrained=soft_constrained,
        float64=float64,
        seed=seed,
        drop_single_point_type=drop_single_point_type,
        load_path=load_path,
        broken=broken,
        force_reinit=force_reinitialize,
    )


def get_X_y_true(model, num_points: int = 50_000):
    if model.data.soln is not None:
        X = model.data.geom.uniform_points(num_points)
        Y = model.data.soln(X)
    else:
        X = model.data.holdout_test_x
        Y = model.data.holdout_test_y
    return X, Y


def get_influence_scores(model_name: str, load_path: str, influence_load_path: str = None):
    influence_load_path = influence_load_path if influence_load_path is not None else load_path
    return np.load(str(influence_load_path / f"{model_name}_influence_scores.npz"))


def get_leave_one_out_diff(
    problem_name: str,
    params_dict: dict,
    seed: int = 0,
    drop_single_point_type: str = "IC",
    load_path: str = None,
):
    model, _, _, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        load_path=load_path,
    )

    model_loo, _, _, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        drop_single_point_type=drop_single_point_type,
        load_path=load_path,
    )

    test_x = model.data.holdout_test_x

    y_pred = model.predict(test_x)
    y_pred_loo = model_loo.predict(test_x)

    return y_pred - y_pred_loo


def get_loo_correlation(
    problem_name: str,
    params_dict: dict,
    seed: int = 0,
    drop_single_point_type: str = "IC",
    load_path: str = None,
    influence_load_path: str = None,
):
    _, _, model_name, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        load_path=load_path,
    )

    influences = get_influence_scores(
        model_name=model_name,
        load_path=load_path,
        influence_load_path=influence_load_path,
    )

    loo_error = get_leave_one_out_diff(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        drop_single_point_type=drop_single_point_type,
        load_path=load_path,
    )

    removed_point_idx = get_removed_point_idx(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        drop_single_point_type=drop_single_point_type,
        load_path=load_path,
    )

    spearman_corr = []
    pearson_corr = []

    if loo_error.shape[1] > 1:
        for i in range(loo_error.shape[1]):
            spearman_corr.append(
                stats.spearmanr(
                    influences["infl_scores_outputs"][i][:, removed_point_idx],
                    loo_error[:, i].flatten(),
                )
            )
            pearson_corr.append(
                stats.pearsonr(
                    influences["infl_scores_outputs"][i][:, removed_point_idx],
                    loo_error[:, i].flatten(),
                )
            )
    else:
        spearman_corr.append(
            stats.spearmanr(
                influences["infl_scores_outputs"][0][:, removed_point_idx],
                loo_error.flatten(),
            )
        )
        pearson_corr.append(
            stats.pearsonr(
                influences["infl_scores_outputs"][0][:, removed_point_idx],
                loo_error.flatten(),
            )
        )

    return spearman_corr, pearson_corr


def get_removed_point_idx(
    problem_name: str,
    params_dict: dict,
    seed: int = 0,
    drop_single_point_type: str = "IC",
    load_path: str = None,
):
    model_original, _, _, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        load_path=load_path,
    )

    model_loo, _, _, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        drop_single_point_type=drop_single_point_type,
        load_path=load_path,
    )

    return np.where((model_original.data.train_x_all == model_loo.data.point_removed).all(axis=1))[
        0
    ][0]


def get_loss(
    problem_name: str,
    params_dict: dict,
    seed: int = 0,
    load_path: str = None,
    target: str = "train",
):
    model, _, _, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        load_path=load_path,
    )

    if target == "train":
        X = model.data.train_x_all
    else:
        X = model.data.holdout_test_x

    X_tensor = torch.tensor(X, dtype=torch.float64, requires_grad=True)

    wrapped_model = ModelWrapper(model.net, model.data.pde, model.data.bcs)
    residuals = wrapped_model(X_tensor)
    loss_fn = PINNLoss()
    loss = loss_fn(residuals, torch.zeros(X.shape[0], 1))

    return loss


def get_directionality_indicator(
    problem_name: str,
    params_dict: dict,
    seed: int = 0,
    load_path: str = None,
    influence_load_path: str = None,
    direction_dimension: int = 0,
    broken: bool = False,
    use_radial_distances: bool = False,
    right_term: str = "total_loss",
    left_term: str = "total_loss",
    method: str = "influences",
):
    """
    Calculate directionality indicator scores.

    For each test point, compute the ratio of influence scores from training points
    that are "upstream" in the specified direction dimension.

    Args:
        problem_name: Name of the problem
        params_dict: Problem parameters
        seed: Random seed
        load_path: Path to load models from
        direction_dimension: Which dimension to use for directionality (0 or 1)
        broken: Whether to use broken equation
        use_radial_distances: Whether to use radial distances instead of linear dimension
        right_term: Right side term for influence file (e.g., 'total_loss', 'pde_loss', 'output_0')
        left_term: Left side term for influence file (e.g., 'total_loss', 'pde_loss', 'bc_0')

    Returns:
        dir_indicator_scores: List of directionality indicator scores for each test point
        point_ratios: List of point ratios for each test point
    """
    model, _, model_name, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        load_path=load_path,
        broken=broken,
    )

    # Load influence/grad_dot file with new naming convention
    _infl_base = influence_load_path if influence_load_path is not None else load_path
    infl_path = _infl_base.joinpath(f"{model_name}_influence_scores")
    infl_file = infl_path / f"{method}_{right_term}_{left_term}.npz"

    if not infl_file.exists():
        raise FileNotFoundError(f"Influence file not found: {infl_file}")
    print(f"Loading influence scores from: {infl_file}")

    infl_data = np.load(infl_file)
    test_x = infl_data["candidate_points"]
    influences_cur = infl_data["scores"]

    train_x = model.data.train_x_all

    print(f"train_x shape: {train_x.shape}")
    print(f"candidates shape: {test_x.shape}")
    print(f"influences shape: {influences_cur.shape}")

    boundary_mask = model.data.geom.on_boundary(test_x)
    if hasattr(model.data.geom, "on_initial"):
        boundary_mask = np.logical_or(boundary_mask, model.data.geom.on_initial(test_x))

    dir_indicator_scores = []
    point_ratios = []

    for i, x in enumerate(test_x):
        if boundary_mask[i]:
            dir_indicator_scores.append(np.nan)
            point_ratios.append(np.nan)
            continue

        if use_radial_distances:
            center = np.array([0, 0])
            indices = np.where(
                np.linalg.norm(train_x - center, axis=1) <= np.linalg.norm(x - center)
            )[0]
        else:
            indices = np.where(train_x[:, direction_dimension] <= x[direction_dimension])[0]

        total_abs_influence = np.abs(influences_cur[i, :]).sum()
        if total_abs_influence > 0:
            upstream_influence = np.abs(influences_cur[i, indices]).sum()
            dir_indicator_scores.append(upstream_influence / total_abs_influence)
        else:
            dir_indicator_scores.append(np.nan)

        point_ratios.append(len(indices) / len(train_x))

    return dir_indicator_scores, point_ratios


def get_correlation(x, y):
    return stats.pearsonr(x, y), stats.spearmanr(x, y)


def get_loss_correlation(
    problem_name: str,
    params_dict: dict,
    seed: int = 0,
    load_path: str = None,
    influence_load_path: str = None,
    broken: bool = False,
):
    model, _, model_name, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        broken=broken,
    )

    train_loss = get_loss(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        load_path=load_path,
        target="train",
    )

    influences = get_influence_scores(
        model_name=model_name,
        load_path=load_path,
        influence_load_path=influence_load_path,
    )

    infl_abs_sum = np.abs(influences["scores"]).sum(axis=0)
    train_loss_sum = train_loss.detach().numpy().sum(axis=1)
    pearson_corr = stats.pearsonr(infl_abs_sum, train_loss_sum)
    spearman_corr = stats.spearmanr(infl_abs_sum, train_loss_sum)

    print(f"Pearson correlation: {pearson_corr}")
    print(f"Spearman correlation: {spearman_corr}")

    return pearson_corr, spearman_corr


def get_influences_for_closest_point(
    point: np.ndarray,
    problem_name: str,
    params_dict: dict,
    seed: int = 0,
    load_path: str = None,
    influence_load_path: str = None,
    broken: bool = False,
    target: str = "test",
    target_key: str = "infl_scores_outputs",
    target_dimension: int = 0,
):
    model, _, model_name, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        broken=broken,
        load_path=load_path,
    )

    if target == "train":
        X = model.data.train_x_all
    else:
        X = model.data.holdout_test_x

    influences = get_influence_scores(
        model_name=model_name,
        load_path=load_path,
        influence_load_path=influence_load_path,
    )

    target_influences = influences[target_key]
    target_influences *= len(model.data.train_x_all)
    if target_key == "infl_scores_outputs":
        target_influences = target_influences[target_dimension]

    distances = np.linalg.norm(X - point, axis=1)
    closest_point_idx = np.argmin(distances)

    if target == "train":
        return target_influences[:, closest_point_idx], X[closest_point_idx]
    return target_influences[closest_point_idx], X[closest_point_idx]


def get_proximity_indicator(
    problem_name: str,
    params_dict: dict,
    seed: int = 0,
    load_path: str = None,
    influence_load_path: str = None,
    sigma: float = 1.0,
    broken: bool = False,
    influence_key: str = "scores",
):
    """
    Calculate proximity indicator scores using scaled RBF kernel.

    For each test point, compute the weighted sum of influence scores from training points,
    where weights are given by the scaled RBF kernel based on distance.

    Args:
        problem_name: Name of the problem
        params_dict: Problem parameters
        seed: Random seed
        load_path: Path to load models from
        sigma: RBF kernel bandwidth parameter
        broken: Whether to use broken equation
        influence_key: Which influence scores to use ('scores', 'infl_scores_pde', 'infl_scores_bc', 'infl_scores_outputs')

    Returns:
        proximity_scores: List of proximity indicator scores for each test point
    """
    from functools import partial

    model, _, model_name, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        load_path=load_path,
        broken=broken,
    )

    influences = get_influence_scores(
        model_name=model_name,
        load_path=load_path,
        influence_load_path=influence_load_path,
    )

    train_x = model.data.train_x_all
    test_x = model.data.holdout_test_x

    # Get geometry bounds for scaling
    x_min, x_max = get_min_max_from_geom(model.data.geom)

    boundary_mask_test = model.data.geom.on_boundary(test_x)
    if hasattr(model.data.geom, "on_initial"):
        boundary_mask_test = np.logical_or(boundary_mask_test, model.data.geom.on_initial(test_x))

    boundary_mask_train = model.data.geom.on_boundary(train_x)
    if hasattr(model.data.geom, "on_initial"):
        boundary_mask_train = np.logical_or(
            boundary_mask_train, model.data.geom.on_initial(train_x)
        )

    # Create partial function for scaled RBF kernel
    scale_x_partial = partial(scale_x, x_min=x_min, x_max=x_max)
    scaled_rbf_kernel_ = partial(scaled_rbf_kernel, scale_fn=scale_x_partial, sigma=sigma)

    proximity_scores = []

    # Handle different influence keys
    if influence_key == "infl_scores_outputs":
        # For outputs, we need to handle multiple dimensions
        influences_ = influences[influence_key]
        num_outputs = influences_.shape[0]
        for output_dim in range(num_outputs):
            # Calculate for each output dimension
            output_scores = []
            influences_cur = influences_[output_dim][:, ~boundary_mask_train]

            for i, x in enumerate(test_x):
                if boundary_mask_test[i]:
                    proximity_scores.append(np.nan)
                    continue

                proximity_weights = scaled_rbf_kernel_(train_x[~boundary_mask_train], x)

                weighted_influence = np.abs(influences_cur[i, :] * proximity_weights).sum()
                total_abs_influence = np.abs(influences_cur[i, :]).sum()

                if total_abs_influence > 0:
                    output_scores.append(weighted_influence / total_abs_influence)
                else:
                    output_scores.append(np.nan)

            # Average across output dimensions
            proximity_scores.append(np.mean(output_scores))
    else:
        # For other keys, handle as single dimension
        influences_cur = influences[influence_key]

        influences_cur = influences_cur[:, ~boundary_mask_train]

        for i, x in enumerate(test_x):
            if boundary_mask_test[i]:
                proximity_scores.append(np.nan)
                continue

            proximity_weights = scaled_rbf_kernel_(train_x[~boundary_mask_train], x)
            weighted_influence = np.abs(influences_cur[i, :] * proximity_weights).sum()
            total_abs_influence = np.abs(influences_cur[i, :]).sum()

            if total_abs_influence > 0:
                proximity_scores.append(weighted_influence / total_abs_influence)
            else:
                proximity_scores.append(np.nan)

    return proximity_scores


def region_fraction(cur_infl, test_mask, train_mask, eps=1e-12):
    """
    Calculate the fraction of influence from train_mask region to test_mask region.

    Args:
        cur_infl: Influence matrix (test_points x train_points)
        test_mask: Boolean mask for test points
        train_mask: Boolean mask for train points
        eps: Small epsilon to avoid division by zero

    Returns:
        float: Fraction of influence from train region to test region
    """
    # num = np.abs(cur_infl[test_mask][:, train_mask]).sum()
    # den = np.abs(cur_infl[test_mask]).sum() + eps
    # return float(num/den)
    # macro average
    rows = cur_infl[np.asarray(test_mask)]  # select B
    if rows.shape[0] == 0:
        return 0.0  # or float('nan') if you prefer signaling "no test points"

    num = np.abs(rows[:, np.asarray(train_mask)]).sum(axis=1)  # per-row numerator
    den = np.abs(rows).sum(axis=1) + eps  # per-row denominator
    return float(np.mean(num / den))


def get_top_k_distance_indicator(
    problem_name: str,
    params_dict: dict,
    seed: int = 0,
    load_path: Path = None,
    influence_load_path: Path = None,
    broken: bool = False,
    influence_key: str = "scores",
    k: int = 25,
):
    """
    Calculate the mean Euclidean distance to the top-k most influential training samples.

    Args:
        problem_name: Name of the problem
        params_dict: Problem parameters
        seed: Random seed
        load_path: Path to load models from
        broken: Whether to use broken equation
        influence_key: Which influence scores to use ('scores', 'infl_scores_pde', 'infl_scores_bc', 'infl_scores_outputs')
        k: Number of top influential training samples to consider

    Returns:
        distance_scores: List of mean distance scores for each test point
    """

    model, _, model_name, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        load_path=load_path,
        broken=broken,
    )

    influences = get_influence_scores(
        model_name=model_name,
        load_path=load_path,
        influence_load_path=influence_load_path,
    )

    train_x = model.data.train_x_all
    test_x = model.data.holdout_test_x

    boundary_mask_test = model.data.geom.on_boundary(test_x)
    if hasattr(model.data.geom, "on_initial"):
        boundary_mask_test = np.logical_or(boundary_mask_test, model.data.geom.on_initial(test_x))

    distance_scores = []

    # Handle different influence keys
    if influence_key == "infl_scores_outputs":
        # For outputs, we need to handle multiple dimensions
        influences_ = influences[influence_key]
        num_outputs = influences_.shape[0]
        for output_dim in range(num_outputs):
            # Calculate for each output dimension
            output_scores = []
            influences_cur = influences_[output_dim]

            for i, x in enumerate(test_x):
                if boundary_mask_test[i]:
                    distance_scores.append(np.nan)
                    continue

                # Get absolute influence scores for this test point
                abs_influences = np.abs(influences_cur[i, :])

                # Find top-k most influential training points
                if len(abs_influences) >= k:
                    top_k_indices = np.argsort(abs_influences)[-k:]
                else:
                    top_k_indices = np.arange(len(abs_influences))

                # Calculate Euclidean distances to top-k points
                distances = np.linalg.norm(x - train_x[top_k_indices], axis=1)

                # Return mean distance
                output_scores.append(np.mean(distances))

            # Average across output dimensions
            distance_scores.append(np.mean(output_scores))
    else:
        # For other keys, handle as single dimension
        influences_cur = influences[influence_key]

        for i, x in enumerate(test_x):
            if boundary_mask_test[i]:
                distance_scores.append(np.nan)
                continue

            # Get absolute influence scores for this test point
            abs_influences = np.abs(influences_cur[i, :])

            # Find top-k most influential training points
            if len(abs_influences) >= k:
                top_k_indices = np.argsort(abs_influences)[-k:]
            else:
                top_k_indices = np.arange(len(abs_influences))

            # Calculate Euclidean distances to top-k points
            distances = np.linalg.norm(x - train_x[top_k_indices], axis=1)

            # Return mean distance
            distance_scores.append(np.mean(distances))

    return distance_scores


def visualize_predictions_comparison(
    problem_name="drift_diffusion",
    seed=0,  # Can be int or "all" to average across all seeds
    figsize=(14, 6),
    model_zoo_base="../../model_zoo_cluster",
    show_loss=False,
    show_error=False,
    use_train_points=False,
    cmap="coolwarm",
    marker_sizes=None,
    logscale=False,
):
    """
    Visualize model predictions across different model configurations.

    Parameters:
    -----------
    problem_name : str
        Name of the problem
    seed : int or "all"
        Random seed for model, or "all" to average across all available seeds
    figsize : tuple
        Figure size
    model_zoo_base : str
        Base path to model zoo
    show_residuals : bool
        If True, show residuals (y_true - y_pred) instead of predictions
    show_error : bool
        If True, show absolute error |y_true - y_pred| instead of predictions
        Takes precedence over show_residuals if both are True

    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from pinnfluence.utils.defaults import BAD_PROBLEMS, PROBLEMS
    from pinnfluence.utils.utils import load_problem

    configs = {"good": ("PROBLEMS", PROBLEMS), "bad": ("BAD_PROBLEMS", BAD_PROBLEMS)}

    if marker_sizes is None:
        marker_sizes = [50, 50]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for idx, (config_name, (dict_name, params_dict_source)) in enumerate(configs.items()):
        ax = axes[idx]

        # Load model
        params_dict = params_dict_source[problem_name]
        load_path = Path(model_zoo_base) / f"{problem_name}_float64"

        # Determine which seeds to use
        if seed == "all":
            if "seeds" in params_dict:
                seeds = params_dict["seeds"]
            else:
                seeds = list(range(10))
        else:
            seeds = [seed]

        try:
            # For averaging across seeds, create uniform reference points
            if seed == "all":
                # Load first model to get geometry
                with Capturing() as cap:
                    model, data, model_name, chkpt_path = load_problem(
                        problem_name=problem_name,
                        params_dict=params_dict,
                        load_path=load_path,
                        seed=seeds[0],
                    )

                # Get initial X
                if use_train_points:
                    X_orig = model.data.train_x_all
                    if model.data.soln is not None:
                        y_true = model.data.soln(X_orig)
                    elif show_error:
                        raise ValueError("Ground truth solution not available for training points.")
                else:
                    X_orig, y_true = get_X_y_true(model)

                # Identify boundary and initial points
                boundary_mask = data.geom.on_boundary(X_orig)
                if hasattr(data.geom, "on_initial"):
                    initial_mask = data.geom.on_initial(X_orig)
                else:
                    initial_mask = np.zeros(len(X_orig), dtype=bool)

                num_boundary = boundary_mask.sum()
                num_initial = initial_mask.sum()

                # Create uniform boundary and initial points
                # These are generated directly from geometry, ensuring consistent spatial locations
                uniform_boundary = (
                    data.geom.uniform_boundary_points(num_boundary)
                    if num_boundary > 0
                    else np.empty((0, X_orig.shape[1]))
                )

                if num_initial > 0 and hasattr(data.geom, "uniform_initial_points"):
                    uniform_initial = data.geom.uniform_initial_points(num_initial)
                else:
                    uniform_initial = np.empty((0, X_orig.shape[1]))

                # Domain points (non-boundary, non-initial) from first seed
                domain_mask = ~(boundary_mask | initial_mask)
                domain_points = X_orig[domain_mask]

                # Create reference X with uniform boundary/initial points
                # For each seed, predictions/losses will be computed at these exact locations
                X = np.vstack([domain_points, uniform_boundary, uniform_initial])

                # Recompute y_true at reference points if needed
                if model.data.soln is not None:
                    y_true = model.data.soln(X)
                elif show_error:
                    # Interpolate y_true to reference points
                    from scipy.interpolate import NearestNDInterpolator

                    interp = NearestNDInterpolator(X_orig, y_true)
                    y_true = interp(X)

                # Collect predictions from all seeds
                all_predictions = []

                for s in seeds:
                    with Capturing() as cap:
                        model, data, model_name, chkpt_path = load_problem(
                            problem_name=problem_name,
                            params_dict=params_dict,
                            load_path=load_path,
                            seed=s,
                        )

                    if show_loss:
                        if use_train_points:
                            # For loss at training points, find closest training points and compute loss there
                            train_x_s = model.data.train_x_all

                            # Compute loss at all training points
                            X_train_tensor = torch.tensor(
                                train_x_s, dtype=torch.float64, requires_grad=True
                            )
                            wrapped_model = ModelWrapper(model.net, model.data.pde, model.data.bcs)
                            residuals = wrapped_model(X_train_tensor)
                            loss_fn = PINNLoss()
                            loss_all = loss_fn(residuals, torch.zeros(train_x_s.shape[0], 1)).sum(
                                axis=1
                            )
                            loss_all = loss_all.detach().numpy()

                            # Map losses to reference points
                            y_pred = np.zeros(len(X))

                            # Domain points: direct copy
                            boundary_mask_s = data.geom.on_boundary(train_x_s)
                            if hasattr(data.geom, "on_initial"):
                                initial_mask_s = data.geom.on_initial(train_x_s)
                            else:
                                initial_mask_s = np.zeros(len(train_x_s), dtype=bool)
                            domain_mask_s = ~(boundary_mask_s | initial_mask_s)

                            y_pred[: len(domain_points)] = loss_all[domain_mask_s]

                            # Boundary points: find closest training boundary point
                            boundary_points_s = train_x_s[boundary_mask_s]
                            boundary_loss_s = loss_all[boundary_mask_s]

                            for i, ref_point in enumerate(uniform_boundary):
                                distances = np.linalg.norm(boundary_points_s - ref_point, axis=1)
                                closest = np.argmin(distances)
                                y_pred[len(domain_points) + i] = boundary_loss_s[closest]

                            # Initial points: find closest training initial point
                            if num_initial > 0:
                                initial_points_s = train_x_s[initial_mask_s]
                                initial_loss_s = loss_all[initial_mask_s]

                                for i, ref_point in enumerate(uniform_initial):
                                    distances = np.linalg.norm(initial_points_s - ref_point, axis=1)
                                    closest = np.argmin(distances)
                                    y_pred[len(domain_points) + len(uniform_boundary) + i] = (
                                        initial_loss_s[closest]
                                    )

                            y_pred = y_pred.reshape(-1, 1)
                        else:
                            # For loss at test points, compute directly at reference points
                            X_tensor = torch.tensor(X, dtype=torch.float64, requires_grad=True)
                            wrapped_model = ModelWrapper(model.net, model.data.pde, model.data.bcs)
                            residuals = wrapped_model(X_tensor)
                            loss_fn = PINNLoss()
                            y_pred = loss_fn(residuals, torch.zeros(X.shape[0], 1)).sum(axis=1)
                            y_pred = y_pred.detach().numpy().reshape(-1, 1)
                    else:
                        # For predictions, compute directly at reference points
                        y_pred = model.predict(X)

                    # for navier stokes simply get the first output component
                    if y_pred.shape[1] > 1:
                        y_pred = y_pred[:, 0].reshape(-1, 1)

                    if show_error:
                        y_pred = np.abs(y_true - y_pred)

                    all_predictions.append(y_pred)

            else:
                # Single seed case
                all_predictions = []
                X = None
                y_true = None

                for s in seeds:
                    with Capturing() as cap:
                        model, data, model_name, chkpt_path = load_problem(
                            problem_name=problem_name,
                            params_dict=params_dict,
                            load_path=load_path,
                            seed=s,
                        )

                    if use_train_points:
                        X = model.data.train_x_all
                        if model.data.soln is not None:
                            y_true = model.data.soln(X)
                        elif show_error:
                            raise ValueError(
                                "Ground truth solution not available for training points."
                            )

                    if X is None:
                        X, y_true = get_X_y_true(model)

                    if show_loss:
                        X_tensor = torch.tensor(X, dtype=torch.float64, requires_grad=True)
                        wrapped_model = ModelWrapper(model.net, model.data.pde, model.data.bcs)
                        residuals = wrapped_model(X_tensor)
                        loss_fn = PINNLoss()
                        y_pred = loss_fn(residuals, torch.zeros(X.shape[0], 1)).sum(axis=1)
                        y_pred = y_pred.detach().numpy().reshape(-1, 1)
                    else:
                        y_pred = model.predict(X)

                    # for navier stokes simply get the first output component
                    if y_pred.shape[1] > 1:
                        y_pred = y_pred[:, 0].reshape(-1, 1)

                    if show_error:
                        y_pred = np.abs(y_true - y_pred)

                    all_predictions.append(y_pred)

            # Average predictions across all seeds
            y_pred = np.mean(all_predictions, axis=0)

            # Determine what to plot
            if show_error:
                plot_data = y_pred
                label = "Absolute Error"
                title_suffix = " (error)"
            elif show_loss:
                plot_data = y_pred
                label = "Residual"
                title_suffix = " (residuals)"
                norm = None
            else:
                plot_data = y_pred
                label = "Predicted Value"
                title_suffix = ""
                norm = None
            if logscale:
                norm = LogNorm()

            sc = ax.scatter(
                *X.T,
                c=plot_data,
                cmap=cmap,
                s=marker_sizes[idx],
                alpha=0.7,
                norm=norm if show_loss else None,
            )

            # Set title based on whether we're averaging or not
            if seed == "all":
                ax.set_title(f"{config_name} (avg {len(all_predictions)} seeds){title_suffix}")
            else:
                ax.set_title(f"{config_name} (seed={seed}){title_suffix}")

            ax.set_xlabel("x")
            if idx == 0:
                ax.set_ylabel("t")

            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(label)

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error loading {config_name}:\n{e!s}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="red",
            )
            if seed == "all":
                ax.set_title(f"{config_name}")
            else:
                ax.set_title(f"{config_name} (seed={seed})")
    return fig, axes


def visualize_influence_comparison(
    target_coords,
    left_term,
    right_term,
    problem_name="drift_diffusion",
    seed=0,  # Can be int or "all" to average across all seeds
    use_test_points=True,
    figsize=(8, 6),
    model_zoo_base="../../model_zoo_cluster",
    influence_zoo_base=None,
    marker_sizes=None,
    marker_size_cross=300,
    absolute=True,
    alpha=None,
):
    """
    Visualize influences for a target point across different model configurations.

    Parameters:
    -----------
    target_coords : tuple
        (x, y) coordinates of the target point
    left_term : str
        Left side term for influence file (e.g., 'total_loss', 'pde_loss', 'bc_0')
    right_term : str
        Right side term for influence file (e.g., 'total_loss', 'output_0')
    problem_name : str
        Name of the problem
    seed : int or "all"
        Random seed for model, or "all" to average across all available seeds
    use_test_points : bool
        If True, find closest point in test set; if False, in training set
    figsize : tuple
        Figure size
    model_zoo_base : str
        Base path to model zoo

    Returns:
    --------
    fig_good, fig_bad : tuple of matplotlib figures
        Two separate figures, one for the "good" configuration and one for the "bad" configuration
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from pinnfluence.utils.defaults import BAD_PROBLEMS, PROBLEMS
    from pinnfluence.utils.utils import load_problem

    configs = {"good": ("PROBLEMS", PROBLEMS), "bad": ("BAD_PROBLEMS", BAD_PROBLEMS)}

    if marker_sizes is None:
        marker_sizes = [50, 50]

    # Create separate figures for each configuration
    figures = {}
    axes_dict = {}

    target_point = np.array(target_coords)

    for idx, (config_name, (dict_name, params_dict_source)) in enumerate(configs.items()):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        figures[config_name] = fig
        axes_dict[config_name] = ax

        # Load model
        params_dict = params_dict_source[problem_name]
        load_path = Path(model_zoo_base) / f"{problem_name}_float64"
        infl_load_path = Path(influence_zoo_base or model_zoo_base) / f"{problem_name}_float64"

        # Determine which seeds to use
        if seed == "all":
            if "seeds" in params_dict:
                seeds = params_dict["seeds"]
            else:
                seeds = list(range(10))
        else:
            seeds = [seed]

        try:
            # For averaging across seeds, create uniform reference points for train set
            if seed == "all":
                # Load first model to get geometry
                with Capturing() as cap:
                    model, data, model_name, chkpt_path = load_problem(
                        problem_name=problem_name,
                        params_dict=params_dict,
                        load_path=load_path,
                        seed=seeds[0],
                        broken=False,
                    )

                # Count boundary and initial points in training set
                train_x_first = data.train_x_all
                boundary_mask = data.geom.on_boundary(train_x_first)
                num_boundary = boundary_mask.sum()

                if hasattr(data.geom, "on_initial"):
                    initial_mask = data.geom.on_initial(train_x_first)
                    num_initial = initial_mask.sum()
                else:
                    num_initial = 0
                    initial_mask = np.zeros(len(train_x_first), dtype=bool)

                # Create uniform reference points
                uniform_boundary = (
                    data.geom.uniform_boundary_points(num_boundary)
                    if num_boundary > 0
                    else np.empty((0, train_x_first.shape[1]))
                )

                if num_initial > 0 and hasattr(data.geom, "uniform_initial_points"):
                    uniform_initial = data.geom.uniform_initial_points(num_initial)
                else:
                    uniform_initial = np.empty((0, train_x_first.shape[1]))

                # Domain points from first seed
                domain_mask = ~(boundary_mask | initial_mask)
                domain_points = train_x_first[domain_mask]

                # Reference training set
                train_x = np.vstack([domain_points, uniform_boundary, uniform_initial])

                # Load influence file from first seed to get test points
                infl_path = infl_load_path.joinpath(f"{model_name}_influence_scores")
                infl_file = infl_path.joinpath(f"influences_{right_term}_{left_term}.npz")

                if not infl_file.exists():
                    ax.text(
                        0.5,
                        0.5,
                        f"Missing:\ninfluences_{right_term}_{left_term}.npz",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=10,
                        color="red",
                    )
                    ax.set_title(f"{config_name.capitalize()}")
                    continue

                infl_data = np.load(infl_file)
                test_x = infl_data["candidate_points"]

                # Find closest point to target
                if use_test_points:
                    points_to_search = test_x
                    point_type = "test"
                else:
                    points_to_search = train_x
                    point_type = "train"

                distances = np.linalg.norm(points_to_search - target_point, axis=1)
                closest_idx = np.argmin(distances)
                closest_point = points_to_search[closest_idx]

                # Collect mapped influences from all seeds
                all_influences = []

                for s in seeds:
                    with Capturing() as cap:
                        model, data, model_name, chkpt_path = load_problem(
                            problem_name=problem_name,
                            params_dict=params_dict,
                            load_path=load_path,
                            seed=s,
                            broken=False,
                        )

                    infl_path = infl_load_path.joinpath(f"{model_name}_influence_scores")
                    infl_file = infl_path.joinpath(f"influences_{right_term}_{left_term}.npz")

                    if not infl_file.exists():
                        continue

                    infl_data = np.load(infl_file)
                    scores = infl_data["scores"] * 1 / len(data.train_x_all)
                    train_x_s = data.train_x_all

                    # Get influences for the closest point
                    if use_test_points:
                        influences_s = scores[closest_idx, :]

                        # Map influences to reference training points (only needed for use_test_points=True)
                        mapped_influences = np.zeros(len(train_x))

                        # Domain points: direct copy
                        boundary_mask_s = data.geom.on_boundary(train_x_s)
                        if hasattr(data.geom, "on_initial"):
                            initial_mask_s = data.geom.on_initial(train_x_s)
                        else:
                            initial_mask_s = np.zeros(len(train_x_s), dtype=bool)
                        domain_mask_s = ~(boundary_mask_s | initial_mask_s)

                        mapped_influences[: len(domain_points)] = influences_s[domain_mask_s]

                        # Boundary points: map to closest boundary point only
                        # Extract only boundary points and their influences from this seed
                        boundary_points_s = train_x_s[boundary_mask_s]
                        boundary_influences_s = influences_s[boundary_mask_s]

                        # For each uniform boundary reference, find closest among boundary points only
                        for i, ref_point in enumerate(uniform_boundary):
                            distances = np.linalg.norm(boundary_points_s - ref_point, axis=1)
                            closest = np.argmin(distances)
                            mapped_influences[len(domain_points) + i] = boundary_influences_s[
                                closest
                            ]

                        # Initial points: map to closest initial point only
                        if num_initial > 0:
                            # Extract only initial points and their influences from this seed
                            initial_points_s = train_x_s[initial_mask_s]
                            initial_influences_s = influences_s[initial_mask_s]

                            # For each uniform initial reference, find closest among initial points only
                            for i, ref_point in enumerate(uniform_initial):
                                distances = np.linalg.norm(initial_points_s - ref_point, axis=1)
                                closest = np.argmin(distances)
                                mapped_influences[
                                    len(domain_points) + len(uniform_boundary) + i
                                ] = initial_influences_s[closest]

                        all_influences.append(mapped_influences)
                    else:
                        # When use_test_points=False, influences are indexed by test points
                        # Test points are the same across seeds, so no mapping needed
                        influences_s = scores[:, closest_idx]
                        all_influences.append(influences_s)

            else:
                # Single seed case
                all_influences = []
                train_x = None
                test_x = None
                closest_point = None
                closest_idx = None

                for s in seeds:
                    with Capturing() as cap:
                        model, data, model_name, chkpt_path = load_problem(
                            problem_name=problem_name,
                            params_dict=params_dict,
                            load_path=load_path,
                            seed=s,
                            broken=False,
                        )

                    if train_x is None:
                        train_x = data.train_x_all

                    infl_path = infl_load_path.joinpath(f"{model_name}_influence_scores")
                    infl_file = infl_path.joinpath(f"influences_{right_term}_{left_term}.npz")

                    if not infl_file.exists():
                        ax.text(
                            0.5,
                            0.5,
                            f"Missing:\ninfluences_{right_term}_{left_term}.npz",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=10,
                            color="red",
                        )
                        ax.set_title(f"{config_name.capitalize()} (seed={s})")
                        continue

                    infl_data = np.load(infl_file)
                    if test_x is None:
                        test_x = infl_data["candidate_points"]
                    scores = infl_data["scores"] * 1 / len(data.train_x_all)

                    if closest_idx is None:
                        if use_test_points:
                            points_to_search = test_x
                        else:
                            points_to_search = train_x

                        distances = np.linalg.norm(points_to_search - target_point, axis=1)
                        closest_idx = np.argmin(distances)
                        closest_point = points_to_search[closest_idx]

                    if use_test_points:
                        influences = scores[closest_idx, :]
                    else:
                        influences = scores[:, closest_idx]

                    all_influences.append(influences)

            if len(all_influences) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No influence files found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="red",
                )
                ax.set_title(f"{config_name.capitalize()}")
                continue

            # Average influences across all seeds
            if absolute:
                all_influences = [np.abs(infl) for infl in all_influences]
            influences = np.mean(all_influences, axis=0)

            if alpha is None:
                if absolute:
                    alpha = (influences / influences.max()) ** 2
                else:
                    alpha = 0.7

            coord_scale = 0.1 if problem_name == "navier_stokes_nd" else 1.0

            # Plot influences on training points
            if use_test_points:
                sc = ax.scatter(
                    *(train_x * coord_scale).T,
                    c=influences,
                    cmap="seismic" if not absolute else "Reds",
                    norm=CenteredNorm() if not absolute else None,
                    s=marker_sizes[idx],
                    alpha=alpha,
                )
            else:
                sc = ax.scatter(
                    *(test_x * coord_scale).T,
                    c=influences,
                    cmap="seismic" if not absolute else "Blues",
                    norm=CenteredNorm() if not absolute else None,
                    s=marker_sizes[idx],
                    alpha=alpha,
                )

            # Mark the target/closest point
            ax.scatter(
                *(np.asarray(closest_point) * coord_scale),
                c="black",
                marker="x",
                s=marker_size_cross,
                linewidths=3,
                zorder=10,
            )

            # Set title based on whether we're averaging or not
            if seed == "all":
                ax.set_title(f"{config_name.capitalize()} (avg {len(all_influences)} seeds)")
            else:
                ax.set_title(f"{config_name.capitalize()} (seed={seed})")

            ax.set_xlabel("x")
            ax.set_ylabel("y" if problem_name in ("poisson_disk", "navier_stokes_nd") else "t")

            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Influence")

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error loading {config_name}:\n{e!s}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="red",
            )
            ax.set_title(f"{config_name.capitalize()} (seed={seed})")

        # Apply tight layout to each figure
        figures[config_name].tight_layout()

    if use_test_points:
        suffix = "Evaluated on TEST point"
    else:
        suffix = "Evaluated on TRAIN point"

    # Return the two separate figures
    return (figures["good"], axes_dict["good"]), (figures["bad"], axes_dict["bad"])


def visualize_self_influence_comparison(
    problem_name="drift_diffusion",
    seed=0,  # Can be int or "all" to average across all seeds
    figsize=(18, 5),
    model_zoo_base="../../model_zoo_cluster",
    influence_zoo_base=None,
    marker_size=30,
):
    """
    Visualize self influences for a target point across different model configurations.

    Parameters:
    -----------
    problem_name : str
        Name of the problem
    seed : int or "all"
        Random seed for model, or "all" to average across all available seeds
    figsize : tuple
        Figure size
    model_zoo_base : str
        Base path to model zoo

    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    configs = {
        "good": ("PROBLEMS", PROBLEMS),
        "bad": ("BAD_PROBLEMS", BAD_PROBLEMS),
        "broken": ("PROBLEMS", PROBLEMS),
    }

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for idx, (config_name, (dict_name, params_dict_source)) in enumerate(configs.items()):
        ax = axes[idx]

        # Load model
        params_dict = params_dict_source[problem_name]
        load_path = Path(model_zoo_base) / f"{problem_name}_float64"
        infl_load_path = Path(influence_zoo_base or model_zoo_base) / f"{problem_name}_float64"

        broken = config_name == "broken"

        # Determine which seeds to use
        if seed == "all":
            if broken and "seeds_broken" in params_dict:
                seeds = params_dict["seeds_broken"]
            elif "seeds" in params_dict:
                seeds = params_dict["seeds"]
            else:
                seeds = list(range(10))
        else:
            seeds = [seed]

        try:
            # For averaging across seeds, create uniform reference points
            if seed == "all":
                # Load first model to get geometry and determine number of points
                with Capturing() as cap:
                    model, data, model_name, chkpt_path = load_problem(
                        problem_name=problem_name,
                        params_dict=params_dict,
                        load_path=load_path,
                        seed=seeds[0],
                        broken=broken,
                    )

                # Count boundary and initial points in training set
                train_x_first = data.train_x_all
                boundary_mask = data.geom.on_boundary(train_x_first)
                num_boundary = boundary_mask.sum()

                if hasattr(data.geom, "on_initial"):
                    initial_mask = data.geom.on_initial(train_x_first)
                    num_initial = initial_mask.sum()
                else:
                    num_initial = 0
                    initial_mask = np.zeros(len(train_x_first), dtype=bool)

                # Create uniform reference points for boundary and initial
                uniform_boundary = (
                    data.geom.uniform_boundary_points(num_boundary)
                    if num_boundary > 0
                    else np.empty((0, train_x_first.shape[1]))
                )

                if num_initial > 0 and hasattr(data.geom, "uniform_initial_points"):
                    uniform_initial = data.geom.uniform_initial_points(num_initial)
                else:
                    uniform_initial = np.empty((0, train_x_first.shape[1]))

                # Domain points (non-boundary, non-initial) from first seed
                domain_mask = ~(boundary_mask | initial_mask)
                domain_points = train_x_first[domain_mask]

                # Combine to create reference training set
                train_x = np.vstack([domain_points, uniform_boundary, uniform_initial])

                # Collect influences mapped to reference points
                all_influences = []

                for s in seeds:
                    with Capturing() as cap:
                        model, data, model_name, chkpt_path = load_problem(
                            problem_name=problem_name,
                            params_dict=params_dict,
                            load_path=load_path,
                            seed=s,
                            broken=broken,
                        )

                    # Load influence file
                    infl_path = infl_load_path.joinpath(f"{model_name}_influence_scores")
                    infl_file = infl_path.joinpath("influences_total_loss_total_loss_self.npz")

                    if not infl_file.exists():
                        continue

                    infl_data = np.load(infl_file)
                    influences_s = infl_data["scores"].diagonal() * 1 / len(data.train_x_all)
                    train_x_s = data.train_x_all

                    # Map influences to reference points
                    mapped_influences = np.zeros(len(train_x))

                    # Domain points: direct copy (same points across all seeds)
                    mapped_influences[: len(domain_points)] = influences_s[domain_mask]

                    # Boundary points: map to closest boundary point only
                    # Extract only boundary points and their influences from this seed
                    boundary_mask_s = data.geom.on_boundary(train_x_s)
                    boundary_points_s = train_x_s[boundary_mask_s]
                    boundary_influences_s = influences_s[boundary_mask_s]

                    # For each uniform boundary reference, find closest among boundary points only
                    for i, ref_point in enumerate(uniform_boundary):
                        distances = np.linalg.norm(boundary_points_s - ref_point, axis=1)
                        closest_idx = np.argmin(distances)
                        mapped_influences[len(domain_points) + i] = boundary_influences_s[
                            closest_idx
                        ]

                    # Initial points: map to closest initial point only
                    if num_initial > 0:
                        # Extract only initial points and their influences from this seed
                        initial_mask_s = data.geom.on_initial(train_x_s)
                        initial_points_s = train_x_s[initial_mask_s]
                        initial_influences_s = influences_s[initial_mask_s]

                        # For each uniform initial reference, find closest among initial points only
                        for i, ref_point in enumerate(uniform_initial):
                            distances = np.linalg.norm(initial_points_s - ref_point, axis=1)
                            closest_idx = np.argmin(distances)
                            mapped_influences[len(domain_points) + len(uniform_boundary) + i] = (
                                initial_influences_s[closest_idx]
                            )

                    all_influences.append(mapped_influences)

            else:
                # Single seed case
                all_influences = []
                train_x = None

                for s in seeds:
                    with Capturing() as cap:
                        model, data, model_name, chkpt_path = load_problem(
                            problem_name=problem_name,
                            params_dict=params_dict,
                            load_path=load_path,
                            seed=s,
                            broken=broken,
                        )

                    if train_x is None:
                        train_x = data.train_x_all

                    # Load influence file
                    infl_path = infl_load_path.joinpath(f"{model_name}_influence_scores")
                    infl_file = infl_path.joinpath("influences_total_loss_total_loss_self.npz")

                    if not infl_file.exists():
                        ax.text(
                            0.5,
                            0.5,
                            "Missing:\ninfluences_total_loss_total_loss_self.npz",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=10,
                            color="red",
                        )
                        ax.set_title(f"{config_name.capitalize()} (seed={s})")
                        continue

                    infl_data = np.load(infl_file)
                    influences_s = infl_data["scores"].diagonal() * 1 / len(data.train_x_all)
                    all_influences.append(influences_s)

            if len(all_influences) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No influence files found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="red",
                )
                ax.set_title(f"{config_name.capitalize()}")
                continue

            # Average influences across all seeds
            influences = np.mean(all_influences, axis=0)

            sc = ax.scatter(
                *train_x.T,
                c=influences,
                cmap="Reds",
                s=marker_size,
                alpha=0.7,
            )

            # Set title based on whether we're averaging or not
            if seed == "all":
                ax.set_title(f"{config_name.capitalize()} (avg {len(all_influences)} seeds)")
            else:
                ax.set_title(f"{config_name.capitalize()} (seed={seed})")

            ax.set_xlabel("x")
            if idx == 0:
                ax.set_ylabel("t")

            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Influence")

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error loading {config_name}:\n{e!s}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="red",
            )
            if seed == "all":
                ax.set_title(f"{config_name.capitalize()}")
            else:
                ax.set_title(f"{config_name.capitalize()} (seed={seed})")

    fig.suptitle("Self influence (total loss)", fontsize=14, y=1.02)
    plt.tight_layout()

    return fig, axes


def visualize_loss_fractions(
    problem_name="drift_diffusion",
    left_term="output_0",
    seed=0,  # Can be int, list of ints, or "all"
    model_zoo_base="../../model_zoo_cluster",
    influence_zoo_base=None,
    figsize=(15, 12),
    cmap="jet",
    markersize=30,
    config_name="good",
    vmin=0.0,
    vmax=1.0,
):
    """
    Visualize the fraction of influence that each loss term contributes to a given test point.

    Since influences are additive, for a fixed left side (e.g., output_0 or total_loss),
    the total influence equals the sum of influences from each right-hand side loss term.
    This function computes and visualizes what fraction each loss term contributes.

    Parameters:
    -----------
    problem_name : str
        Name of the problem (e.g., 'drift_diffusion', 'burgers')
    left_term : str
        Left side term to analyze (e.g., 'output_0', 'total_loss', 'pde_loss', 'bc_loss')
    seed : int, list, or "all"
        Random seed(s) for model. If "all", averages across all available seeds.
        If list, averages across specified seeds.
    model_zoo_base : str
        Base path to model zoo
    figsize : tuple
        Figure size
    cmap : str
        Colormap for fraction visualization
    markersize : int
        Size of scatter plot markers
    config_name : str
        Configuration to use: "good", "bad", or "broken"
    vmin : float
        Minimum value for colorbar (default: 0.0)
    vmax : float
        Maximum value for colorbar (default: 1.0)

    Returns:
    --------
    fig, axes : matplotlib figure and axes
    mean_fractions : dict
        Dictionary containing mean fraction for each loss term
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    from pinnfluence.utils.defaults import BAD_PROBLEMS, PROBLEMS
    from pinnfluence.utils.utils import Capturing, load_problem

    # Select configuration
    configs = {
        "good": ("PROBLEMS", PROBLEMS),
        "bad": ("BAD_PROBLEMS", BAD_PROBLEMS),
        "broken": ("PROBLEMS", PROBLEMS),
    }

    if config_name not in configs:
        raise ValueError(f"config_name must be one of {list(configs.keys())}")

    dict_name, params_dict_source = configs[config_name]
    params_dict = params_dict_source[problem_name]
    load_path = Path(model_zoo_base) / f"{problem_name}_float64"
    infl_load_path = Path(influence_zoo_base or model_zoo_base) / f"{problem_name}_float64"

    broken = config_name == "broken"

    # Determine which seeds to use
    if seed == "all":
        if broken and "seeds_broken" in params_dict:
            seeds = params_dict["seeds_broken"]
        elif "seeds" in params_dict:
            seeds = params_dict["seeds"]
        else:
            seeds = list(range(10))
    elif isinstance(seed, list):
        seeds = seed
    else:
        seeds = [seed]

    print(f"Using seeds: {seeds}")

    # Load first model to get structure
    with Capturing() as cap:
        model, data, model_name, chkpt_path = load_problem(
            problem_name=problem_name,
            params_dict=params_dict,
            load_path=load_path,
            seed=seeds[0],
            broken=broken,
        )

    # Get influence scores directory for first seed
    infl_dir = infl_load_path / f"{model_name}_influence_scores"

    # Load a sample file to get num_pdes and num_bcs
    sample_files = list(infl_dir.glob("influences_*.npz"))
    if not sample_files:
        raise FileNotFoundError(f"No influence files found in {infl_dir}")

    sample_data = np.load(sample_files[0])
    num_pdes = int(sample_data["num_pdes"])
    num_bcs = int(sample_data["num_bcs"])

    print(f"Problem has {num_pdes} PDE term(s) and {num_bcs} BC term(s)")

    # Build list of all right-hand side loss terms
    loss_terms = []

    # Add individual PDE terms
    for pde_idx in range(num_pdes):
        loss_terms.append(f"pde_{pde_idx}")

    # Add individual BC terms
    for bc_idx in range(num_bcs):
        loss_terms.append(f"bc_{bc_idx}")

    print(f"Loss terms to analyze: {loss_terms}")

    # Collect influence scores across all seeds
    all_influences = {term: [] for term in loss_terms}
    test_x = None
    test_mask = None  # Mask for filtering test points based on left_term

    for s in seeds:
        with Capturing() as cap:
            model_s, data_s, model_name_s, chkpt_path_s = load_problem(
                problem_name=problem_name,
                params_dict=params_dict,
                load_path=load_path,
                seed=s,
                broken=broken,
            )

        infl_dir_s = infl_load_path / f"{model_name_s}_influence_scores"

        for term in loss_terms:
            # Load influence file for this term
            infl_file = infl_dir_s / f"influences_{term}_{left_term}.npz"

            if not infl_file.exists():
                print(f"Warning: {infl_file} not found, skipping")
                continue

            infl_data = np.load(infl_file)

            if test_x is None:
                test_x = infl_data["candidate_points"]

                # Create mask for test points based on left_term
                # If left_term is a BC, only include points on that BC
                if left_term.startswith("bc_"):
                    bc_idx = int(left_term.split("_")[1])
                    if bc_idx < len(model_s.data.bcs):
                        bc = model_s.data.bcs[bc_idx]
                        # Check which test points are on this BC
                        if bc_idx == 0 and hasattr(model_s.data.geom, "on_initial"):
                            test_mask = model_s.data.geom.on_initial(test_x)
                        else:
                            test_mask = model_s.data.bcs[bc_idx].on_boundary(
                                test_x, np.ones(len(test_x), dtype=bool)
                            )
                        print(
                            f"Filtering to {test_mask.sum()} test points on {left_term} (out of {len(test_x)})"
                        )
                    else:
                        raise ValueError(f"BC index {bc_idx} out of range")
                else:
                    # Use all test points
                    test_mask = np.ones(len(test_x), dtype=bool)

            # Extract influence scores
            scores = infl_data["scores"]
            all_influences[term].append(scores)

    if test_x is None:
        raise RuntimeError("Could not load any influence files")

    # Average across seeds for each loss term
    avg_influences = {}
    for term in loss_terms:
        if len(all_influences[term]) > 0:
            avg_influences[term] = np.mean(all_influences[term], axis=0)
        else:
            print(f"Warning: No data found for {term}, skipping")

    if len(avg_influences) == 0:
        raise RuntimeError("No valid influence data found")

    # Apply test mask to filter test points (if BC-specific)
    test_x_filtered = test_x[test_mask]
    avg_influences_filtered = {term: scores[test_mask] for term, scores in avg_influences.items()}

    # Compute denominator (sum of absolute influences from all terms)
    denominator = np.zeros(test_x_filtered.shape[0])
    for term in avg_influences_filtered:
        denominator += np.abs(avg_influences_filtered[term]).sum(axis=1)

    # Avoid division by zero
    denominator = np.maximum(denominator, 1e-12)

    # Compute fractions for each loss term
    fractions = {}
    mean_fractions = {}

    for term in avg_influences_filtered:
        frac = np.abs(avg_influences_filtered[term]).sum(axis=1) / denominator
        fractions[term] = frac
        mean_fractions[term] = float(frac.mean())

    # Verify fractions sum to 1 (within numerical precision)
    total_fraction = sum(fractions.values())
    print(f"\nFraction sum verification: {total_fraction.mean():.6f} (should be ~1.0)")

    # Print mean fractions
    print("\nMean fractions for each loss term:")
    for term in sorted(mean_fractions.keys()):
        print(f"  {term:15s}: {mean_fractions[term]:.4f}")

    # Create visualization
    n_terms = len(fractions)
    ncols = min(3, n_terms)
    nrows = (n_terms + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Plot fraction for each loss term
    norm = TwoSlopeNorm(vcenter=0.5, vmin=vmin, vmax=vmax)

    for idx, (term, frac) in enumerate(sorted(fractions.items())):
        ax = axes[idx]

        sc = ax.scatter(
            test_x_filtered[:, 0],
            test_x_filtered[:, 1],
            c=frac,
            cmap=cmap,
            norm=norm,
            s=markersize,
            alpha=0.7,
        )

        ax.set_title(f"{term}\n(mean: {mean_fractions[term]:.3f})", fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("t")

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Fraction")

    # Hide unused subplots
    for idx in range(n_terms, len(axes)):
        axes[idx].axis("off")

    # Add overall title
    if len(seeds) == 1:
        seed_str = f"seed={seeds[0]}"
    else:
        seed_str = f"avg over {len(seeds)} seeds"

    # Add info about test point filtering
    if left_term.startswith("bc_"):
        filter_str = f" (filtered to {len(test_x_filtered)}/{len(test_x)} points on {left_term})"
    else:
        filter_str = ""

    fig.suptitle(
        f"Loss Fraction Analysis: {left_term} ← loss terms{filter_str}\n{config_name} ({seed_str})",
        fontsize=14,
        y=0.995,
    )

    plt.tight_layout()

    return fig, axes, mean_fractions


def visualize_loss_fractions_lineplot(
    problem_name="drift_diffusion",
    left_term="output_0",
    seed=0,  # Can be int, list of ints, or "all"
    model_zoo_base="../../model_zoo_cluster",
    influence_zoo_base=None,
    figsize=(8, 6),
    config_name="good",
    dimension="auto",  # "auto", "time", "space", "radial", or index (0 or 1)
    num_bins=50,  # Number of bins for grouping points along the dimension
    plot_wrt="test",  # "test" or "train" - which points to bin along the dimension
):
    """
    Visualize loss fractions as line plots along a specified dimension.

    For 2D problems (x, t), this averages across one dimension and shows how
    fractions evolve along the other (typically time). For spatial problems,
    can compute radial distance from center.

    Parameters:
    -----------
    problem_name : str
        Name of the problem (e.g., 'drift_diffusion', 'burgers')
    left_term : str
        Left side term to analyze (e.g., 'output_0', 'total_loss')
    seed : int, list, or "all"
        Random seed(s) for model
    model_zoo_base : str
        Base path to model zoo
    figsize : tuple
        Figure size
    config_name : str
        Configuration to use: "good", "bad", or "broken"
    dimension : str or int
        Dimension to plot along:
        - "auto": automatically detect (time for time-dependent, radial for spatial)
        - "time": time dimension (typically index 1)
        - "space": space dimension (typically index 0)
        - "radial": radial distance from center
        - 0 or 1: explicit dimension index
    num_bins : int
        Number of bins for grouping points along the dimension
    plot_wrt : str
        Which points to bin along the dimension:
        - "test": bin test points (default)
        - "train": bin training points

    Returns:
    --------
    fig, ax : matplotlib figure and axes
    results : dict
        Dictionary containing mean fractions, std fractions, mean coherence, and std coherence
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from pinnfluence.utils.defaults import BAD_PROBLEMS, PROBLEMS
    from pinnfluence.utils.utils import Capturing, load_problem

    # Select configuration
    configs = {
        "good": ("PROBLEMS", PROBLEMS),
        "bad": ("BAD_PROBLEMS", BAD_PROBLEMS),
        "broken": ("PROBLEMS", PROBLEMS),
        "soap": ("PROBLEMS", PROBLEMS),
        "nncg": ("PROBLEMS", PROBLEMS),
    }

    if config_name not in configs:
        raise ValueError(f"config_name must be one of {list(configs.keys())}")

    dict_name, params_dict_source = configs[config_name]
    params_dict = dict(params_dict_source[problem_name])
    if config_name == "soap":
        params_dict["optimizer"] = "SOAP"
        params_dict["n_iterations_lbfgs"] = 0
    elif config_name == "nncg":
        params_dict["optimizer"] = "NNCG"
    load_path = Path(model_zoo_base) / f"{problem_name}_float64"
    infl_load_path = Path(influence_zoo_base or model_zoo_base) / f"{problem_name}_float64"

    broken = config_name == "broken"

    # Determine which seeds to use
    if seed == "all":
        if broken and "seeds_broken" in params_dict:
            seeds = params_dict["seeds_broken"]
        elif "seeds" in params_dict:
            seeds = params_dict["seeds"]
        else:
            seeds = list(range(10))
    elif isinstance(seed, list):
        seeds = seed
    else:
        seeds = [seed]

    print(f"Using seeds: {seeds}")

    # Load first model to get structure
    with Capturing() as cap:
        model, data, model_name, chkpt_path = load_problem(
            problem_name=problem_name,
            params_dict=params_dict,
            load_path=load_path,
            seed=seeds[0],
            broken=broken,
        )

    # Get influence scores directory for first seed
    infl_dir = infl_load_path / f"{model_name}_influence_scores"

    print(f"Loading influence scores from: {infl_dir}")

    # Load a sample file to get num_pdes and num_bcs
    sample_files = list(infl_dir.glob("influences_*.npz"))
    if not sample_files:
        raise FileNotFoundError(f"No influence files found in {infl_dir}")

    sample_data = np.load(sample_files[0])
    num_pdes = int(sample_data["num_pdes"])
    num_bcs = int(sample_data["num_bcs"])

    print(f"Problem has {num_pdes} PDE term(s) and {num_bcs} BC term(s)")

    # Build list of all right-hand side loss terms
    loss_terms = []
    for pde_idx in range(num_pdes):
        loss_terms.append(f"pde_{pde_idx}")
    for bc_idx in range(num_bcs):
        loss_terms.append(f"bc_{bc_idx}")

    print(f"Loss terms to analyze: {loss_terms}")

    # Collect influence scores across all seeds
    all_influences = {term: [] for term in loss_terms}
    test_x = None
    test_mask = None
    train_x = None

    for s in seeds:
        with Capturing() as cap:
            model_s, data_s, model_name_s, chkpt_path_s = load_problem(
                problem_name=problem_name,
                params_dict=params_dict,
                load_path=load_path,
                seed=s,
                broken=broken,
            )

        infl_dir_s = infl_load_path / f"{model_name_s}_influence_scores"

        for term in loss_terms:
            infl_file = infl_dir_s / f"influences_{term}_{left_term}.npz"

            if not infl_file.exists():
                if term == "pde_0" and problem_name != "navier_stokes_nd":
                    infl_file = infl_dir_s / f"influences_pde_loss_{left_term}.npz"
                    if not infl_file.exists():
                        print(f"Warning: {infl_file} not found, skipping")
                        continue

                elif term == "bc_0" and num_bcs == 1:
                    infl_file = infl_dir_s / f"influences_bc_loss_{left_term}.npz"
                    if not infl_file.exists():
                        print(f"Warning: {infl_file} not found, skipping")
                        continue
                else:
                    print(f"Warning: {infl_file} not found, skipping")
                    continue

            infl_data = np.load(infl_file)

            if test_x is None:
                test_x = infl_data["candidate_points"]
                train_x = data_s.train_x_all

                # Create mask for test points based on left_term
                # If left_term is a BC, only include points on that BC
                if left_term.startswith("bc_"):
                    bc_idx = int(left_term.split("_")[1])
                    if bc_idx < len(model_s.data.bcs):
                        # Check which test points are on this BC
                        if bc_idx == 0 and hasattr(data_s.geom, "on_initial"):
                            test_mask = data_s.geom.on_initial(test_x)
                        else:
                            test_mask = data_s.bcs[bc_idx].on_boundary(
                                test_x, data_s.bcs[bc_idx].geom.on_boundary(test_x)
                            )
                        print(
                            f"Filtering to {test_mask.sum()} test points on {left_term} (out of {len(test_x)})"
                        )
                    else:
                        raise ValueError(f"BC index {bc_idx} out of range")
                else:
                    # Use all test points
                    test_mask = np.ones(len(test_x), dtype=bool)

            scores = infl_data["scores"]
            all_influences[term].append(scores)

    if test_x is None:
        raise RuntimeError("Could not load any influence files")

    if train_x is None:
        raise RuntimeError("Could not load training points")

    # Apply test mask to filter test points (only relevant for test-based plotting)
    if plot_wrt == "test":
        test_x_filtered = test_x[test_mask]
        all_influences_filtered = {
            term: [scores[test_mask] for scores in all_influences[term]] for term in loss_terms
        }
        num_points = test_x_filtered.shape[0]
        points_x = test_x_filtered
    else:  # plot_wrt == "train"
        # For training points, we don't filter test points but use all influences
        all_influences_filtered = all_influences
        num_points = train_x.shape[0]
        points_x = train_x

    num_seeds = len(seeds)

    # Compute fractions and pointwise coherence per seed, then average
    all_fractions = {term: [] for term in loss_terms}
    all_coherence = []

    for s_idx in range(num_seeds):
        # Get shape information from first available term
        shape_info = None
        for term in loss_terms:
            if len(all_influences_filtered[term]) > s_idx:
                shape_info = all_influences_filtered[term][s_idx].shape
                break

        if shape_info is None:
            continue

        if plot_wrt == "test":
            # Shape: (num_test_points, n_train)
            num_test_points, n_train = shape_info

            # Compute denominator (sum of absolute influences) for this seed
            # Shape: (num_test_points,) - summed over training points
            denom_s = np.zeros(num_test_points)
            for term in loss_terms:
                if len(all_influences_filtered[term]) > s_idx:
                    denom_s += np.abs(all_influences_filtered[term][s_idx]).sum(axis=1)
            denom_s = np.maximum(denom_s, 1e-12)

            # Compute pointwise coherence: for each test point, compute coherence
            # at each training point, then average over training points
            signed_sum_pointwise = np.zeros((num_test_points, n_train))
            abs_sum_pointwise = np.zeros((num_test_points, n_train))

            for term in loss_terms:
                if len(all_influences_filtered[term]) > s_idx:
                    signed_sum_pointwise += all_influences_filtered[term][s_idx]
                    abs_sum_pointwise += np.abs(all_influences_filtered[term][s_idx])

            abs_sum_pointwise = np.maximum(abs_sum_pointwise, 1e-12)

            # Pointwise coherence: |signed| / |abs| at each (test, train) pair
            # Then average over training points for each test point
            pointwise_coherence = np.abs(signed_sum_pointwise) / abs_sum_pointwise
            coherence_s = pointwise_coherence.mean(axis=1)
            all_coherence.append(coherence_s)

            # Compute fractions for this seed
            for term in loss_terms:
                if len(all_influences_filtered[term]) > s_idx:
                    frac_s = np.abs(all_influences_filtered[term][s_idx]).sum(axis=1) / denom_s
                    all_fractions[term].append(frac_s)

        else:  # plot_wrt == "train"
            # New logic: compute fractions per training point (sum over test points)
            # Shape: (num_test_points, n_train)
            num_test_points, n_train = shape_info

            # Compute denominator (sum of absolute influences) for this seed
            # Shape: (n_train,) - summed over test points
            denom_s = np.zeros(n_train)
            for term in loss_terms:
                if len(all_influences_filtered[term]) > s_idx:
                    denom_s += np.abs(all_influences_filtered[term][s_idx]).sum(axis=0)
            denom_s = np.maximum(denom_s, 1e-12)

            # Compute pointwise coherence: for each training point, compute coherence
            # at each test point, then average over test points
            signed_sum_pointwise = np.zeros((num_test_points, n_train))
            abs_sum_pointwise = np.zeros((num_test_points, n_train))

            for term in loss_terms:
                if len(all_influences_filtered[term]) > s_idx:
                    signed_sum_pointwise += all_influences_filtered[term][s_idx]
                    abs_sum_pointwise += np.abs(all_influences_filtered[term][s_idx])

            abs_sum_pointwise = np.maximum(abs_sum_pointwise, 1e-12)

            # Pointwise coherence: |signed| / |abs| at each (test, train) pair
            # Then average over test points for each training point
            pointwise_coherence = np.abs(signed_sum_pointwise) / abs_sum_pointwise
            coherence_s = pointwise_coherence.mean(axis=0)  # shape: (n_train,)
            all_coherence.append(coherence_s)

            # Compute fractions for this seed
            for term in loss_terms:
                if len(all_influences_filtered[term]) > s_idx:
                    frac_s = np.abs(all_influences_filtered[term][s_idx]).sum(axis=0) / denom_s
                    all_fractions[term].append(frac_s)

    # Average fractions and coherence across seeds
    fractions = {}
    fractions_std = {}
    mean_fractions = {}
    std_fractions = {}

    for term in loss_terms:
        if len(all_fractions[term]) > 0:
            fractions[term] = np.mean(all_fractions[term], axis=0)
            fractions_std[term] = np.std(all_fractions[term], axis=0)
            mean_fractions[term] = float(fractions[term].mean())
            std_fractions[term] = float(np.mean(fractions_std[term]))
        else:
            print(f"Warning: No data found for {term}, skipping")

    coherence = np.mean(all_coherence, axis=0)
    coherence_std = np.std(all_coherence, axis=0)
    mean_coherence = float(coherence.mean())
    std_coherence = float(np.mean(coherence_std))

    if len(fractions) == 0:
        raise RuntimeError("No valid influence data found")

    # Determine which dimension to use
    if dimension == "auto":
        # Auto-detect: use time for time-dependent problems, radial for spatial
        time_problems = [
            "burgers",
            "diffusion",
            "drift_diffusion",
            "allen_cahn",
            "wave",
        ]
        if problem_name in time_problems:
            dimension = 1  # Time is typically second dimension
            dim_label = "Time"
        elif problem_name == "navier_stokes_nd":
            dimension = 0
            dim_label = "Space"
        else:
            dimension = "radial"
            dim_label = "Radial Distance from Center"
    elif dimension == "time":
        dimension = 1
        dim_label = "Time"
    elif dimension == "space":
        dimension = 0
        dim_label = "Space"
    elif dimension == "radial":
        dim_label = "Radial Distance from Center"
    else:
        # Assume it's an integer index
        dim_label = f"Dimension {dimension}"

    # Compute the coordinate along the chosen dimension
    if dimension == "radial":
        # Distance from center
        center = np.array([0.0, 0.0])  # Disk center at origin
        coord = np.sqrt(((points_x - center) ** 2).sum(axis=1))
    else:
        # Use specified dimension
        coord = points_x[:, dimension]

    # Sort by coordinate for cleaner line plots
    sort_idx = np.argsort(coord)
    coord_sorted = coord[sort_idx]
    fractions_sorted = {term: frac[sort_idx] for term, frac in fractions.items()}
    fractions_std_sorted = {term: std[sort_idx] for term, std in fractions_std.items()}
    coherence_sorted = coherence[sort_idx]
    coherence_std_sorted = coherence_std[sort_idx]

    # Bin the data to reduce noise
    bins = np.linspace(coord_sorted.min(), coord_sorted.max(), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    binned_fractions = {term: np.zeros(num_bins) for term in fractions}
    binned_fractions_std = {term: np.zeros(num_bins) for term in fractions}
    binned_coherence = np.zeros(num_bins)
    binned_coherence_std = np.zeros(num_bins)

    for i in range(num_bins):
        mask = (coord_sorted >= bins[i]) & (coord_sorted < bins[i + 1])
        if mask.sum() > 0:
            for term in fractions:
                binned_fractions[term][i] = fractions_sorted[term][mask].mean()
                binned_fractions_std[term][i] = fractions_std_sorted[term][mask].mean()
            binned_coherence[i] = coherence_sorted[mask].mean()
            binned_coherence_std[i] = coherence_std_sorted[mask].mean()

    if problem_name == "navier_stokes_nd":
        bin_centers = bin_centers * 0.1

    # Create line plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the cancellation factor (1 - coherence)
    ax.plot(
        bin_centers,
        1 - binned_coherence,
        color="gray",
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
        label=f"Cancellation $\\kappa$ (mean: {1 - mean_coherence:.3f})",
    )

    # # Plot each loss term
    for term in sorted(binned_fractions.keys()):
        ax.plot(
            bin_centers,
            binned_fractions[term],
            label=f"{loss_term_names[problem_name][term]} (mean: {mean_fractions[term]:.3f})",
            linewidth=2,
        )
        ax.fill_between(
            bin_centers,
            binned_fractions[term] - binned_fractions_std[term],
            binned_fractions[term] + binned_fractions_std[term],
            alpha=0.2,
        )

    ax.set_xlabel(dim_label)
    ax.set_ylabel("Loss-Fraction")
    ax.set_ylim(0, 1)
    # ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # Add overall title
    if len(seeds) == 1:
        seed_str = f"seed={seeds[0]}"
    else:
        seed_str = f"avg over {len(seeds)} seeds"

    if plot_wrt == "test" and left_term.startswith("bc_"):
        filter_str = (
            f" (filtered to {len(test_x_filtered)}/{len(test_x)} test points on {left_term})"
        )
    else:
        filter_str = ""

    points_type = "test points" if plot_wrt == "test" else "training points"
    ax.set_title(
        f"Loss Fraction vs {dim_label} ({points_type}): {loss_term_names[problem_name][left_term]} ← loss terms{filter_str}\n{config_name} ({seed_str})",
        fontsize=14,
    )

    plt.tight_layout()

    # Compile results
    results = {
        "mean_fractions": mean_fractions,
        "std_fractions": std_fractions,
        "mean_coherence": mean_coherence,
        "std_coherence": std_coherence,
        "binned_fractions": binned_fractions,
        "binned_fractions_std": binned_fractions_std,
        "binned_coherence": binned_coherence,
        "bin_centers": bin_centers,
    }

    return fig, ax, results


def create_horizontal_legend(
    colors,
    labels,
    linestyles=None,
    linewidths=None,
    markers=None,
    figsize=(16, 0.5),
    ncol=None,
    frameon=False,
    loc="center",
    save_path=None,
    dpi=300,
):
    """
    Create a standalone horizontal legend figure.

    Parameters:
    -----------
    colors : list
        List of colors for each legend entry
    labels : list
        List of labels for each legend entry
    linestyles : list, optional
        List of linestyles (e.g., '-', '--', '-.', ':'). If None, uses solid lines.
    linewidths : list or float, optional
        List of linewidths or single value. Default is 2.
    markers : list, optional
        List of markers (e.g., 'o', 's', '^'). If None, no markers.
    figsize : tuple, optional
        Figure size (width, height). Default is (8, 0.5).
    fontsize : int, optional
        Font size for legend text. Default is 12.
    ncol : int, optional
        Number of columns. If None, uses len(labels).
    frameon : bool, optional
        Whether to draw frame around legend. Default is False.
    loc : str, optional
        Location of legend. Default is 'center'.
    save_path : str, optional
        If provided, saves the figure to this path.
    dpi : int, optional
        DPI for saved figure. Default is 300.

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects

    Examples:
    ---------
    # Simple line legend
    create_horizontal_legend(
        colors=['red', 'blue', 'green'],
        labels=['Model A', 'Model B', 'Model C'],
        save_path='legend.pdf'
    )

    # With different linestyles
    create_horizontal_legend(
        colors=['red', 'blue', 'green'],
        labels=['Train', 'Val', 'Test'],
        linestyles=['-', '--', '-.'],
        linewidths=3
    )

    # With markers
    create_horizontal_legend(
        colors=['red', 'blue'],
        labels=['Method 1', 'Method 2'],
        markers=['o', 's'],
        linestyles=['-', '-']
    )
    """
    # Set defaults
    if linestyles is None:
        linestyles = ["-"] * len(labels)
    if linewidths is None:
        linewidths = [2] * len(labels)
    elif isinstance(linewidths, (int, float)):
        linewidths = [linewidths] * len(labels)
    if markers is None:
        markers = [None] * len(labels)
    if ncol is None:
        ncol = len(labels)

    # Create legend handles
    handles = []
    for color, label, ls, lw, marker in zip(colors, labels, linestyles, linewidths, markers):
        handle = Line2D(
            [0],
            [0],
            color=color,
            linestyle=ls,
            linewidth=lw,
            marker=marker,
            markersize=8,
            label=label,
        )
        handles.append(handle)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # Create legend
    legend = ax.legend(handles=handles, loc=loc, ncol=ncol, frameon=frameon)

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)

    return fig, ax
