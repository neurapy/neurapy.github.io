"""
This script precalculates the influence scores for a given model and saves them to a .npz file.

This is useful for evaluation of scoring and sampling strategies without the need to recalculate the influence scores
for each experiment.

Usage:
    python -m pinnfluence_resampling.precalculate_influences [options]

    Use --help to see all available options.
"""

import os
import tempfile

import deepxde as dde
import numpy as np

from . import problem_factory
from .utils.models import ModelWrapper, NetPredWrapper, PINNLoss
from .utils.parse_args import parse_precalculate_args as parse_args
from .utils.sampling import (
    calculate_influence_scores,
    instantiate_IF,
    sample_random_points,
)
from .utils.utils import set_default_device


def parse_loss_term(term_str, model, num_pdes, num_bcs):
    """
    Parses a loss term string and returns configuration for ModelWrapper/NetPredWrapper and PINNLoss.

    Args:
        term_str: String like "total_loss", "pde_loss", "bc_loss", "pde_0", "bc_1", "output_0"
        model: The PINN model object
        num_pdes: Number of PDE terms
        num_bcs: Number of boundary condition terms

    Returns:
        Tuple of (model_wrapper, loss_fn, test_reduction_type)
    """
    if term_str == "total_loss":
        model_wrapper = ModelWrapper(
            net=model.net, pde=model.data.pde, bcs=model.data.bcs, include_pde=True
        )
        loss_fn = PINNLoss(include_all_losses=True)
        test_reduction_type = "none"
    elif term_str == "pde_loss":
        model_wrapper = ModelWrapper(net=model.net, pde=model.data.pde, bcs=[], include_pde=True)
        loss_fn = PINNLoss(include_all_losses=True)
        test_reduction_type = "none"
    elif term_str == "bc_loss":
        model_wrapper = ModelWrapper(
            net=model.net, pde=model.data.pde, bcs=model.data.bcs, include_pde=False
        )
        loss_fn = PINNLoss(include_all_losses=True)
        test_reduction_type = "none"
    elif term_str.startswith("pde_"):
        try:
            pde_idx = int(term_str.split("_")[1])
            if pde_idx >= num_pdes:
                raise ValueError(f"PDE index {pde_idx} out of range (0-{num_pdes - 1})")
            model_wrapper = ModelWrapper(
                net=model.net, pde=model.data.pde, bcs=[], include_pde=True
            )
            loss_fn = PINNLoss(include_all_losses=False, include_specific_ids=[pde_idx])
            test_reduction_type = "none"
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Invalid PDE term format: {term_str}. Expected 'pde_N' where N is an integer"
            ) from e
    elif term_str.startswith("bc_"):
        try:
            bc_idx = int(term_str.split("_")[1])
            if bc_idx >= num_bcs:
                raise ValueError(f"BC index {bc_idx} out of range (0-{num_bcs - 1})")
            model_wrapper = ModelWrapper(
                net=model.net, pde=model.data.pde, bcs=model.data.bcs, include_pde=False
            )
            loss_fn = PINNLoss(include_all_losses=False, include_specific_ids=[bc_idx])
            test_reduction_type = "none"
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Invalid BC term format: {term_str}. Expected 'bc_N' where N is an integer"
            ) from e
    elif term_str.startswith("output_"):
        try:
            output_idx = int(term_str.split("_")[1])
            n_outputs = model.net.linears[-1].out_features
            if output_idx >= n_outputs:
                raise ValueError(f"Output index {output_idx} out of range (0-{n_outputs - 1})")
            model_wrapper = NetPredWrapper(model.net, pred_idx=output_idx)
            loss_fn = None
            test_reduction_type = "none"
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Invalid output term format: {term_str}. Expected 'output_N' where N is an integer"
            ) from e
    else:
        raise ValueError(
            f"Unknown loss term: {term_str}. Valid options: total_loss, pde_loss, bc_loss, pde_N, bc_N, output_N"
        )

    return model_wrapper, loss_fn, test_reduction_type


def compute_single_influence(
    IF_instance, left_term, right_term, model, candidate_points, batch_size, num_pdes, num_bcs
):
    """
    Computes a single influence matrix for the given left/right configuration.

    Args:
        IF_instance: The instantiated influence function object
        left_term: String describing the test loss term (left side)
        right_term: String describing the training loss term (right side)
        model: The PINN model object
        candidate_points: The candidate points to compute influence for
        batch_size: Batch size for computation
        num_pdes: Number of PDE terms
        num_bcs: Number of BC terms

    Returns:
        Influence scores as numpy array
    """
    # Configure right side (training loss)
    right_model, right_loss, _ = parse_loss_term(right_term, model, num_pdes, num_bcs)
    IF_instance.model = right_model
    IF_instance.loss_fn = right_loss

    # Configure left side (test loss)
    left_model, left_loss, left_reduction = parse_loss_term(left_term, model, num_pdes, num_bcs)
    IF_instance.model_test = left_model
    IF_instance.test_loss_fn = left_loss
    IF_instance.test_reduction_type = left_reduction

    print(f"Computing influence: right={right_term}, left={left_term}")
    infl_scores = calculate_influence_scores(
        tda_instance=IF_instance,
        candidate_points=candidate_points,
        batch_size=batch_size,
        show_progress=True,
    ).astype(np.float32)

    return infl_scores


def legacy_main(
    model,
    data,
    model_name,
    save_path,
    use_holdout_test,
    precalc_infl_sample_uniformly,
    n_candidate_points,
    scoring_method,
    use_train_set,
    wrt_individual_loss_terms,
):
    """
    Legacy implementation: computes all influence matrices and saves to monolithic file.
    """
    # reproduce sampling of Scorer class
    dde.config.set_random_seed(42)

    if not use_holdout_test:
        if precalc_infl_sample_uniformly:
            candidate_points = data.geom.uniform_points(n_candidate_points)
        else:
            candidate_points = sample_random_points(
                geometry=model.data.geom, num_points=n_candidate_points
            )
    else:
        # Use the holdout test set for candidate points
        # This is useful for evaluating the scoring strategy on the test set
        # without the need to recalculate the influence scores
        candidate_points = model.data.holdout_test_x

    batch_size = 1024

    if scoring_method != "PINNfluence":
        raise ValueError("Legacy precomputation only supports PINNfluence")
    else:
        if os.path.exists(f"{save_path}/{model_name}_influence_scores.npz"):
            print(
                f"Influence scores for model {model_name} already exist at {save_path}/{model_name}_influence_scores.npz. Skipping precalculation."
            )
        else:
            print(f"Calculating influence scores for model {model_name}...")

            IF_instance = instantiate_IF(
                model,
                model_name=None,
                use_train_set=use_train_set,
                show_progress=False,
                prefer_load_R=False,
            )

            # influence w.r.t. output
            n_outputs = model.net.linears[-1].out_features
            if n_outputs > 1:
                infl_scores_outputs = []
                for i in range(n_outputs):
                    IF_instance.model_test = NetPredWrapper(model.net, pred_idx=i)
                    IF_instance.test_loss_fn = None

                    print(f"Calculating influence scores w.r.t. output {i}")
                    infl_scores_i = calculate_influence_scores(
                        tda_instance=IF_instance,
                        candidate_points=candidate_points,
                        batch_size=batch_size,
                    ).astype(np.float32)
                    infl_scores_outputs.append(infl_scores_i)
            else:
                IF_instance.model_test = NetPredWrapper(model.net)
                IF_instance.test_loss_fn = None

                print("Calculating influence scores w.r.t. output")
                infl_scores_i = calculate_influence_scores(
                    tda_instance=IF_instance,
                    candidate_points=candidate_points,
                    batch_size=batch_size,
                ).astype(np.float32)
                infl_scores_outputs = [infl_scores_i]

            # influence w.r.t. all loss terms
            print("Calculating influence scores w.r.t. all loss terms")
            IF_instance.test_loss_fn = IF_instance.loss_fn
            IF_instance.model_test = IF_instance.model

            infl_scores = calculate_influence_scores(
                tda_instance=IF_instance,
                candidate_points=candidate_points,
                batch_size=batch_size,
                show_progress=True,
            ).astype(np.float32)

            # in all our experiments we have equal PDEs and n_outputs
            # thus we simplify it here
            # you may need to change this if you have different number of PDEs and outputs
            num_pdes = n_outputs
            num_bcs = len(model.data.bcs)

            # PDE loss
            pde_loss_fn = PINNLoss(
                include_all_losses=False, include_specific_ids=list(range(num_pdes))
            )
            IF_instance.test_loss_fn = pde_loss_fn
            IF_instance.test_reduction_type = "none"

            print("Calculating influence scores w.r.t. PDE loss")
            infl_scores_pde = calculate_influence_scores(
                tda_instance=IF_instance, candidate_points=candidate_points, batch_size=batch_size
            ).astype(np.float32)

            # Individual PDE loss terms (only if more than 1)
            infl_scores_pde_individual = []
            if num_pdes > 1:
                for pde_idx in range(num_pdes):
                    pde_loss_fn_i = PINNLoss(
                        include_all_losses=False, include_specific_ids=[pde_idx]
                    )
                    IF_instance.test_loss_fn = pde_loss_fn_i
                    IF_instance.test_reduction_type = "none"

                    print(f"Calculating influence scores w.r.t. PDE loss term {pde_idx}")
                    infl_scores_pde_i = calculate_influence_scores(
                        tda_instance=IF_instance,
                        candidate_points=candidate_points,
                        batch_size=batch_size,
                    ).astype(np.float32)
                    infl_scores_pde_individual.append(infl_scores_pde_i)

            # BC loss
            bc_loss_fn = PINNLoss(
                include_all_losses=False,
                include_specific_ids=list(range(num_pdes, num_pdes + num_bcs)),
            )
            IF_instance.test_loss_fn = bc_loss_fn
            IF_instance.test_reduction_type = "none"

            print("Calculating influence scores w.r.t. BC loss")
            infl_scores_bc = calculate_influence_scores(
                tda_instance=IF_instance, candidate_points=candidate_points, batch_size=batch_size
            ).astype(np.float32)

            # Individual BC loss terms (only if more than 1)
            infl_scores_bc_individual = []
            if num_bcs > 1:
                for bc_idx in range(num_bcs):
                    bc_loss_fn_i = PINNLoss(
                        include_all_losses=False, include_specific_ids=[num_pdes + bc_idx]
                    )
                    IF_instance.test_loss_fn = bc_loss_fn_i
                    IF_instance.test_reduction_type = "none"

                    print(f"Calculating influence scores w.r.t. BC loss term {bc_idx}")
                    infl_scores_bc_i = calculate_influence_scores(
                        tda_instance=IF_instance,
                        candidate_points=candidate_points,
                        batch_size=batch_size,
                    ).astype(np.float32)
                    infl_scores_bc_individual.append(infl_scores_bc_i)

            # Self-influence calculations (training points on themselves)
            print("Calculating self-influence scores w.r.t. total loss")
            IF_instance.test_loss_fn = IF_instance.loss_fn
            IF_instance.model_test = IF_instance.model
            IF_instance.test_reduction_type = "none"

            self_infl_scores = calculate_influence_scores(
                tda_instance=IF_instance,
                candidate_points=model.data.train_x_all,
                batch_size=batch_size,
            ).astype(np.float32)

            # Self-influence w.r.t. output dimensions
            print("Calculating self-influence scores w.r.t. output dimensions")
            self_infl_scores_outputs = []
            for i in range(n_outputs):
                IF_instance.model_test = NetPredWrapper(model.net, pred_idx=i)
                IF_instance.test_loss_fn = None

                print(f"Calculating self-influence scores w.r.t. output {i}")
                self_infl_scores_i = calculate_influence_scores(
                    tda_instance=IF_instance,
                    candidate_points=model.data.train_x_all,
                    batch_size=batch_size,
                ).astype(np.float32)
                self_infl_scores_outputs.append(self_infl_scores_i)

            infl_scores_abs = np.abs(infl_scores).sum(axis=0)
            infl_scores_pos = infl_scores.sum(axis=0)
            infl_scores_neg = -infl_scores.sum(axis=0)

            # Base save dict (always saved)
            save_dict = {
                "candidate_points": candidate_points,
                "scores_abs": infl_scores_abs,
                "scores_pos": infl_scores_pos,
                "scores_neg": infl_scores_neg,
                "scores": infl_scores,
                "infl_scores_pde": infl_scores_pde,
                "infl_scores_bc": infl_scores_bc,
                "train_x": model.data.train_x_all,
                "holdout_test_x": model.data.holdout_test_x,
                "self_infl_scores": self_infl_scores,
                "self_infl_scores_outputs": self_infl_scores_outputs,
            }

            # Add individual PDE and BC terms if calculated
            if num_pdes > 1:
                for pde_idx, infl_pde_i in enumerate(infl_scores_pde_individual):
                    save_dict[f"infl_scores_pde_{pde_idx}"] = infl_pde_i
            if num_bcs > 1:
                for bc_idx, infl_bc_i in enumerate(infl_scores_bc_individual):
                    save_dict[f"infl_scores_bc_{bc_idx}"] = infl_bc_i
            if n_outputs > 1:
                for out_idx, infl_out_i in enumerate(infl_scores_outputs):
                    save_dict[f"infl_scores_output_{out_idx}"] = infl_out_i

            np.savez_compressed(
                f"{save_path}/{model_name}_influence_scores.npz",
                **save_dict,
            )

        if wrt_individual_loss_terms:
            # Reuse the same IF_instance (same Hessian) and swap models/losses for individual loss terms
            # The Hessian is computed with respect to all losses, but we change the right-side model/loss
            # to focus on specific loss terms

            n_outputs = model.net.linears[-1].out_features
            num_pdes = n_outputs
            num_bcs = len(model.data.bcs)

            tmp_dir = tempfile.TemporaryDirectory()

            # Reinstantiate IF to get a fresh instance for individual loss term calculations
            IF_instance = instantiate_IF(
                model, use_train_set=use_train_set, show_progress=False, tmp_dir=tmp_dir
            )

            # Helper function to compute all left-side influences for a given right-side configuration
            def compute_left_side_influences(IF_inst, name_prefix):
                results = {}

                # Per output dimension (left side = network output)
                output_infls = []
                for out_idx in range(n_outputs):
                    IF_inst.model_test = NetPredWrapper(model.net, pred_idx=out_idx)
                    IF_inst.test_loss_fn = None
                    print(f"  [{name_prefix}] Calculating influence w.r.t. output {out_idx}")
                    infl_out = calculate_influence_scores(
                        tda_instance=IF_inst,
                        candidate_points=candidate_points,
                        batch_size=batch_size,
                    ).astype(np.float32)
                    output_infls.append(infl_out)
                results["outputs"] = output_infls

                IF_inst.model_test = ModelWrapper(
                    net=model.net, pde=model.data.pde, bcs=model.data.bcs, include_pde=True
                )
                IF_inst.test_loss_fn = PINNLoss(include_all_losses=True)
                IF_inst.test_reduction_type = "none"
                print(f"  [{name_prefix}] Calculating influence w.r.t. total loss")
                results["total_loss"] = calculate_influence_scores(
                    tda_instance=IF_inst, candidate_points=candidate_points, batch_size=batch_size
                ).astype(np.float32)

                # PDE loss on left side
                IF_inst.model_test = ModelWrapper(
                    net=model.net, pde=model.data.pde, bcs=[], include_pde=True
                )
                IF_inst.test_loss_fn = PINNLoss(include_all_losses=True)
                IF_inst.test_reduction_type = "none"
                print(f"  [{name_prefix}] Calculating influence w.r.t. PDE loss")
                results["pde_loss"] = calculate_influence_scores(
                    tda_instance=IF_inst, candidate_points=candidate_points, batch_size=batch_size
                ).astype(np.float32)

                # BC loss on left side
                IF_inst.model_test = ModelWrapper(
                    net=model.net, pde=model.data.pde, bcs=model.data.bcs, include_pde=False
                )
                IF_inst.test_loss_fn = PINNLoss(include_all_losses=True)
                IF_inst.test_reduction_type = "none"
                print(f"  [{name_prefix}] Calculating influence w.r.t. BC loss")
                results["bc_loss"] = calculate_influence_scores(
                    tda_instance=IF_inst, candidate_points=candidate_points, batch_size=batch_size
                ).astype(np.float32)

                # Individual PDE loss terms on left side (only if more than 1)
                if num_pdes > 1:
                    pde_individual = []
                    for pde_idx in range(num_pdes):
                        IF_inst.model_test = ModelWrapper(
                            net=model.net, pde=model.data.pde, bcs=[], include_pde=True
                        )
                        IF_inst.test_loss_fn = PINNLoss(
                            include_all_losses=False, include_specific_ids=[pde_idx]
                        )
                        IF_inst.test_reduction_type = "none"
                        print(
                            f"  [{name_prefix}] Calculating influence w.r.t. PDE loss term {pde_idx}"
                        )
                        infl_pde_i = calculate_influence_scores(
                            tda_instance=IF_inst,
                            candidate_points=candidate_points,
                            batch_size=batch_size,
                        ).astype(np.float32)
                        pde_individual.append(infl_pde_i)
                    results["pde_individual"] = pde_individual

                # Individual BC loss terms on left side (only if more than 1)
                if num_bcs > 1:
                    bc_individual = []
                    for bc_idx in range(num_bcs):
                        IF_inst.model_test = ModelWrapper(
                            net=model.net, pde=model.data.pde, bcs=model.data.bcs, include_pde=False
                        )
                        IF_inst.test_loss_fn = PINNLoss(
                            include_all_losses=False, include_specific_ids=[bc_idx]
                        )
                        IF_inst.test_reduction_type = "none"
                        print(
                            f"  [{name_prefix}] Calculating influence w.r.t. BC loss term {bc_idx}"
                        )
                        infl_bc_i = calculate_influence_scores(
                            tda_instance=IF_inst,
                            candidate_points=candidate_points,
                            batch_size=batch_size,
                        ).astype(np.float32)
                        bc_individual.append(infl_bc_i)
                    results["bc_individual"] = bc_individual

                return results

            # Store results for individual loss terms
            individual_loss_term_results = {}

            # Calculate for each individual PDE loss term on the right side
            for pde_idx in range(num_pdes):
                print(f"\n=== Setting up IF for PDE loss term {pde_idx} (right side) ===")
                # Right side: only this specific PDE loss
                IF_instance.model = ModelWrapper(
                    net=model.net, pde=model.data.pde, bcs=[], include_pde=True
                )
                IF_instance.loss_fn = PINNLoss(
                    include_all_losses=False, include_specific_ids=[pde_idx]
                )
                individual_loss_term_results[f"pde_{pde_idx}"] = compute_left_side_influences(
                    IF_instance, f"pde_{pde_idx}"
                )

            # Calculate for each individual BC loss term on the right side
            for bc_idx in range(num_bcs):
                print(f"\n=== Setting up IF for BC loss term {bc_idx} (right side) ===")
                # Right side: only this specific BC loss
                IF_instance.model = ModelWrapper(
                    net=model.net, pde=model.data.pde, bcs=model.data.bcs, include_pde=False
                )
                IF_instance.loss_fn = PINNLoss(
                    include_all_losses=False, include_specific_ids=[bc_idx]
                )
                individual_loss_term_results[f"bc_{bc_idx}"] = compute_left_side_influences(
                    IF_instance, f"bc_{bc_idx}"
                )

            # Flatten the nested dict structure for np.savez compatibility
            individual_save_dict = {
                "candidate_points": candidate_points,
                "train_x": model.data.train_x_all,
                "num_pdes": num_pdes,
                "num_bcs": num_bcs,
            }

            # Add results for each loss term configuration
            for config_key, results in individual_loss_term_results.items():
                for out_idx, out_infl in enumerate(results["outputs"]):
                    individual_save_dict[f"{config_key}_output_{out_idx}"] = out_infl
                individual_save_dict[f"{config_key}_total_loss"] = results["total_loss"]
                individual_save_dict[f"{config_key}_pde_loss"] = results["pde_loss"]
                individual_save_dict[f"{config_key}_bc_loss"] = results["bc_loss"]

                # Add individual PDE terms if they exist
                if "pde_individual" in results:
                    for pde_idx, pde_infl in enumerate(results["pde_individual"]):
                        individual_save_dict[f"{config_key}_pde_{pde_idx}"] = pde_infl

                # Add individual BC terms if they exist
                if "bc_individual" in results:
                    for bc_idx, bc_infl in enumerate(results["bc_individual"]):
                        individual_save_dict[f"{config_key}_bc_{bc_idx}"] = bc_infl

            np.savez_compressed(
                f"{save_path}/{model_name}_influence_scores_individual_loss_terms.npz",
                **individual_save_dict,
            )
            print(
                f"\nSaved individual loss term influences to {save_path}/{model_name}_influence_scores_individual_loss_terms.npz"
            )


def main(
    n_candidate_points: int = 10_000,
    seed: int = 42,
    lr: float = 0.001,
    layers: list = [2] + [32] * 3 + [1],
    n_iterations: int = 10_000,
    n_iterations_lbfgs: int = 0,
    num_domain: int = 1_000,
    num_boundary: int = 0,
    num_initial: int = 0,
    save_path: str = "./model_zoo",
    problem_name: str = "burgers",
    optimizer: str = "adam",
    use_float64: bool = False,
    scoring_method: str = "PINNfluence",
    use_holdout_test: bool = False,
    precalc_infl_sample_uniformly: bool = False,
    device: str = "cpu",
    soft_constrained: bool = False,
    use_train_set: bool = False,
    load_path: str = None,
    broken: bool = False,
    wrt_individual_loss_terms: bool = False,
    left: str = None,
    right: str = None,
    self_influence: bool = False,
    legacy: bool = False,
):
    # Validate arguments
    if not legacy:
        if left is None or right is None:
            raise ValueError("Both --left and --right must be specified unless --legacy is used")
        if wrt_individual_loss_terms:
            raise ValueError("--wrt_individual_loss_terms is only available with --legacy")

    assert scoring_method == "PINNfluence", "Can only precompute PINNfluence"
    if use_float64:
        dde.config.set_default_float("float64")
    dde.config.set_random_seed(seed)

    set_default_device(device)

    # Construct the problem and load the pretrained checkpoint
    model, data, model_name, chkpt_path = problem_factory.construct_problem(
        problem_name=problem_name,
        lr=lr,
        layers=layers,
        n_iterations=n_iterations,
        n_iterations_lbfgs=n_iterations_lbfgs,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        optimizer=optimizer,
        seed=seed,
        float64=use_float64,
        soft_constrained=soft_constrained,
        load_path=load_path,
        broken=broken,
    )

    assert chkpt_path is not None, (
        "Could not load checkpoint. Influences shall be only calculated for already trained models"
    )

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Branch based on mode
    if legacy:
        # ===== LEGACY MODE: Call legacy_main() =====
        legacy_main(
            model=model,
            data=data,
            model_name=model_name,
            save_path=save_path,
            use_holdout_test=use_holdout_test,
            precalc_infl_sample_uniformly=precalc_infl_sample_uniformly,
            n_candidate_points=n_candidate_points,
            scoring_method=scoring_method,
            use_train_set=use_train_set,
            wrt_individual_loss_terms=wrt_individual_loss_terms,
        )
    else:
        # ===== NEW MODE: Single influence matrix computation =====
        assert scoring_method == "PINNfluence", "New mode only supports PINNfluence"

        # Get number of PDEs and BCs
        n_outputs = model.net.linears[-1].out_features
        num_pdes = n_outputs
        num_bcs = len(model.data.bcs)

        # Setup candidate points
        dde.config.set_random_seed(42)

        if self_influence:
            candidate_points = model.data.train_x_all
            print(
                f"Using self-influence mode: {len(candidate_points)} training points as candidates"
            )
        elif use_holdout_test:
            candidate_points = model.data.holdout_test_x
            print(f"Using holdout test set: {len(candidate_points)} points as candidates")
        else:
            if precalc_infl_sample_uniformly:
                candidate_points = data.geom.uniform_points(n_candidate_points)
            else:
                candidate_points = sample_random_points(
                    geometry=model.data.geom, num_points=n_candidate_points, num_bcs=num_bcs
                )
            print(f"Sampled {len(candidate_points)} candidate points")

        batch_size = 1024

        # Instantiate IF
        print("Instantiating influence function...")
        IF_instance = instantiate_IF(
            model,
            use_train_set=True,
            show_progress=False,
            model_name=model_name,
            prefer_load_R=False,
        )

        # Compute single influence matrix
        infl_scores = compute_single_influence(
            IF_instance=IF_instance,
            left_term=left,
            right_term=right,
            model=model,
            candidate_points=candidate_points,
            batch_size=batch_size,
            num_pdes=num_pdes,
            num_bcs=num_bcs,
        )

        print(f"Computed influence scores with shape: {infl_scores.shape}")

        # Create subdirectory for saving
        influence_dir = os.path.join(save_path, f"{model_name}_influence_scores")
        if not os.path.exists(influence_dir):
            os.makedirs(influence_dir)

        # Create filename
        suffix = "_self" if self_influence else ""
        filename = f"influences_{right}_{left}{suffix}.npz"
        filepath = os.path.join(influence_dir, filename)

        # Save to file
        np.savez_compressed(
            filepath,
            scores=infl_scores,
            candidate_points=candidate_points,
            num_pdes=num_pdes,
            num_bcs=num_bcs,
            n_outputs=n_outputs,
            left_term=left,
            right_term=right,
            self_influence=self_influence,
        )
        print(f"\nSaved influence matrix to {filepath}")


if __name__ == "__main__":
    args = parse_args()
    main(
        n_candidate_points=args.n_candidate_points,
        seed=args.seed,
        lr=args.lr,
        layers=args.layers,
        n_iterations=args.n_iterations,
        n_iterations_lbfgs=args.n_iterations_lbfgs,
        num_domain=args.num_domain,
        num_boundary=args.num_boundary,
        num_initial=args.num_initial,
        save_path=args.save_path,
        problem_name=args.problem,
        optimizer=args.optimizer,
        use_float64=args.float64,
        scoring_method=args.scoring_method,
        use_holdout_test=args.precalc_infl_use_holdout_test,
        precalc_infl_sample_uniformly=args.precalc_infl_sample_uniformly,
        device=args.device,
        soft_constrained=args.soft_constrained,
        use_train_set=True,
        load_path=args.load_path,
        broken=args.broken,
        wrt_individual_loss_terms=args.wrt_individual_loss_terms,
        left=args.left,
        right=args.right,
        self_influence=getattr(args, "self_influence", False),
        legacy=args.legacy,
    )
