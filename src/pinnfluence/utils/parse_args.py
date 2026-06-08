import argparse

from .defaults import DEFAULTS


def common_arg_parser():
    parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="Random seed")
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULTS["lr"],
        help="Learning rate",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULTS["layers"],
        help="Layer sizes",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=DEFAULTS["n_iterations"],
        help="Number of iterations",
    )
    parser.add_argument(
        "--n_iterations_lbfgs",
        type=int,
        default=DEFAULTS["n_iterations_lbfgs"],
        help="Number of LBFGS iterations",
    )
    parser.add_argument(
        "--num_domain",
        type=int,
        default=DEFAULTS["num_domain"],
        help="Number of domain points",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=DEFAULTS["model_zoo"],
        help="Path to save model",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to load model",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="burgers",
        help="Problem/PDE to run experiment for",
        choices=[
            "allen_cahn",
            "burgers",
            "diffusion",
            "drift_diffusion",
            "heat_equation_1d",
            "heat_equation_2d",
            "heat_equation_2d_body",
            "navier_stokes",
            "navier_stokes_nd",
            "navier_stokes_nd_cylinder_in_center",
            "poisson_disk",
            "wave",
        ],
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Which optimizer to use for training",
        choices=["adam", "L-BFGS", "SOAP", "NNCG"],
        default=DEFAULTS["optimizer"],
    )
    parser.add_argument(
        "--float64",
        action="store_true",
        help="Use float64",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for training.",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--soft_constrained",
        action="store_true",
        help="Use soft constrained training",
    )
    parser.add_argument(
        "--num_boundary",
        type=int,
        default=0,
        help="Number of boundary points",
    )
    parser.add_argument(
        "--num_initial",
        type=int,
        default=0,
        help="Number of initial points",
    )
    parser.add_argument(
        "--drop_single_point_type",
        type=str,
        default="none",
        help="Type of single point to drop",
        choices=["none", "IC", "BC", "domain"],
    )
    parser.add_argument(
        "--broken",
        action="store_true",
        help="Use broken training",
    )
    return parser


def parse_pretrain_args():
    parser = argparse.ArgumentParser(
        description="Pretrain a PINN model",
        parents=[common_arg_parser()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use wandb",
    )
    parser.add_argument(
        "--dont_force_retrain",
        action="store_false",
        dest="force_retrain",
        help="Don't force retraining, i.e. continue training from checkpoint if available",
    )
    return parser.parse_args()


def parse_run_experiment_args():
    parser = argparse.ArgumentParser(
        description="Run a finetuning experiment. If you want to use a pretrained model, please match the arguments of the respective pretraining run.",
        parents=[common_arg_parser()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="distribution",
        choices=["top_k", "distribution"],
        help="Sampling strategy",
    )
    parser.add_argument(
        "--n_candidate_points",
        type=int,
        default=10000,
        help="Number of candidate points",
    )
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to select")
    parser.add_argument("--distribution_k", type=int, default=2, help="Distribution parameter k")
    parser.add_argument("--distribution_c", type=int, default=0, help="Distribution parameter c")
    parser.add_argument(
        "--scoring_method",
        type=str,
        default="PINNfluence",
        choices=[
            "PINNfluence",
            "RAR",
            "random",
            "grad_dot",
            "steepest_prediction_gradient",
            "steepest_loss_gradient",
        ],
        help="Scoring strategy",
    )
    parser.add_argument(
        "--scoring_sign",
        type=str,
        default="abs",
        choices=["abs", "pos", "neg"],
        help="Sign for scoring",
    )
    parser.add_argument(
        "--pertubation_strategy",
        type=str,
        default="add",
        choices=["add", "replace"],
        help="Training strategy",
    )
    parser.add_argument(
        "--n_iterations_finetune",
        type=int,
        default=1000,
        help="Number of finetuning iterations",
    )
    parser.add_argument(
        "--n_iterations_lbfgs_finetune",
        type=int,
        default=0,
        help="L-BFGS iterations for finetuning",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="full",
        choices=["full", "train", "valid"],
        help="Model version to load",
    )
    parser.add_argument(
        "--recover_run",
        action="store_true",
        help="Recover and continue finetuning",
    )
    parser.add_argument(
        "--n_cycles_finetune", type=int, default=100, help="Number of finetuning cycles"
    )
    return parser.parse_args()


def parse_precalculate_args():
    parser = argparse.ArgumentParser(
        description="Precalculate influence scores. Please match the arguments of the respective pretraining run you with to precalculate influences for.",
        parents=[common_arg_parser()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n_candidate_points",
        type=int,
        default=10_000,
        help="Number of candidate points",
    )
    parser.add_argument(
        "--scoring_method",
        type=str,
        default="PINNfluence",
        choices=["PINNfluence", "grad_dot"],
        help="Scoring strategy",
    )
    parser.add_argument(
        "--precalc_infl_use_holdout_test",
        action="store_true",
        help="Use holdout test set for precalc",
    )
    parser.add_argument(
        "--precalc_infl_sample_uniformly",
        action="store_true",
        help="Sample uniformly for precalc",
    )
    parser.add_argument(
        "--use_train_set",
        action="store_true",
        help="Use train set for precalc",
    )
    parser.add_argument(
        "--wrt_individual_loss_terms",
        action="store_true",
        help="Calculate influences w.r.t. individual loss terms on the right side (only available with --legacy)",
    )
    parser.add_argument(
        "--left",
        type=str,
        default=None,
        help="Test loss term (required unless --legacy is used). Options: total_loss, pde_loss, bc_loss, pde_N, bc_N, output_N",
    )
    parser.add_argument(
        "--right",
        type=str,
        default=None,
        help="Training loss term (required unless --legacy is used). Options: total_loss, pde_loss, bc_loss, pde_N, bc_N",
    )
    parser.add_argument(
        "--self-influence",
        action="store_true",
        help="Calculate self-influence using train_x_all as candidates",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy monolithic file save behavior",
    )
    return parser.parse_args()
