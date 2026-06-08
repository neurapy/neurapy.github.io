import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tqdm import tqdm

from pinnfluence import problem_factory

app = FastAPI()

MODEL_ZOO_ROOT = Path("model_zoo_icml")
ROOT_DIR_NAME = "<root>"


def _strip_pt_suffix(name: str) -> str:
    return name.replace("_full.pt", "").replace("_train.pt", "").replace("_valid.pt", "")


def _resolve_dir(dir_name: str) -> Path:
    if dir_name in (ROOT_DIR_NAME, "", "."):
        return MODEL_ZOO_ROOT
    return MODEL_ZOO_ROOT / dir_name


def _find_influence_file(base: str) -> Path | None:
    """Look for `<base>_influence_scores.npz` anywhere under the zoo root.

    Influence files for a given checkpoint may live alongside the .pt or at
    the zoo root, so we search both layouts.
    """
    target = f"{base}_influence_scores.npz"
    direct = MODEL_ZOO_ROOT / target
    if direct.exists():
        return direct
    matches = list(MODEL_ZOO_ROOT.rglob(target))
    return matches[0] if matches else None


# --- Utility functions ---
def list_model_zoo_dirs():
    """Return every directory (root + subdirs) that contains at least one
    usable model — i.e. a `.pt` checkpoint with a matching influence file."""
    dirs = []
    # root
    if any(
        _strip_pt_suffix(p.name) and _find_influence_file(_strip_pt_suffix(p.name))
        for p in MODEL_ZOO_ROOT.glob("*.pt")
    ):
        dirs.append(ROOT_DIR_NAME)
    # subdirs
    for d in sorted(MODEL_ZOO_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if any(_find_influence_file(_strip_pt_suffix(p.name)) for p in d.glob("*.pt")):
            dirs.append(d.name)
    return dirs


def list_models_in_dir(dir_name):
    dir_path = _resolve_dir(dir_name)
    if not dir_path.exists():
        return []
    models = []
    for pt in sorted(dir_path.glob("*.pt")):
        base = _strip_pt_suffix(pt.name)
        if _find_influence_file(base) is not None:
            models.append(pt.name)
    return models


def get_influence_file(dir_name, model_name):
    base = _strip_pt_suffix(model_name)
    return _find_influence_file(base)


def get_model_file(dir_name, model_name):
    dir_path = _resolve_dir(dir_name)
    model_file = dir_path / model_name
    return model_file if model_file.exists() else None


# --- API Models ---
class ListDirsResponse(BaseModel):
    directories: list[str]


class ListModelsRequest(BaseModel):
    directory: str


class ListModelsResponse(BaseModel):
    models: list[str]


class ModelSelection(BaseModel):
    directory: str
    model_name: str


class PredictRequest(ModelSelection):
    X: list


class InfluenceRequest(ModelSelection):
    key: str


class MetaRequest(ModelSelection):
    pass


class LossRequest(ModelSelection):
    loss_type: str
    X: list
    y: list | None = None


class UniformPointsRequest(ModelSelection):
    num_samples: int = 10000


class GetInfluencesPathRequest(ModelSelection):
    pass


# --- Model/Influence Cache ---
@lru_cache(maxsize=2)
def get_model_and_data(directory, model_name):
    model_file = get_model_file(directory, model_name)
    if model_file is None:
        raise RuntimeError(f"Model file not found: {model_name} in {directory}")
    # Parse model_name to extract problem and params
    # Example: burgers_adam_50000_adam_0_lbfgs_2000_domain_0_boundary_0_initial_3_x_32_hidden_float64_True_42_hard_full.pt
    parts = model_name.split("_")
    problem_name = parts[0]
    optimizer = parts[1]
    n_iterations = int(parts[2])
    n_iterations_lbfgs = int(parts[4])
    num_domain = int(parts[6])
    num_boundary = int(parts[8])
    num_initial = int(parts[10])
    layers = [2] + [int(parts[12])] * int(parts[11]) + [1]
    float64 = parts[14] == "True"
    seed = int(parts[15])
    soft_constrained = parts[16] == "soft"
    model_version = (
        "full" if "full" in model_name else ("train" if "train" in model_name else "valid")
    )
    model, data, model_name_out, chkpt_path = problem_factory.construct_problem(
        problem_name=problem_name,
        optimizer=optimizer,
        n_iterations=n_iterations,
        n_iterations_lbfgs=n_iterations_lbfgs,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        layers=layers,
        seed=seed,
        float64=float64,
        model_version=model_version,
        load_path=str(MODEL_ZOO_ROOT),
        soft_constrained=soft_constrained,
    )
    return model, data


@lru_cache(maxsize=8)
def get_influences_cached(directory, model_name):
    infl_file = get_influence_file(directory, model_name)
    if infl_file is None:
        raise RuntimeError(f"Influence file not found for {model_name} in {directory}")
    arrs = np.load(infl_file, allow_pickle=True)
    return {k: arrs[k] for k in arrs.files}


# --- API Endpoints ---
@app.get("/list_dirs", response_model=ListDirsResponse)
def list_dirs():
    return ListDirsResponse(directories=list_model_zoo_dirs())


@app.post("/list_models", response_model=ListModelsResponse)
def list_models(req: ListModelsRequest):
    return ListModelsResponse(models=list_models_in_dir(req.directory))


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        model, data = get_model_and_data(req.directory, req.model_name)
        X = np.array(req.X, dtype=np.float32)
        preds = model.predict(X)
        if isinstance(preds, np.ndarray):
            preds = preds.tolist()
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_stream")
def predict_stream(req: PredictRequest):
    try:
        model, data = get_model_and_data(req.directory, req.model_name)
        X = np.array(req.X, dtype=np.float32)
        preds = model.predict(X)

        def iter_rows():
            for row in preds:
                yield json.dumps(row.tolist()) + "\n"

        return StreamingResponse(iter_rows(), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/influence_stream")
def influence_stream(req: InfluenceRequest):
    try:
        arrs = get_influences_cached(req.directory, req.model_name)
        if req.key not in arrs:
            raise KeyError(
                f"Key '{req.key}' not found in influence arrays. Available keys: {list(arrs.keys())}"
            )
        arr = np.array(arrs[req.key])

        def iter_rows():
            for row in tqdm(arr):
                yield json.dumps(np.array(row, dtype=np.float16).tolist()) + "\n"

        return StreamingResponse(iter_rows(), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/meta")
def get_meta(req: MetaRequest):
    try:
        model, data = get_model_and_data(req.directory, req.model_name)
        n_bc = len(data.bcs)
        X = data.train_x[:10]
        n_outputs = model.net.linears[-1].out_features
        geom = data.geom
        from deepxde import geometry

        if isinstance(geom, geometry.GeometryXTime):
            x_min, x_max = geom.geometry.bbox
            x_min = float(x_min[0])
            x_max = float(x_max[0])
            y_min, y_max = geom.timedomain.bbox
            y_min = float(y_min[0])
            y_max = float(y_max[0])
        elif isinstance(geom, geometry.Geometry):
            try:
                xy_bl, xy_tr = geom.bbox
                x_min, y_min = float(xy_bl[0]), float(xy_bl[1])
                x_max, y_max = float(xy_tr[0]), float(xy_tr[1])
            except:
                x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0
        else:
            x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0
        try:
            pde_out = model.predict(X, operator=data.pde)
            if isinstance(pde_out, (list, tuple)):
                n_pde = len(pde_out)
            elif hasattr(pde_out, "shape") and len(pde_out.shape) > 1:
                n_pde = pde_out.shape[1]
            else:
                n_pde = 1
        except Exception:
            n_pde = 1
        return {
            "n_bc": n_bc,
            "n_pde": n_pde,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "n_outputs": n_outputs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/loss")
def get_loss(req: LossRequest):
    try:
        model, data = get_model_and_data(req.directory, req.model_name)
        X = np.array(req.X, dtype=np.float32)
        y = np.array(req.y, dtype=np.float32) if req.y is not None else None
        # Use the same compute_loss logic as before, but adapted to your model/data
        # For now, just return zeros as a placeholder
        loss = np.zeros((X.shape[0], 1))
        return {"loss": loss.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/uniform_points")
def uniform_points(req: UniformPointsRequest):
    try:
        model, data = get_model_and_data(req.directory, req.model_name)
        geom = data.geom
        points = geom.uniform_points(req.num_samples)
        if geom.on_boundary(points).sum() == 0:
            boundary_points = geom.uniform_boundary_points(req.num_samples // 10)
            points = points[
                np.random.choice(
                    points.shape[0],
                    size=points.shape[0] - boundary_points.shape[0],
                    replace=False,
                )
            ]
            points = np.concatenate([points, boundary_points])
        return {"points": points.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_influences_path")
def get_influences_path(req: GetInfluencesPathRequest):
    try:
        infl_file = get_influence_file(req.directory, req.model_name)
        return {"path": str(infl_file) if infl_file else ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
