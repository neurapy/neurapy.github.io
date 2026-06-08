import json

import numpy as np
import plotly.graph_objects as go
import requests
import torch

PROBLEM_AXES = {
    "Aleksandrov": ("x", "t"),
    "burgers": ("x", "t"),
    "burgers_data_loss": ("x", "t"),
    "burgers_w_RAR": ("x", "t"),
    "Cerny": ("x", "t"),
    "diffusion_1d_inverse": ("x", "t"),
    "drift_diffusion_equation": ("x", "t"),
    "magnetic_rod_dde": ("x", "t"),
    "navier_stokes_steady": (
        "x",
        "y",
    ),
    "navier_stokes_steady_manual": (
        "x",
        "y",
    ),
    "navier_stokes_steady_inverse": ("x", "y"),
    "PEB2D": ("x", "t"),
    "SiliToyModel": ("x", "t"),
    "simple_ODE": ("t"),
    "spm_mini": ("x", "t"),
}


def compute_pde_loss(model, X, y=None):
    return np.array(model.predict(X, operator=model.data.pde))


def compute_bc_loss(model, bc, X, y=None):
    loss = np.zeros((X.shape[0], 1))
    if hasattr(bc, "on_boundary"):
        mask_boundary = bc.on_boundary(X, bc.geom.on_boundary(X))
    elif hasattr(bc, "on_initial"):
        mask_boundary = bc.on_initial(X, bc.geom.on_initial(X))
    else:
        raise ValueError("Boundary condition does not have 'on_boundary' or 'on_initial' method.")
    x_subset = X[mask_boundary]
    X_tensor = torch.tensor(x_subset, dtype=torch.float32, requires_grad=True)
    preds = model.net(X_tensor)
    idx_boundary = np.where(mask_boundary)[0]
    if x_subset.shape[0] > 0:
        bc_err = bc.error(
            x_subset,
            X_tensor,
            preds,
            0,
            x_subset.shape[0],
        )

        if isinstance(bc_err, torch.Tensor):
            bc_err = bc_err.detach().cpu().numpy()
        loss[idx_boundary, 0] = bc_err.flatten()
    return loss


def compute_loss(loss_type, pinnfluence_instance, X, y=None):
    model = pinnfluence_instance.trainer.model
    loss = None
    losses = []
    if loss_type == "loss":
        pde_losses = compute_pde_loss(model, X, y)
        # shape is [N, 1]
        if pde_losses.ndim == 2:
            losses.append(pde_losses)
        # shape is [n_pde, N, 1]
        else:
            losses.append(np.sum(pde_losses, axis=0).reshape(-1, 1))
        bcs = model.data.bcs
        if len(bcs) > 0:
            for bc in bcs:
                bc_loss = compute_bc_loss(model, bc, X, y)
                losses.append(bc_loss)
        loss = np.sum(losses, axis=0)
    elif loss_type == "pde":
        pde_losses = compute_pde_loss(model, X, y)
        if pde_losses.ndim == 2:
            loss = pde_losses.flatten()
        else:
            loss = np.sum(pde_losses, axis=0)
        return loss
    elif loss_type.startswith("pde"):
        loss = compute_pde_loss(model, X, y)
        idx = int(loss_type[4:])
        # If multiple PDE losses (shape [n_pde, N, 1]), select the correct one
        if loss.ndim == 3:
            # loss shape: [n_pde, N, 1]
            loss = loss[idx, :, 0].reshape(-1, 1)
        elif loss.ndim == 2:
            # loss shape: [N, 1] (only one PDE)
            loss = loss.flatten()
        else:
            raise ValueError(f"Unexpected PDE loss shape: {loss.shape}")
        return loss
    elif loss_type.startswith("bc"):
        bcs = model.data.bcs
        if len(bcs) > 0:
            idx = int(loss_type[3:])
            if idx < len(bcs):
                loss = compute_bc_loss(model, bcs[idx], X, y)
            else:
                raise ValueError(f"Index {idx} out of bounds for BCs.")
        else:
            raise ValueError("No BCs found.")
        return loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return loss


# --- Plotting Helpers ---
def show_change_in_prediction(test_x, output_name, influences, closest_train_point, marker_size=10):
    xx_nm = test_x[:, 0]
    tt_min = test_x[:, 1]
    influences_abs = np.abs(influences)
    max_abs = np.max(influences_abs)
    colorscale = [(0.0, "blue"), (0.5, "white"), (1.0, "red")]

    fig = go.Figure(
        data=go.Scatter(
            x=xx_nm,
            y=tt_min,
            mode="markers",
            marker=dict(
                color=influences,
                colorscale=colorscale,
                cmin=-max_abs,
                cmax=max_abs,
                showscale=True,
                colorbar=dict(title=f"{output_name} [scaled]"),
                size=marker_size,
            ),
            hovertemplate=f"{output_name}: %{{marker.color:.2e}}<extra></extra>",
            showlegend=False,  # Prevent this trace from appearing in the legend
        )
    )
    closest_train_x_nm = closest_train_point[0]
    closest_train_t_min = closest_train_point[1]
    fig.add_trace(
        go.Scatter(
            x=[closest_train_x_nm],
            y=[closest_train_t_min],
            mode="markers",
            marker=dict(symbol="star", color="gold", size=18, line=dict(width=2, color="black")),
            name="Selected Training Point",
            showlegend=True,
            legendgroup="selected",
        )
    )
    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            font=dict(color="black"),
        )
    )
    return fig


def show_mean_influence(train_x, key, influences, marker_size=10, colormap="Jet", log_scale=True):
    xx_nm = train_x[:, 0]
    tt_min = train_x[:, 1]
    if log_scale:
        mean_influences = np.log(influences.mean(axis=0) + 1e-10)
    else:
        mean_influences = influences.mean(axis=0)
    fig = go.Figure(
        data=go.Scatter(
            x=xx_nm,
            y=tt_min,
            mode="markers",
            marker=dict(
                color=mean_influences,
                colorscale=colormap,
                showscale=True,
                colorbar=dict(title="Mean Influence"),
                size=marker_size,
            ),
        )
    )
    fig.update_layout(
        title=f"Mean Abs Influence of Each Training Point on '{key}'",
        xaxis_title="y",
        yaxis_title="x",
    )
    return fig


def remote_predict_stream(X, config_path, api_url="http://127.0.0.1:8000/predict_stream"):
    """
    Stream predictions from the API server as a numpy array.
    """
    data = {"config_path": config_path, "X": X.tolist()}
    response = requests.post(api_url, json=data, stream=True)
    response.raise_for_status()
    rows = [json.loads(line) for line in response.iter_lines() if line]
    return np.array(rows)


def remote_influence_stream(key, config_path, api_url="http://127.0.0.1:8000/influence_stream"):
    """
    Stream influence array from the API server as a numpy array.
    """
    data = {"config_path": config_path, "key": key}
    response = requests.post(api_url, json=data, stream=True)
    response.raise_for_status()
    rows = [json.loads(line) for line in response.iter_lines() if line]
    return np.array(rows)


def remote_meta(config_path, api_url="http://127.0.0.1:8000/meta"):
    data = {"config_path": config_path}
    response = requests.post(api_url, json=data)
    response.raise_for_status()
    return response.json()


def remote_loss(loss_type, X, config_path, api_url="http://127.0.0.1:8000/loss"):
    data = {"config_path": config_path, "loss_type": loss_type, "X": X.tolist()}
    response = requests.post(api_url, json=data)
    response.raise_for_status()
    return np.array(response.json()["loss"])
