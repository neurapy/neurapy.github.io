import json

import numpy as np
import plotly.colors
import plotly.graph_objects as go
import requests
import streamlit as st
from utils import (
    show_change_in_prediction,
    show_mean_influence,
)

ALL_COLORMAPS = sorted([c for c in plotly.colors.PLOTLY_SCALES.keys()])

API_URL = "http://127.0.0.1:8001"


@st.cache_data()
def remote_list_dirs():
    response = requests.get(f"{API_URL}/list_dirs")
    response.raise_for_status()
    return response.json()["directories"]


@st.cache_data()
def remote_list_models(directory):
    response = requests.post(f"{API_URL}/list_models", json={"directory": directory})
    response.raise_for_status()
    return response.json()["models"]


@st.cache_data()
def remote_predict_stream(X, directory, model_name, api_endpoint="predict_stream", api_port=8000):
    api_url = f"{API_URL}/{api_endpoint}"
    data = {"directory": directory, "model_name": model_name, "X": X.tolist()}
    response = requests.post(api_url, json=data, stream=True)
    response.raise_for_status()
    rows = [json.loads(line) for line in response.iter_lines() if line]
    return np.array(rows)


@st.cache_data()
def remote_influence_stream(
    key, directory, model_name, api_endpoint="influence_stream", api_port=8000
):
    api_url = f"{API_URL}/{api_endpoint}"
    data = {"directory": directory, "model_name": model_name, "key": key}
    response = requests.post(api_url, json=data, stream=True)
    response.raise_for_status()
    rows = [json.loads(line) for line in response.iter_lines() if line]
    return np.array(rows)


@st.cache_data()
def remote_meta(directory, model_name, api_endpoint="meta", api_port=8000):
    api_url = f"{API_URL}/{api_endpoint}"
    data = {"directory": directory, "model_name": model_name}
    response = requests.post(api_url, json=data)
    response.raise_for_status()
    return response.json()


# @st.cache_data()
def remote_loss(loss_type, X, directory, model_name, api_endpoint="loss", api_port=8000):
    api_url = f"{API_URL}/{api_endpoint}"
    data = {
        "directory": directory,
        "model_name": model_name,
        "loss_type": loss_type,
        "X": X.tolist(),
    }
    response = requests.post(api_url, json=data)
    response.raise_for_status()
    return np.array(response.json()["loss"])


@st.cache_data()
def remote_uniform_points(
    directory,
    model_name,
    num_samples=10000,
    api_endpoint="uniform_points",
    api_port=8000,
):
    api_url = f"{API_URL}/{api_endpoint}"
    data = {
        "directory": directory,
        "model_name": model_name,
        "num_samples": num_samples,
    }
    response = requests.post(api_url, json=data)
    response.raise_for_status()
    arr = np.array(response.json()["points"])
    return arr


@st.cache_resource()
def remote_get_influences_path(
    directory, model_name, api_endpoint="get_influences_path", api_port=8000
):
    api_url = f"{API_URL}/{api_endpoint}"
    data = {"directory": directory, "model_name": model_name}
    response = requests.post(api_url, json=data)
    response.raise_for_status()
    return response.json()["path"]


@st.cache_data()
def load_influences(directory, model_name, key):
    path = remote_get_influences_path(directory, model_name)
    if not path:
        raise FileNotFoundError(f"No influence file found for {model_name} in {directory}")
    arr = np.load(path)
    return arr[key]


# --- Main Streamlit App ---


def main(api_port=8000):
    st.set_page_config(layout="centered")
    st.title("PINNfluence Interactive Demo")

    # Sidebar: Model Zoo Directory and Model Selection
    st.sidebar.header("Model Selection")
    directories = remote_list_dirs()
    selected_dir = st.sidebar.selectbox("Select a model zoo directory", directories)
    models = remote_list_models(selected_dir)
    if not models:
        st.info("No models found in this directory.")
        st.stop()
    selected_model = st.sidebar.selectbox("Select a model", models)

    # Meta info
    meta = remote_meta(selected_dir, selected_model)
    n_bc = meta.get("n_bc", 1)
    n_pde = meta.get("n_pde", 1)
    n_outputs = meta.get("n_outputs", 1)
    outputs = [f"output_{i}" for i in range(n_outputs)]
    if len(outputs) == 1:
        output_name = outputs[0]
        output_idx = 0
    else:
        outputs_to_show = {" ".join(output.title().split("_")): output for output in outputs}
        output_name = st.sidebar.selectbox("Select Output to Visualize", outputs_to_show.keys())
        output_idx = int(outputs_to_show[output_name].split("_")[-1])

    key_options = (
        ["loss"]
        + ["pde", "bc"]
        + [f"output_dim_{i}" for i in range(len(outputs))]
        + [f"bc_{i}" for i in range(n_bc)]
        + [f"pde_{i}" for i in range(n_pde)]
    )
    key = st.sidebar.selectbox("Inspect Influences For", key_options)
    k = st.sidebar.slider("Top k Influential Samples", 1, 100, 10)
    influence_sign = st.sidebar.selectbox("Influence Sign", ["absolute", "positive", "negative"])

    # --- Generate Valid Domain Points ---
    col1, col2 = st.columns(2)
    with col1:
        num_samples = st.number_input(
            "Number of Samples", min_value=100, max_value=50_000, value=10_000
        )
    with col2:
        colormap_pred_plot = st.selectbox(
            "Colormap for Predictions",
            ALL_COLORMAPS,
            index=ALL_COLORMAPS.index("Jet") if "Jet" in ALL_COLORMAPS else 0,
        )
    xt_vals = remote_uniform_points(selected_dir, selected_model, num_samples=num_samples)
    with st.spinner("Generating predictions for valid domain points..."):
        y_pred = remote_predict_stream(xt_vals, selected_dir, selected_model)
    # y_pred is (N, output_dim)
    # For 2D, plot as heatmap with clickable scatter overlay
    st.warning(
        "Note: Please switch to Scatterplot if the heatmap doesn't show up. This happens when the geometry doesn't implement uniform sampling."
    )
    selector_heatmap_scatter = st.radio(
        "Select Plot Type", ["heatmap", "scatter"], key="selector_heatmap_scatter"
    )
    if selector_heatmap_scatter == "heatmap":
        fig = go.Figure(
            go.Heatmap(
                x=xt_vals[:, 0],
                y=xt_vals[:, 1],
                z=y_pred[:, output_idx],
                colorscale=colormap_pred_plot,
                colorbar=dict(title=f"{output_name}"),
                hoverinfo="none",
            )
        )
        # Add transparent scatter for clickability
        fig.add_scatter(
            x=xt_vals[:, 0],
            y=xt_vals[:, 1],
            mode="markers",
            marker=dict(symbol="square", size=20, color="rgba(0,0,0,0)"),
            name="clickable_points",
        )
    else:
        marker_size = st.slider("Marker Size", 1, 100, 5, key="marker_size_scatter_plot")
        fig = go.Figure(
            go.Scatter(
                x=xt_vals[:, 0],
                y=xt_vals[:, 1],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=y_pred[:, output_idx],
                    colorscale=colormap_pred_plot,
                    colorbar=dict(title=f"{output_name}"),
                ),
            )
        )

    # Display the Plotly chart and capture click events
    selected_points = st.plotly_chart(fig, on_select="rerun", selection_mode="points")
    if "selected_test_idx" not in st.session_state:
        st.session_state["selected_test_idx"] = 0
    if (
        selected_points
        and "selection" in selected_points
        and selected_points["selection"].get("points")
    ):
        point = selected_points["selection"]["points"][0]
        click_x, click_y = point["x"], point["y"]
        dists = np.linalg.norm(xt_vals - np.array([click_x, click_y]), axis=1)
        st.session_state["selected_test_idx"] = int(np.argmin(dists))
    test_idx = st.session_state["selected_test_idx"]
    clicked_x, clicked_t = xt_vals[test_idx]
    st.write(f"Selected Test Point: ({clicked_x:.3f}, {clicked_t:.3f})")
    test_point = np.array([clicked_x, clicked_t])
    # Influence and train/test samples loading
    test_samples = load_influences(selected_dir, selected_model, "test_x")
    distances = np.linalg.norm(test_samples - test_point, axis=1)
    closest_test_idx = np.argmin(distances)
    train_samples = load_influences(selected_dir, selected_model, "train_x")
    influence_arr = load_influences(selected_dir, selected_model, key)
    influence_scores = -influence_arr[closest_test_idx, :] * (1 / train_samples.shape[0])
    if influence_sign == "positive":
        indices = np.argsort(influence_scores)[::-1][:k]
    elif influence_sign == "negative":
        indices = np.argsort(-influence_scores)[::-1][:k]
    else:
        indices = np.argsort(np.abs(influence_scores))[::-1][:k]
    top_k_influences = influence_scores[indices]
    top_k_samples = train_samples[indices]
    st.header(f"Top {k} Influential Training Points for Selected Test Point")

    st.badge("Note: Test points and training points don't necessarily match the heatmap points.")

    # --- Restore the scatter plot for top-k influential points ---
    if selector_heatmap_scatter == "heatmap":
        fig_topk = go.Figure(
            data=go.Heatmap(
                x=xt_vals[:, 0],
                y=xt_vals[:, 1],
                z=y_pred[:, output_idx],
                colorscale=colormap_pred_plot,
                hoverinfo="none",
                colorbar=dict(title=f"{output_name}"),
            )
        )
    else:
        fig_topk = go.Figure(
            data=go.Scatter(
                x=xt_vals[:, 0],
                y=xt_vals[:, 1],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=y_pred[:, output_idx],
                    colorscale=colormap_pred_plot,
                    colorbar=dict(title=f"{output_name}"),
                ),
            )
        )
    # Add top-k influential points
    fig_topk.add_scatter(
        x=top_k_samples[:, 0],
        y=top_k_samples[:, 1],
        mode="markers",
        marker=dict(
            size=14,
            color=top_k_influences,
            colorscale=[(0, "blue"), (0.5, "white"), (1, "red")],
            cmin=-np.max(np.abs(top_k_influences)),
            cmax=np.max(np.abs(top_k_influences)),
            colorbar=dict(
                title="Influence",
                x=1.13,  # Move this colorbar further right to avoid overlap
            ),
            line=dict(width=2, color="black"),
        ),
        name="Top-k Influential Points",
        showlegend=True,
        legendgroup="topk",
    )
    # Highlight the selected test point
    fig_topk.add_scatter(
        x=[test_point[0]],
        y=[test_point[1]],
        mode="markers",
        marker=dict(size=18, color="gold", symbol="star", line=dict(width=2, color="black")),
        name="Selected Test Point",
        showlegend=True,
        legendgroup="selected",
    )
    # Move the legend to the left to avoid overlap with colorbars
    fig_topk.update_layout(
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
    st.plotly_chart(fig_topk, use_container_width=True)

    st.header("Deeper Influence Analysis")
    train_distances = np.linalg.norm(train_samples - test_point, axis=1)
    closest_train_idx = np.argmin(train_distances)
    closest_train_point = train_samples[closest_train_idx]
    pred_influences = -load_influences(selected_dir, selected_model, f"output_dim_{output_idx}")[
        :, closest_train_idx
    ]
    marker_size = st.slider("Marker Size", 1, 100, 10, key="marker_size_diff_plot")
    change_fig = show_change_in_prediction(
        test_samples,
        output_name,
        pred_influences,
        closest_train_point,
        marker_size=marker_size,
    )
    st.plotly_chart(change_fig, use_container_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        marker_size = st.slider("Marker Size", 1, 100, 10, key="marker_size_global_plot")
    with col2:
        colormap = st.selectbox(
            "Colormap",
            ALL_COLORMAPS,
            key="colormap_global_plot",
            index=ALL_COLORMAPS.index("Jet") if "Jet" in ALL_COLORMAPS else 0,
        )
    with col3:
        log_scale = st.checkbox("Log Scale", value=False, key="log_scale_global_plot")
        absolute_values = st.checkbox(
            "Absolute Values", value=True, key="absolute_values_global_plot"
        )
    influence_values = load_influences(selected_dir, selected_model, key)
    if absolute_values:
        influence_values = np.abs(influence_values)
    mean_fig = show_mean_influence(
        train_samples,
        key,
        influence_values,
        marker_size=marker_size,
        colormap=colormap,
        log_scale=log_scale,
    )
    st.plotly_chart(mean_fig, use_container_width=True)
    st.header("Loss Plot (PDE and BC Terms)")
    loss_type = st.selectbox(
        "Select Loss Type",
        options=["loss"]
        + [f"pde_{i}" for i in range(n_pde)]
        + [f"bc_{i}" for i in range(n_bc)]
        + ["pde", "bc"],
    )
    try:
        loss_values = remote_loss(loss_type, xt_vals, selected_dir, selected_model)
        if (
            loss_values.shape[0] == xt_vals.shape[0]
            and loss_values.ndim == 2
            and loss_values.shape[1] == 1
        ):
            loss_values = loss_values.ravel()
        elif loss_values.shape[0] == xt_vals.shape[0]:
            pass
        else:
            st.error(f"Shape mismatch: xt_vals {xt_vals.shape}, loss_values {loss_values.shape}")
            return
        col1, col2, col3 = st.columns(3)
        with col1:
            marker_size = st.slider("Marker Size", 1, 25, 5, key="marker_size_loss_plot")
        with col2:
            colormap = st.selectbox(
                "Colormap",
                ALL_COLORMAPS,
                key="colormap_loss_plot",
                index=ALL_COLORMAPS.index("Jet") if "Jet" in ALL_COLORMAPS else 0,
            )
        with col3:
            log_scale = st.checkbox("Log Scale", value=False, key="log_scale_loss_plot")
            absolute_values = st.checkbox(
                "Absolute Values", value=True, key="absolute_values_loss_plot"
            )
            hide_zero_values = st.checkbox(
                "Hide Zero Values", value=True, key="hide_zero_values_loss_plot"
            )
        if log_scale:
            loss_values = np.log(loss_values)
        if absolute_values:
            loss_values = np.abs(loss_values)
        if hide_zero_values:
            _loss_values = loss_values[loss_values != 0]
            _xt_vals = xt_vals[loss_values != 0]
        else:
            _loss_values = loss_values
            _xt_vals = xt_vals
        loss_fig = go.Figure(
            data=go.Scatter(
                x=_xt_vals[:, 0],
                y=_xt_vals[:, 1],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=_loss_values,
                    colorscale=colormap,
                    colorbar=dict(title=f"{loss_type} Loss"),
                    cmin=0 if absolute_values else None,
                    cmax=None if absolute_values else None,
                ),
                hovertemplate="%{x:.3f}, %{y:.3f}<br>Loss: %{marker.color:.2e}<extra></extra>",
            )
        )

        st.plotly_chart(loss_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error computing {loss_type} loss: {e}")


if __name__ == "__main__":
    main()
