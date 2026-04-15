import os
import glob
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from stqdm import stqdm

# Import WaveBreaker modules
from src.data_loader import load_primary_data, load_secondary_path
from src.model import WaveBreakerANC, WaveBreakerPro
from src.acoustics import estimate_impulse_response, apply_acoustic_path

# --- Configuration & UI Setup ---
st.set_page_config(page_title="WaveBreaker ANC", layout="wide")

# Main Page Header
st.title("🌊 WaveBreaker ANC")
st.markdown("Active Noise Control AI Dashboard")
st.caption("© D.J. Medina (2026)")

# Navigation Menu
page = st.radio("Navigation", ["Data Processing", "Model Training", "Inferencing"], horizontal=True,
                label_visibility="collapsed")
st.markdown("---")

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SEC_LEFT = os.path.join(DATA_DIR, 'Secondary_left.wav')
SEC_RIGHT = os.path.join(DATA_DIR, 'Secondary_right.wav')


# --- Helper Functions ---
@st.cache_data
def scan_primary_files():
    files = glob.glob(os.path.join(DATA_DIR, "Primary_*.wav"))
    return [os.path.basename(f) for f in files]


def process_and_cache_data(filename, seq_length):
    processed_dir = os.path.join(DATA_DIR, 'processed', f'seq_{seq_length}')
    os.makedirs(processed_dir, exist_ok=True)

    file_path = os.path.join(DATA_DIR, filename)
    cache_x = os.path.join(processed_dir, f'X_chunks_{filename}.pt')
    cache_y = os.path.join(processed_dir, f'Y_chunks_{filename}.pt')

    if os.path.exists(cache_x) and os.path.exists(cache_y):
        return cache_x, cache_y

    X_vib_raw, Y_noise_raw, _ = load_primary_data(file_path)
    X_tensor = torch.tensor(X_vib_raw, dtype=torch.float32).t()
    Y_tensor = torch.tensor(Y_noise_raw, dtype=torch.float32).t()

    num_chunks = X_tensor.shape[1] // seq_length
    X_tensor = X_tensor[:, :num_chunks * seq_length]
    Y_tensor = Y_tensor[:, :num_chunks * seq_length]

    X_chunks = X_tensor.view(16, num_chunks, seq_length).transpose(0, 1)
    Y_chunks = Y_tensor.view(2, num_chunks, seq_length).transpose(0, 1)

    torch.save(X_chunks, cache_x)
    torch.save(Y_chunks, cache_y)
    return cache_x, cache_y


def get_impulse_responses():
    processed_dir = os.path.join(DATA_DIR, 'processed', 'acoustics')
    os.makedirs(processed_dir, exist_ok=True)
    cache_m1 = os.path.join(processed_dir, 'ir_m1.pt')
    cache_m2 = os.path.join(processed_dir, 'ir_m2.pt')

    if os.path.exists(cache_m1) and os.path.exists(cache_m2):
        return torch.load(cache_m1), torch.load(cache_m2)

    sec_l, sec_r = load_secondary_path(SEC_LEFT, SEC_RIGHT)
    ir_m1 = estimate_impulse_response(sec_l[:, 0], sec_l[:, 1])
    ir_m2 = estimate_impulse_response(sec_r[:, 0], sec_r[:, 2])

    torch.save(ir_m1, cache_m1)
    torch.save(ir_m2, cache_m2)
    return ir_m1, ir_m2


# ==========================================
# PAGE: DATA PROCESSING
# ==========================================
if page == "Data Processing":
    st.header("Data Processing & Caching")
    st.write("Scan raw primary `.wav` files and chunk them into sequence-specific datasets.")

    available_files = scan_primary_files()
    if not available_files:
        st.error("No Primary audio files found in the `data/` folder.")
    else:
        selected_files = st.multiselect("Select Primary Files to Process", available_files, default=available_files)
        seq_length = st.number_input("Sequence Length", min_value=100, max_value=4000, value=500, step=100)

        if st.button("Process & Cache Data"):
            progress_bar = st.progress(0)
            status = st.empty()
            get_impulse_responses()

            for i, file in enumerate(selected_files):
                status.text(f"Processing {file}...")
                cx, cy = process_and_cache_data(file, seq_length)

                st.markdown(f"### Visualizing: `{file}`")
                X_sample = torch.load(cx)[0]
                Y_sample = torch.load(cy)[0]

                c1, c2 = st.columns(2)
                with c1:
                    st.write("Reference Vibration (16 Ch)")
                    st.line_chart(X_sample.t().numpy())
                with c2:
                    st.write("Primary Noise (M1 & M2)")
                    st.line_chart(Y_sample.t().numpy())

                progress_bar.progress((i + 1) / len(selected_files))
            status.text("✅ Complete!")

# ==========================================
# PAGE: MODEL TRAINING
# ==========================================
elif page == "Model Training":
    st.header("Model Training")

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Select Architecture",
                                  ["WaveBreaker Basic (GRU)", "WaveBreaker Pro (TCN + Attention)"])
        seq_length = st.number_input("Sequence Length to Load", min_value=100, max_value=4000, value=500, step=100)
    with col2:
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=10)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=4)

    available_files = scan_primary_files()
    train_files = st.multiselect("Select Data for Training", available_files)

    if st.button("Start Training"):
        if not train_files:
            st.error("Please select data files.")
        else:
            ir_m1, ir_m2 = get_impulse_responses()
            all_X, all_Y = [], []
            for f in train_files:
                cx, cy = process_and_cache_data(f, seq_length)
                all_X.append(torch.load(cx))
                all_Y.append(torch.load(cy))

            X_tensor = torch.cat(all_X, dim=0)
            Y_tensor = torch.cat(all_Y, dim=0)
            dataset = TensorDataset(X_tensor, Y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            model = WaveBreakerANC() if "Basic" in model_type else WaveBreakerPro()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.MSELoss()

            metrics_text = st.container()
            loss_history = []

            model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                start_time = time.time()
                progress_bar = stqdm(dataloader, desc=f"Epoch {epoch + 1:02d}/{epochs}", leave=False)

                for batch_idx, (X_v, Y_p) in enumerate(progress_bar):
                    optimizer.zero_grad()
                    y_a = model(X_v)
                    y_p1 = apply_acoustic_path(y_a[:, 0:1, :], ir_m1)
                    y_p2 = apply_acoustic_path(y_a[:, 1:2, :], ir_m2)
                    y_p_c = torch.cat([y_p1, y_p2], dim=1)
                    loss = criterion(Y_p + y_p_c, torch.zeros_like(Y_p))
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    progress_bar.set_postfix(Loss=f"{loss.item():.6f}")

                avg_l = epoch_loss / len(dataloader)
                loss_history.append(avg_l)
                metrics_text.text(
                    f"--> Epoch {epoch + 1:02d} | Avg Loss: {avg_l:.6f} | Time: {time.time() - start_time:.2f}s")

            torch.save(model.state_dict(), os.path.join(DATA_DIR, "trained_model.pth"))
            st.session_state['trained_model_type'] = model_type

            st.markdown("---")
            st.subheader("Diagnostics: Convergence & Parameter Geometry")
            res_c1, res_c2 = st.columns(2)

            with res_c1:
                st.write("**Loss Convergence**")
                st.line_chart(loss_history)

            with res_c2:
                st.write("**Weights & Biases Geometry**")
                l_name = st.selectbox("Select Layer", [n for n, p in model.named_parameters()])
                param = dict(model.named_parameters())[l_name].data.cpu().numpy()

                if param.ndim >= 2:
                    if param.ndim > 2: param = param.reshape(param.shape[0], -1)
                    f2d = go.Figure(data=go.Heatmap(z=param, colorscale='Viridis'))
                    st.plotly_chart(f2d, use_container_width=True)
                    f3d = go.Figure(data=[go.Surface(z=param)])
                    st.plotly_chart(f3d, use_container_width=True)
                else:
                    st.line_chart(param)

# ==========================================
# PAGE: INFERENCING
# ==========================================
elif page == "Inferencing":
    st.header("Inferencing & Evaluation")
    m_path = os.path.join(DATA_DIR, "trained_model.pth")
    if not os.path.exists(m_path):
        st.warning("Train a model first.")
    else:
        st.success(f"Loaded: {st.session_state.get('trained_model_type', 'Model')}")
        test_f = st.selectbox("Select Inference Data", scan_primary_files())
        s_len = st.number_input("Seq Length", min_value=100, max_value=4000, value=500)

        if st.button("Run Inference"):
            ir1, ir2 = get_impulse_responses()
            cx, cy = process_and_cache_data(test_f, s_len)
            X_t, Y_t = torch.load(cx), torch.load(cy)

            m_type = st.session_state.get('trained_model_type', '')
            model = WaveBreakerANC() if "Basic" in m_type else WaveBreakerPro()
            model.load_state_dict(torch.load(m_path))
            model.eval()

            with torch.no_grad():
                y_a = model(X_t[0:1])
                y_p1 = apply_acoustic_path(y_a[:, 0:1, :], ir1)
                y_p2 = apply_acoustic_path(y_a[:, 1:2, :], ir2)
                err = Y_t[0:1] + torch.cat([y_p1, y_p2], dim=1)

                st.subheader("Interactive Acoustic Comparison (M1)")
                orig = Y_t[0, 0, :].numpy()
                resid = err[0, 0, :].numpy()
                st.line_chart(np.vstack([orig, resid]).T)

                red = (1 - (np.var(resid) / np.var(orig))) * 100
                st.metric("Power Reduction", f"{red:.2f}%")