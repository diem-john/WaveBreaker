<h1>🌊 WaveBreaker </h1>
© D.J. Medina (2026)

___
⚙️ Installation <br>
WaveBreaker requires Python 3.8+ and runs best inside an Anaconda environment.

1. Create Conda a Conda Environment
```
conda create -n WaveBreaker python=3.10
conda activate WaveBreaker
```
2. Install Core Dependencies
```
# Example for CUDA 11.8 (Check pytorch.org for your specific system)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install scientific and audio processing libraries
conda install -c conda-forge pysoundfile numpy scipy matplotlib
```
3. Install Dashboard Dependencies
```
conda install -c conda-forge streamlit
pip install stqdm
```

___
🎮 How to Use
1. Option A: The Streamlit Dashboard (Recommended) <br>
The dashboard provides a visual, interactive way to manage the entire ANC pipeline.
```
streamlit run app.py
```
2. Option B: The Command Line Interface
If you prefer running training jobs directly from the terminal (e.g., on a remote server), use `main.py`.
```
python main.py
```