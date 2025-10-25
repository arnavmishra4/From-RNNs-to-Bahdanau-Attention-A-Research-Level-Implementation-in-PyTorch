# 🔥 RNN-LSTM-Attention: Sequence Modeling in PyTorch

This repository is part of my **PyTorch Research Mastery Series** — a self-driven exercise to implement, document, and modularize core sequence-model architectures from scratch using **pure PyTorch**.

It includes custom implementations of:
- A fully parameterized **RNNCell** (without relying on `nn.RNN`)
- A modular **CustomLSTM** with bidirectional support
- An additive **Attention Mechanism** (Bahdanau-style)
- A composite **LSTM + Attention** model designed for flexible sequence tasks  

All models are written **from first principles**, emphasizing **clarity, reproducibility, and research-grade structure**.

---

## 🧠 Overview

The goal of this project was to understand and re-implement the fundamental building blocks of sequence modeling architectures used in modern AI research.  
Each component is structured in an isolated module (`models/`), with clean data abstraction, configuration, and training logic — mirroring FAIR and DeepMind-style internal repos.

The full pipeline supports **sequence regression/classification** tasks and can handle any CSV-based or tensor-based sequential dataset.

---

## 🧩 Architecture Components

| Component | File | Description |
|------------|------|-------------|
| 🧱 `RNNCell` | `models/rnn_cell.py` | Implements a basic recurrent cell using explicit weight matrices and tanh activation |
| ⚙️ `CustomLSTM` | `models/custom_lstm.py` | Modular wrapper around PyTorch's `nn.LSTM` with support for bidirectional passes |
| 🎯 `CustomAttention` | `models/attention.py` | Implements additive attention with trainable parameters (W_a, U_a, v_a) |
| 🧩 `LSTMAttention` | `models/lstm_attention.py` | Combines LSTM encoder with attention for context-aware output |
| 📦 `data_loader.py` |  | Dataset + DataLoader utilities for real sequential datasets (e.g., time series, signals, etc.) |
| ⚙️ `config.py` |  | Global hyperparameter configuration (hidden size, layers, LR, etc.) |
| 🧮 `utils/metrics.py` |  | Evaluation metrics (MSE, RMSE, R²) for regression-type outputs |

---

## 🧭 Folder Structure

```
rnn_lstm_attention/
│
├── main.py
├── config.py
├── data_loader.py
│
├── models/
│   ├── rnn_cell.py
│   ├── custom_lstm.py
│   ├── attention.py
│   ├── lstm_attention.py
│
├── utils/
│   ├── metrics.py
│
├── data/
│   └── your_dataset.csv
└── README.md
```

---

## 🧪 Training

Run the main script with any real dataset (CSV or tensor).  
Each row in the dataset should represent a flattened sequence, and the last column should contain the target label/value.

```bash
python main.py --model lstm_attention --data_path ./data/your_dataset.csv --epochs 100
```

You can also experiment with:

```bash
python main.py --model rnn
```

---

## ⚙️ Configuration

Edit `config.py` to modify hyperparameters:

```python
class Config:
    input_size = 1
    hidden_size = 32
    output_size = 1
    num_layers = 1
    bidirectional = True
    batch_size = 16
    lr = 1e-3
```

---

## 📊 Metrics & Evaluation

During evaluation, the framework computes:

- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **R² Score** — Coefficient of Determination

Metrics are logged directly in the terminal during evaluation.

---

## 🧱 Research Motivation

This repository is part of my ongoing **PyTorch Deep Research Grind** — a personal mastery track to understand AI model internals at a research-ready level.

Instead of relying on high-level libraries, this project breaks down every architectural component (recurrent units, gating mechanisms, attention scoring, etc.) to achieve *first-principles understanding* of how modern sequence models function.

> The objective wasn't to build a flashy demo — it was to learn how things actually work under the hood, the way DeepMind or FAIR research engineers would.

---

## 💡 Key Learnings

- Built recurrent networks directly from matrix operations (`Wxh`, `Whh`, etc.)
- Understood hidden state propagation and gradient flow through time
- Integrated attention mechanisms for dynamic sequence weighting
- Implemented modular, reusable PyTorch components with clean documentation
- Structured research-grade training pipelines (config, utils, loaders, etc.)

---

## 🧰 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```
torch
numpy
pandas
scikit-learn
```

---

## 🧾 Citation / Usage

If you use this implementation as a reference for your own research or coursework, please cite:

```
Arnav Mishra, "RNN–LSTM–Attention: A Modular Sequence Modeling Implementation in PyTorch (Research Exercise Repository)", 2025.
```

---

## 🧠 Future Additions

- GRU and Transformer-based extensions
- Multi-feature sequence support
- Layer-wise introspection and visualization
- TorchScript / ONNX export utilities

---

## 🏁 Author

**Arnav Mishra**  
AI Researcher · PyTorch Core & Sequence Modeling Enthusiast  
Bhopal, India

---

> *"The only way to truly master PyTorch is to break it open and rebuild it yourself."*  
> — A guiding thought behind this repository.
