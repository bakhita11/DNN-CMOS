#!/usr/bin/env python3
# ============================================================
# physics_residual_training.py
# Adds a BOUNDED-OUTPUT physics-residual penalty to the DNN surrogate loss
# and compares baseline (MSE only) vs physics-constrained (MSE + lambda*residual).
#
# Physical constraint enforced (soft penalty, differentiable):
#   The inverter output voltage must satisfy 0 <= Vout <= VDD.
#   Predictions outside this physical band are penalized quadratically.
#
# Reports for BOTH models on the held-out test set:
#   - R2, RMSE
#   - fraction of predictions outside the physical [0, VDD] band
#   - mean magnitude of bound violation
# Run in an environment with PyTorch (e.g., Colab).
# ============================================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(42)
torch.manual_seed(42)

def generate_dataset(num_samples=30000):
    L = np.random.uniform(2e-9, 100e-9, num_samples)
    W = np.random.uniform(0.2e-6, 10e-6, num_samples)
    VDD = np.random.uniform(0.5, 1.2, num_samples)
    Temp = np.random.uniform(250, 400, num_samples)
    Vth = np.random.normal(0.4, 0.05, num_samples)
    time = np.random.uniform(0, 1e-6, num_samples)
    current = (W / L) * np.maximum((VDD - Vth), 0)**2
    delay = L / (current + 1e-9)
    Vout = VDD / (1 + np.exp(-12 * (time - delay)))
    Vout = Vout + np.random.normal(0, 0.015, num_samples)
    return dict(L=L, W=W, VDD=VDD, Temp=Temp, Vth=Vth, time=time, Vout=Vout)

d = generate_dataset()
X = np.column_stack([d['L'], d['W'], d['VDD'], d['Temp'], d['Vth'], d['time'],
                     d['W']/d['L'], d['VDD']-d['Vth']])
y = d['Vout'].reshape(-1, 1)
vdd_col = d['VDD'].reshape(-1, 1)

Xs = MinMaxScaler(); ys = MinMaxScaler()
Xn = Xs.fit_transform(X); yn = ys.fit_transform(y)

idx = np.arange(len(Xn))
Xtr, Xte, ytr, yte, itr, ite = train_test_split(Xn, yn, idx, test_size=0.2, random_state=42)

Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
ytr_t = torch.tensor(ytr, dtype=torch.float32)
Xte_t = torch.tensor(Xte, dtype=torch.float32)

y_min = ys.data_min_[0]; y_max = ys.data_max_[0]
def to_norm(v): return (v - y_min) / (y_max - y_min)
lower_n = torch.tensor(to_norm(0.0), dtype=torch.float32)
upper_tr = torch.tensor(to_norm(vdd_col[itr]), dtype=torch.float32)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8,128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,1))
    def forward(self,x): return self.net(x)

mse = nn.MSELoss()

def bound_penalty(pred):
    over  = torch.relu(pred - upper_tr)
    under = torch.relu(lower_n - pred)
    return (over**2 + under**2).mean()

def train(use_physics, lam=10.0, epochs=150):
    torch.manual_seed(42)
    model = Net()
    opt = optim.Adam(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=10)
    for ep in range(epochs):
        model.train()
        out = model(Xtr_t)
        loss = mse(out, ytr_t)
        if use_physics:
            loss = loss + lam * bound_penalty(out)
        opt.zero_grad(); loss.backward(); opt.step()
        sched.step(loss)
    return model

def evaluate(model, tag):
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).numpy()
    pred_real = ys.inverse_transform(pred)
    true_real = ys.inverse_transform(yte)
    r2 = r2_score(true_real, pred_real)
    rmse = np.sqrt(mean_squared_error(true_real, pred_real))
    vdd_te = vdd_col[ite].flatten()
    pr = pred_real.flatten()
    frac_oob = np.mean((pr < -0.005) | (pr > vdd_te + 0.005))
    viol = np.maximum(0, pr - vdd_te) + np.maximum(0, -pr)
    mean_viol = viol.mean()
    print(f"\n===== {tag} =====")
    print(f"  Test R2                          : {r2:.4f}")
    print(f"  Test RMSE                        : {rmse:.4f}")
    print(f"  Predictions outside [0,VDD]      : {frac_oob*100:.3f}%")
    print(f"  Mean bound-violation magnitude   : {mean_viol*1e3:.3f} mV")
    return dict(r2=r2, rmse=rmse, oob=frac_oob, viol=mean_viol)

if __name__ == "__main__":
    print("Training baseline (MSE only)...")
    m_base = train(use_physics=False)
    r_base = evaluate(m_base, "Baseline (MSE only)")

    print("\nTraining physics-constrained (MSE + bounded-output penalty)...")
    m_phys = train(use_physics=True)
    r_phys = evaluate(m_phys, "Physics-constrained")

    print("\n===== SUMMARY =====")
    print(f"{'':<34}{'Baseline':>12}{'Physics':>12}")
    print(f"{'Test R2':<34}{r_base['r2']:>12.4f}{r_phys['r2']:>12.4f}")
    print(f"{'Out-of-bound predictions (%)':<34}{r_base['oob']*100:>12.3f}{r_phys['oob']*100:>12.3f}")
    print(f"{'Mean violation (mV)':<34}{r_base['viol']*1e3:>12.3f}{r_phys['viol']*1e3:>12.3f}")