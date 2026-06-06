#!/usr/bin/env python3
# ============================================================
# physics_bounded_training.py
# Enforces the physical constraint 0 <= Vout <= VDD by CONSTRUCTION using a
# hard architectural bound (scaled sigmoid output), guaranteeing physically
# admissible predictions everywhere -- in-distribution AND out-of-distribution.
#
# Compares:
#   Baseline      : unconstrained linear output (standard DNN surrogate)
#   Bounded model : final output = VDD * sigmoid(z)  -> always in (0, VDD)
#
# Reports R2 (in-distribution) and the fraction / magnitude of physically
# out-of-bound predictions on an out-of-distribution (OOD) probe -- the regime
# a Bayesian optimizer can push into during design-space exploration.
#
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

# scale INPUTS only; keep the target in PHYSICAL volts so the bounded model can
# multiply its sigmoid by the physical VDD carried alongside each sample.
Xs = MinMaxScaler()
Xn = Xs.fit_transform(X)

idx = np.arange(len(Xn))
(Xtr, Xte, ytr, yte, vtr, vte, itr, ite) = train_test_split(
    Xn, y, vdd_col, idx, test_size=0.2, random_state=42)

Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
ytr_t = torch.tensor(ytr, dtype=torch.float32)         # physical volts
Xte_t = torch.tensor(Xte, dtype=torch.float32)
vtr_t = torch.tensor(vtr, dtype=torch.float32)         # physical VDD per sample
vte_t = torch.tensor(vte, dtype=torch.float32)

# normalize the loss by a fixed scale so MSE magnitudes are comparable
YSCALE = float(y.max())

class BaselineNet(nn.Module):
    """Unconstrained linear output (standard surrogate)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8,128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,1))
    def forward(self, x, vdd):
        return self.net(x)                       # unbounded volts

class BoundedNet(nn.Module):
    """Hard physical bound: output = VDD * sigmoid(z) in (0, VDD) by construction."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8,128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,1))
    def forward(self, x, vdd):
        z = self.net(x)
        return vdd * torch.sigmoid(z)            # guaranteed in (0, VDD)

mse = nn.MSELoss()

def train(model, epochs=150):
    torch.manual_seed(42)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=10)
    for ep in range(epochs):
        model.train()
        out = model(Xtr_t, vtr_t)
        loss = mse(out / YSCALE, ytr_t / YSCALE)
        opt.zero_grad(); loss.backward(); opt.step()
        sched.step(loss.item())
    return model

def evaluate_id(model, tag):
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t, vte_t).numpy().flatten()
    true = yte.flatten()
    r2 = r2_score(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    vdd_te = vte.flatten()
    frac_oob = np.mean((pred < -0.005) | (pred > vdd_te + 0.005))
    print(f"\n===== {tag} : in-distribution test set =====")
    print(f"  Test R2                       : {r2:.4f}")
    print(f"  Test RMSE                     : {rmse:.4f}")
    print(f"  Predictions outside [0,VDD]   : {frac_oob*100:.3f}%")
    return dict(r2=r2, rmse=rmse, oob=frac_oob)

def make_ood_probe(n=5000):
    rng = np.random.default_rng(7)
    L = rng.uniform(100e-9, 300e-9, n)
    W = rng.uniform(10e-6, 30e-6, n)
    VDD = rng.uniform(1.2, 2.0, n)
    Temp = rng.uniform(200, 450, n)
    Vth = rng.normal(0.4, 0.08, n)
    time = rng.uniform(0, 2e-6, n)
    Xo = np.column_stack([L, W, VDD, Temp, Vth, time, W/L, VDD - Vth])
    return Xo, VDD

def evaluate_ood(model, tag):
    Xo, vdd_o = make_ood_probe()
    Xo_n = Xs.transform(Xo)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xo_n, dtype=torch.float32),
                     torch.tensor(vdd_o.reshape(-1,1), dtype=torch.float32)).numpy().flatten()
    frac_oob = np.mean((pred < -0.005) | (pred > vdd_o + 0.005))
    viol = np.maximum(0, pred - vdd_o) + np.maximum(0, -pred)
    print(f"\n----- {tag} : OUT-OF-DISTRIBUTION probe -----")
    print(f"  Predictions outside [0,VDD]   : {frac_oob*100:.3f}%")
    print(f"  Mean bound-violation (mV)     : {viol.mean()*1e3:.3f}")
    print(f"  Max  bound-violation (mV)     : {viol.max()*1e3:.3f}")
    return dict(oob=frac_oob, viol=viol.mean(), maxviol=viol.max())

if __name__ == "__main__":
    print("Training baseline (unconstrained output)...")
    base = train(BaselineNet())
    rb = evaluate_id(base, "Baseline (unconstrained)")
    ob = evaluate_ood(base, "Baseline (unconstrained)")

    print("\nTraining bounded model (hard VDD*sigmoid output)...")
    bnd = train(BoundedNet())
    rp = evaluate_id(bnd, "Bounded (VDD*sigmoid)")
    op = evaluate_ood(bnd, "Bounded (VDD*sigmoid)")

    print("\n===== SUMMARY =====")
    print(f"{'':<36}{'Baseline':>12}{'Bounded':>12}")
    print(f"{'Test R2 (in-distribution)':<36}{rb['r2']:>12.4f}{rp['r2']:>12.4f}")
    print(f"{'In-dist out-of-bound (%)':<36}{rb['oob']*100:>12.3f}{rp['oob']*100:>12.3f}")
    print(f"{'OOD out-of-bound (%)':<36}{ob['oob']*100:>12.3f}{op['oob']*100:>12.3f}")
    print(f"{'OOD mean violation (mV)':<36}{ob['viol']*1e3:>12.3f}{op['viol']*1e3:>12.3f}")
    print(f"{'OOD max violation (mV)':<36}{ob['maxviol']*1e3:>12.3f}{op['maxviol']*1e3:>12.3f}")
