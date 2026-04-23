import torch
import torch.nn as nn
from torch.autograd import grad
from torch.optim import Adam
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from tqdm import tqdm
import time
import os
import json
import datetime
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class FeedForward(nn.Module):
    def __init__(self, layers):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(modules[-1].weight, gain=1.0)
            nn.init.zeros_(modules[-1].bias)
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
    def forward(self, x):
        return self.net(x)


class PINN:
    def __init__(self, X, config):
        self.model_type = config['model_type']
        self.constraint_type = config['constraint_type']
        self.hole_c = config["hole_c"]
        self.L_fixed = float(config.get('L_fixed', 2.5))
        self.L_fixed_t = torch.tensor(self.L_fixed, dtype=torch.float32, device=device)


        # Parameters required for normalization
        self.max_X = X.max(axis=0)
        self.min_X = X.min(axis=0)

        # --- inputs: (x, y, cx, cy, r) five dimensions ---
        self.x = torch.tensor(X[:, 0:1], dtype=torch.float32, device=device)
        self.y = torch.tensor(X[:, 1:2], dtype=torch.float32, device=device)
        self.cx = torch.tensor(X[:, 2:3], dtype=torch.float32, device=device)
        self.cy = torch.tensor(X[:, 3:4], dtype=torch.float32, device=device)
        self.r = torch.tensor(X[:, 4:5], dtype=torch.float32, device=device)

        # physical constants
        self.q = 100.0
        self.v = 0.0
        self.E = 12000000.0
        self.t = 0.01
        self.D0 = (self.E * self.t**3) / (12.0 * (1 - self.v**2))
        D_np = self.D0 * np.array([[1.0, self.v, 0.0],
                                   [self.v, 1.0, 0.0],
                                   [0.0, 0.0, (1 - self.v) / 2.0]])
        self.D = torch.tensor(D_np, dtype=torch.float32, device=device)
        
        self.layers = config["layers"]
        self.net = FeedForward(self.layers).to(device)
        
        params = list(self.net.parameters())
        self.optimizer = Adam(params, lr=config['lr'])
        self.lbfgs = optim.LBFGS(params,
                                 lr=1.0,
                                 max_iter=50,
                                 max_eval=80,
                                 history_size=100,
                                 line_search_fn='strong_wolfe')
        self.loss_c_log = []
        self.loss_f_log = []

    #  Normalization
    def normalize(self, inp):
        min_X_t = torch.tensor(self.min_X, dtype=torch.float32, device=device)
        max_X_t = torch.tensor(self.max_X, dtype=torch.float32, device=device)
        return 2.0 * (inp - min_X_t) / (max_X_t - min_X_t + 1e-12) - 1.0

    def net_forward(self, x, y, cx, cy, r):
        # concat
        inp_params = torch.cat([x, y, cx, cy, r], dim=1)
        inp_params_n = self.normalize(inp_params)
        
   
        # hard constrain boundary，containing the inner hole
        g_outer = x * (self.L_fixed_t - x) * y * (1.0 - y)
        g_inner = (x - cx)**2 + (y - cy)**2 - r**2
        if self.hole_c == "clamped":
            boundary_func = (g_outer * g_inner)**2  
        if self.hole_c == "supported": 
            boundary_func= g_inner * (g_outer )**2 
        if self.hole_c == "free": 
            boundary_func = (g_outer )**2  
        u = boundary_func * self.net(inp_params_n)
        return u

    def compute_derivatives(self, x, y, cx, cy, r):
        x_clone = x.clone().detach().requires_grad_(True)
        y_clone = y.clone().detach().requires_grad_(True)
        cx_clone, cy_clone, r_clone = [t.clone().detach() for t in (cx, cy, r)]
        u = self.net_forward(x_clone, y_clone, cx_clone, cy_clone, r_clone)
        ones_u = torch.ones_like(u)
        u_x, u_y = grad(u, (x_clone, y_clone), grad_outputs=ones_u, create_graph=True)
        u_xx = grad(u_x, x_clone, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y_clone, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_xy = grad(u_x, y_clone, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        return {'u': u, 'u_x': u_x, 'u_y': u_y, 'u_xx': u_xx, 'u_yy': u_yy, 'u_xy': u_xy}


    def loss_terms(self):
        derivs_f = self.compute_derivatives(self.x, self.y, self.cx, self.cy, self.r)
        
        u, u_xx, u_yy, u_xy = derivs_f['u'], derivs_f['u_xx'], derivs_f['u_yy'], derivs_f['u_xy']
        k = torch.cat([-u_xx, -u_yy, -2.0 * u_xy], dim=1)
        M = torch.matmul(k, self.D)
        
        strain_energy_density = 0.5 * torch.sum(M * k, dim=1, keepdim=True)
        external_work_density = self.q * u
        potential_energy_density = strain_energy_density - external_work_density
        
        # total_potential_energy = potential_energy_density / (self.L_fixed_t - torch.pi * self.r * self.r)
        total_potential_energy = potential_energy_density 
        loss_f = total_potential_energy.mean()
        return loss_f

    # --- train and train_lbfgs ---
    def train(self, config):
        epochs = config["epochs"]
        print_every = config["print_every"]
        for it in tqdm(range(epochs), desc="Adam Training"):
            self.optimizer.zero_grad()
            loss_f = self.loss_terms()
            loss = loss_f
            loss.backward()
            self.optimizer.step()
            self.loss_f_log.append(loss_f.item()) 
            if it % print_every == 0 or it == epochs - 1:
                print(f'It: {it:d}, Loss_f: {loss_f.item():.3e}')

    def train_lbfgs(self, config):
        epochs = config["epochs_lbfgs"]
        print_every = config["print_every"]
        start_time = time.time()
        def closure():
            self.lbfgs.zero_grad()
            loss_f = self.loss_terms()
            loss =  loss_f
            loss.backward()
            self.loss_f_log.append(loss_f.item())
            return loss 

        for it in tqdm(range(epochs)):

            self.lbfgs.step(closure)
            if it % print_every == 0 or it == epochs - 1:
                elapsed = time.time() - start_time
                print(f'It(LBFGS): {it:d}, Loss_f: {self.loss_f_log[-1]:.3e}, Time: {elapsed:.2f}')
                start_time = time.time()

    def predict(self, x_np, y_np, cx_np, cy_np, r_np):
        self.net.eval()
        x = torch.tensor(x_np, dtype=torch.float32, device=device, requires_grad=True)
        y = torch.tensor(y_np, dtype=torch.float32, device=device, requires_grad=True)
        cx = torch.tensor(cx_np, dtype=torch.float32, device=device)
        cy = torch.tensor(cy_np, dtype=torch.float32, device=device)
        r = torch.tensor(r_np, dtype=torch.float32, device=device)
        
        derivs = self.compute_derivatives(x, y, cx, cy, r)
        u = derivs['u']
        u_xx =  derivs['u_xx'];u_yy =  derivs['u_yy'];u_xy =  derivs['u_xy']
        self.net.train()
        return u.detach().cpu().numpy(),u_xx.detach().cpu().numpy(), u_yy.detach().cpu().numpy(),u_xy.detach().cpu().numpy()


    def evaluate(self, cx_test, cy_test, r_test):
        print(f"\n--- Evaluating for L={self.L_fixed}, cx={cx_test}, cy={cy_test}, r={r_test} ---")
        n_grid = 201
        x_space = np.linspace(0, self.L_fixed, n_grid)
        y_space = np.linspace(0, 1, n_grid)
        x_grid, y_grid = np.meshgrid(x_space, y_space)
        
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()

        dist_sq = (x_flat - cx_test)**2 + (y_flat - cy_test)**2
        valid_indices = dist_sq >= r_test**2
        
        x_pred = x_flat[valid_indices][:, None]
        y_pred = y_flat[valid_indices][:, None]

        cx_pred = np.full_like(x_pred, cx_test)
        cy_pred = np.full_like(x_pred, cy_test)
        r_pred = np.full_like(x_pred, r_test)

        u_pred,u_xx_pred,u_yy_pred,u_xy_pred = self.predict(x_pred, y_pred, cx_pred, cy_pred, r_pred)
        u_pred,u_xx_pred,u_yy_pred,u_xy_pred =u_pred.flatten(),u_xx_pred.flatten(),u_yy_pred.flatten(),u_xy_pred.flatten()

        plt.figure(figsize=(8, 5), dpi=150)
        plt.tricontourf(x_pred.flatten(), y_pred.flatten(), u_pred, levels=50, cmap='Spectral')
        cbar = plt.colorbar()
        cbar.set_label('Predicted Deflection u(x, y)')
        
        circle = plt.Circle((cx_test, cy_test), r_test, color='white', fill=True)
        plt.gca().add_artist(circle)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'PINN Prediction for Plate with Hole\nL={self.L_fixed}, center=({cx_test:.2f}, {cy_test:.2f}), r={r_test:.2f}')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5), dpi=150)
        plt.tricontourf(x_pred.flatten(), y_pred.flatten(), u_xx_pred, levels=50, cmap='Spectral')
        cbar = plt.colorbar()
        cbar.set_label('Predicted Deflection u_xx(x, y)')
        
        circle = plt.Circle((cx_test, cy_test), r_test, color='white', fill=True)
        plt.gca().add_artist(circle)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'PINN Prediction for Plate with Hole\nL={self.L_fixed}, center=({cx_test:.2f}, {cy_test:.2f}), r={r_test:.2f}')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5), dpi=150)
        plt.tricontourf(x_pred.flatten(), y_pred.flatten(), u_yy_pred, levels=50, cmap='Spectral')
        cbar = plt.colorbar()
        cbar.set_label('Predicted Deflection u_yy(x, y)')

        circle = plt.Circle((cx_test, cy_test), r_test, color='white', fill=True)
        plt.gca().add_artist(circle)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'PINN Prediction for Plate with Hole\nL={self.L_fixed}, center=({cx_test:.2f}, {cy_test:.2f}), r={r_test:.2f}')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5), dpi=300)
        plt.tricontourf(x_pred.flatten(), y_pred.flatten(), u_xy_pred, levels=50, cmap='Spectral')
        cbar = plt.colorbar()
        cbar.set_label('Predicted Deflection u_xy(x, y)')
        
        # Draw circular hole
        circle = plt.Circle((cx_test, cy_test), r_test, color='white', fill=True)
        plt.gca().add_artist(circle)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'PINN Prediction for Plate with Hole\nL={self.L_fixed}, center=({cx_test:.2f}, {cy_test:.2f}), r={r_test:.2f}')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


def relative_l2_error(y_true, y_pred):
    return np.sqrt(np.sum((y_pred - y_true) ** 2)) / np.sqrt(np.sum(y_true ** 2))


def make_data_Sobol_plate_with_hole(config):
    n_points_target = config["n_domain"]
    L_fixed = float(config.get("L_fixed", 2.5))
    cx_rel_range = config["cx_rel_range"] # range of cx relative to L
    cy_range = config["cy_range"]
    r_range = config["r_range"]
    
    print("Generating training data for plate with hole using Sobol sequence (L fixed)...")
    
    # Sobol d=5: (x_hat, y, cx_rel, cy, r)
    d = 5
    sobol_engine = qmc.Sobol(d=d, scramble=True, seed=config['seed'])
    
    lower_bounds = [0, 0, cx_rel_range[0], cy_range[0], r_range[0]]
    upper_bounds = [1, 1, cx_rel_range[1], cy_range[1], r_range[1]]

    X_valid = []

    n_generated = 0
    n_to_sample_initially = int(n_points_target * 2.0) 
    
    with tqdm(total=n_points_target, desc="Accept-Reject Sampling") as pbar:
        while len(X_valid) < n_points_target:
            if n_generated % n_to_sample_initially == 0:
                 uniform_samples = sobol_engine.random(n=n_to_sample_initially)
                 X_scaled = qmc.scale(uniform_samples, lower_bounds, upper_bounds)

            idx_in_batch = n_generated % n_to_sample_initially
            sample = X_scaled[idx_in_batch]

            x_hat, y, cx_rel, cy, r = sample

            x = x_hat * L_fixed
            cx = cx_rel * L_fixed #
            
            if (x - cx)**2 + (y - cy)**2 >= r**2:
                X_valid.append([x, y, cx, cy, r])
                pbar.update(1)

            n_generated += 1
            if n_generated > n_points_target * 50: 
                print("\nWarning: Rejection rate is very high. Check geometry ranges.")
                break

    print(f"\nGenerated {len(X_valid)} valid data points from {n_generated} candidates.")
    return np.array(X_valid)


def make_data_Random_plate_with_hole(config):
    n_points_target = config["n_domain"]
    L_fixed = float(config.get("L_fixed", 2.5))
    cx_rel_range = config["cx_rel_range"]  
    cy_range = config["cy_range"]
    r_range = config["r_range"]

    rng = np.random.default_rng(config['seed'])
    lower_bounds = [0, 0, cx_rel_range[0], cy_range[0], r_range[0]]
    upper_bounds = [1, 1, cx_rel_range[1], cy_range[1], r_range[1]]

    X_valid = []
    n_generated = 0
    n_to_sample_initially = int(n_points_target * 2.0) 

    with tqdm(total=n_points_target, desc="Accept-Reject Sampling (Random)") as pbar:
        while len(X_valid) < n_points_target:
            if n_generated % n_to_sample_initially == 0:
                U = rng.random((n_to_sample_initially, 5))
                X_scaled = qmc.scale(U, lower_bounds, upper_bounds)
            idx_in_batch = n_generated % n_to_sample_initially
            sample = X_scaled[idx_in_batch]
            x_hat, y, cx_rel, cy, r = sample
            x = x_hat * L_fixed
            cx = cx_rel * L_fixed  
            if (x - cx)**2 + (y - cy)**2 >= r**2:
                X_valid.append([x, y, cx, cy, r])
                pbar.update(1)

            n_generated += 1
            if n_generated > n_points_target * 50:
                print("\nWarning: Rejection rate is very high. Check geometry ranges.")
                break

    print(f"\nGenerated {len(X_valid)} valid data points from {n_generated} candidates (Random).")
    return np.array(X_valid)


def main(config):
    save_dir = config["save_dir"]
    total_start = time.time()
    if config["sample_method"] =="make_data_Sobol_plate_with_hole":
        X_star = make_data_Sobol_plate_with_hole(config)
    if config["sample_method"] =="make_data_Random_plate_with_hole":
        X_star = make_data_Random_plate_with_hole(config)

    model = PINN(X_star, config)
    
    model.train(config)
    model.train_lbfgs(config)
    torch.save(model.net.state_dict(), os.path.join(save_dir, "model.pth"))
    total_time = time.time()-total_start
    config["total_time"] = total_time

    # --- save config and loss ---
    config["min_X"] = model.min_X.tolist()
    config["max_X"] = model.max_X.tolist()
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    loss_df = pd.DataFrame({"loss_f": model.loss_f_log})
    loss_df.to_csv(os.path.join(save_dir, "loss_log.csv"), index=False)
    print(f"\nModel and logs saved to: {save_dir}")
    
    if config.get("run_eval", True):
        model.evaluate(cx_test=0.5, cy_test=0.5, r_test=0.2)
        model.evaluate(cx_test=0.2, cy_test=0.5, r_test=0.1)
        model.evaluate(cx_test=0.5, cy_test=0.5, r_test=0.1)


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("results_hole", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    config = {
        "save_dir": save_dir,
        "seed": 230258995,  
        "n_domain": 2**18, 
        "sample_method" :"make_data_Sobol_plate_with_hole",

        "L_fixed": 2.5,
        "cx_rel_range": (0.1, 0.9),  
        "cy_range": (0.1, 0.9),      
        "r_range": (0.061, 0.361),     
        "model_type": 'GAPINN',
        "constraint_type": 'hard',
        "hole_c":"free",  # clamped supported free

        #  inputs :5 (x, y, cx, cy, r)
        # "layers": [5] + 4*[50] + [1], 
        "layers": [5] + 2*[150] + [1], 
        
        "epochs": 0, 
        "epochs_lbfgs": 200, 
        "print_every": 1,
        "lr": 1e-3, 
    }
    print("config：",config)
    
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    main(config=config)
