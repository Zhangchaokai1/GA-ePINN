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

# --- Check GPU availability ---
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
        
      

        # # --- Input data has six dimensions (x, y, L, cx, cy, r) ---
        self.x = torch.tensor(X[:, 0:1], dtype=torch.float32, device=device)
        self.y = torch.tensor(X[:, 1:2], dtype=torch.float32, device=device)
        self.L = torch.tensor(X[:, 2:3], dtype=torch.float32, device=device)
        self.cx = torch.tensor(X[:, 3:4], dtype=torch.float32, device=device)
        self.cy = torch.tensor(X[:, 4:5], dtype=torch.float32, device=device)
        self.r = torch.tensor(X[:, 5:6], dtype=torch.float32, device=device)


        # # 2. Parameters needed for normalization
        ## Fetch absolute ranges for each base parameter from config
        # L_min, L_max = config["L_range"]
        # cx_rel_min, cx_rel_max = config["cx_rel_range"]
        # cy_min, cy_max = config["cy_range"]
        # r_min, r_max = config["r_range"]
        # x_rel_min, x_rel_max = 0.0, 1.0  # x relative range is always [0, 1]
        # y_min, y_max = 0.0, 1.0          # y physical range is fixed at [0, 1]
        # min_vals = [x_rel_min, y_min, L_min, cx_rel_min, cy_min, r_min]
        # max_vals = [x_rel_max, y_max, L_max, cx_rel_max, cy_max, r_max]
        # self.min_X = torch.tensor(min_vals, dtype=torch.float32, device=device)
        # self.max_X = torch.tensor(max_vals, dtype=torch.float32, device=device)

        # 2. Parameters required for normalization
        self.max_X = X.max(axis=0)
        self.min_X = X.min(axis=0)

        # Physical constants 
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

    #  Normalize
    def normalize(self, inp):
        min_X_t = torch.tensor(self.min_X, dtype=torch.float32, device=device)
        max_X_t = torch.tensor(self.max_X, dtype=torch.float32, device=device)
        return 2.0 * (inp - min_X_t) / (max_X_t - min_X_t + 1e-12) - 1.0

    def net_forward(self, x, y, L, cx, cy, r):
        # Concatenate all parameters and normalize
        inp_params = torch.cat([x, y, L, cx, cy, r], dim=1)
        inp_params_n = self.normalize(inp_params)
        
        # # 1. Outer boundary function (rectangle)
        # if self.config["BC_b"]== "clamped":
        #     g_outer = (x * (L - x) * y * (1.0 - y))**2
        # if self.config["BC_b"]== "support":
        #     g_outer = x * (L - x) * y * (1.0 - y)
        # if self.config["BC_b"]== "free":
        #     g_outer = 1.0
        
        # # 2. Inner boundary function (circular hole)
        # if self.config["BC_hole"]== "clamped":
        #     g_inner = ((x - cx)**2 + (y - cy)**2 - r**2)**2
        # if self.config["BC_hole"]== "support":
        #     g_inner = (x - cx)**2 + (y - cy)**2 - r**2
        # if self.config["BC_hole"]== "free":
        #     g_inner = 1.0

        g_outer = (x * (L - x) * y * (1.0 - y))**2
        g_inner = ((x - cx)**2 + (y - cy)**2 - r**2)**2

        # 3. Combine boundaries
        boundary_func = g_outer * g_inner
        u = boundary_func * self.net(inp_params_n)
        return u

    def compute_derivatives(self, x, y, L, cx, cy, r):
        x_clone = x.clone().detach().requires_grad_(True)
        y_clone = y.clone().detach().requires_grad_(True)
        # Other parameters do not need gradients
        L_clone, cx_clone, cy_clone, r_clone = [t.clone().detach() for t in (L, cx, cy, r)]
        u = self.net_forward(x_clone, y_clone, L_clone, cx_clone, cy_clone, r_clone)
        ones_u = torch.ones_like(u)
        u_x, u_y = grad(u, (x_clone, y_clone), grad_outputs=ones_u, create_graph=True)
        u_xx = grad(u_x, x_clone, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y_clone, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_xy = grad(u_x, y_clone, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        return {'u': u, 'u_x': u_x, 'u_y': u_y, 'u_xx': u_xx, 'u_yy': u_yy, 'u_xy': u_xy}


    def loss_terms(self):
        derivs_f = self.compute_derivatives(self.x, self.y, self.L, self.cx, self.cy, self.r)
        
        u, u_xx, u_yy, u_xy = derivs_f['u'], derivs_f['u_xx'], derivs_f['u_yy'], derivs_f['u_xy']
        k = torch.cat([-u_xx, -u_yy, -2.0 * u_xy], dim=1)
        M = torch.matmul(k, self.D)
        
        strain_energy_density = 0.5 * torch.sum(M * k, dim=1, keepdim=True)
        external_work_density = self.q * u
        potential_energy_density = strain_energy_density - external_work_density
        
        # --- Monte Carlo area = rectangle area - hole area ---
        # Using the mean total energy as the loss remains equivalent for optimization
        total_potential_energy = potential_energy_density /(self.L -torch.pi*self.r*self.r)
        # total_potential_energy = potential_energy_density /(self.L) 
        loss_f = total_potential_energy.mean()
        # self.loss_f_log.append(loss_f.item()) # Logged during the training loop
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
            self.loss_f_log.append(loss_f.item())  # Log Adam loss
            if it % print_every == 0 or it == epochs - 1:
                print(f'It: {it:d}, Loss_f: {loss_f.item():.3e}')

    def train_lbfgs(self, config):
        epochs = config["epochs_lbfgs"]
        def closure():
            self.lbfgs.zero_grad()
            loss_f = self.loss_terms()
            loss = loss_f
            loss.backward()
            return loss
        for it in tqdm(range(epochs), desc="LBFGS Training"):
            self.lbfgs.step(closure)
            loss_f = self.loss_terms() 
            self.loss_f_log.append(loss_f.item()) # Log LBFGS loss

            if it % 1 == 0 or it == epochs - 1: 
                 print(f'It(LBFGS): {it:d}, Loss_f: {loss_f.item():.3e}')

    def predict(self, x_np, y_np, L_np, cx_np, cy_np, r_np):
        self.net.eval()
        x = torch.tensor(x_np, dtype=torch.float32, device=device, requires_grad=True)
        y = torch.tensor(y_np, dtype=torch.float32, device=device, requires_grad=True)
        L = torch.tensor(L_np, dtype=torch.float32, device=device)
        cx = torch.tensor(cx_np, dtype=torch.float32, device=device)
        cy = torch.tensor(cy_np, dtype=torch.float32, device=device)
        r = torch.tensor(r_np, dtype=torch.float32, device=device)
        
        derivs = self.compute_derivatives(x, y, L, cx, cy, r)
        u = derivs['u']
        u_xx =  derivs['u_xx'];u_yy =  derivs['u_yy'];u_xy =  derivs['u_xy']
        self.net.train()
        return u.detach().cpu().numpy(),u_xx.detach().cpu().numpy(), u_yy.detach().cpu().numpy(),u_xy.detach().cpu().numpy()


    def evaluate(self, L_test, cx_test, cy_test, r_test):
        # Skip loading FEM for now; just run prediction
        print(f"\n--- Evaluating for L={L_test}, cx={cx_test}, cy={cy_test}, r={r_test} ---")
        # 1. Generate grid points for plotting
        n_grid = 201
        x_space = np.linspace(0, L_test, n_grid)
        y_space = np.linspace(0, 1, n_grid)
        x_grid, y_grid = np.meshgrid(x_space, y_space)
        
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()

        # 2. Remove interior points so they are not used for prediction or plotting
        dist_sq = (x_flat - cx_test)**2 + (y_flat - cy_test)**2
        valid_indices = dist_sq >= r_test**2
        
        x_pred = x_flat[valid_indices][:, None]
        y_pred = y_flat[valid_indices][:, None]
        
        # 3. Prepare model inputs
        num_valid_points = x_pred.shape[0]
        L_pred = np.full_like(x_pred, L_test)
        cx_pred = np.full_like(x_pred, cx_test)
        cy_pred = np.full_like(x_pred, cy_test)
        r_pred = np.full_like(x_pred, r_test)

        # 4. Use the PINN to predict displacements
        u_pred,u_xx_pred,u_yy_pred,u_xy_pred = self.predict(x_pred, y_pred, L_pred, cx_pred, cy_pred, r_pred)
        u_pred,u_xx_pred,u_yy_pred,u_xy_pred =u_pred.flatten(),u_xx_pred.flatten(),u_yy_pred.flatten(),u_xy_pred.flatten()
        # 5. Visualize u predictions
        plt.figure(figsize=(8, 5), dpi=150)
        plt.tricontourf(x_pred.flatten(), y_pred.flatten(), u_pred, levels=50, cmap='Spectral')
        cbar = plt.colorbar()
        cbar.set_label('Predicted Deflection u(x, y)')
        
        # Draw circular hole
        circle = plt.Circle((cx_test, cy_test), r_test, color='white', fill=True)
        plt.gca().add_artist(circle)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'PINN Prediction for Plate with Hole\nL={L_test}, center=({cx_test:.2f}, {cy_test:.2f}), r={r_test:.2f}')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        # 6. Visualize u_xx predictions
        plt.figure(figsize=(8, 5), dpi=150)
        plt.tricontourf(x_pred.flatten(), y_pred.flatten(), u_xx_pred, levels=50, cmap='Spectral')
        cbar = plt.colorbar()
        cbar.set_label('Predicted Deflection u_xx(x, y)')
        
        # Draw circular hole
        circle = plt.Circle((cx_test, cy_test), r_test, color='white', fill=True)
        plt.gca().add_artist(circle)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'PINN Prediction for Plate with Hole\nL={L_test}, center=({cx_test:.2f}, {cy_test:.2f}), r={r_test:.2f}')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        # 7. Visualize u_xx predictions
        plt.figure(figsize=(8, 5), dpi=150)
        plt.tricontourf(x_pred.flatten(), y_pred.flatten(), u_yy_pred, levels=50, cmap='Spectral')
        cbar = plt.colorbar()
        cbar.set_label('Predicted Deflection u_yy(x, y)')
        
        # Draw circular hole
        circle = plt.Circle((cx_test, cy_test), r_test, color='white', fill=True)
        plt.gca().add_artist(circle)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'PINN Prediction for Plate with Hole\nL={L_test}, center=({cx_test:.2f}, {cy_test:.2f}), r={r_test:.2f}')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        # 8. Visualize u_xx predictions
        plt.figure(figsize=(8, 5), dpi=300)
        plt.tricontourf(x_pred.flatten(), y_pred.flatten(), u_xy_pred, levels=50, cmap='Spectral')
        cbar = plt.colorbar()
        cbar.set_label('Predicted Deflection u_xy(x, y)')
        
        # Draw circular hole
        circle = plt.Circle((cx_test, cy_test), r_test, color='white', fill=True)
        plt.gca().add_artist(circle)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'PINN Prediction for Plate with Hole\nL={L_test}, center=({cx_test:.2f}, {cy_test:.2f}), r={r_test:.2f}')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


def relative_l2_error(y_true, y_pred):
    return np.sqrt(np.sum((y_pred - y_true) ** 2)) / np.sqrt(np.sum(y_true ** 2))

# --- Data generation for a plate with a hole ---
def make_data_Sobol_plate_with_hole(config):
    n_points_target = config["n_domain"]
    L_range = config["L_range"]
    cx_rel_range = config["cx_rel_range"] # cx range relative to L
    cy_range = config["cy_range"]
    r_range = config["r_range"]
    
    # Sobol sequence dimension d=6: (x_hat, y, L, cx_rel, cy, r)
    d = 6
    sobol_engine = qmc.Sobol(d=d, scramble=True, seed=config['seed'])
    
    lower_bounds = [0, 0, L_range[0], cx_rel_range[0], cy_range[0], r_range[0]]
    upper_bounds = [1, 1, L_range[1], cx_rel_range[1], cy_range[1], r_range[1]]

    X_valid = []
    
    # Use acceptance-rejection sampling
    n_generated = 0
    # Oversample initially to compensate for rejected points
    n_to_sample_initially = int(n_points_target * 2.0) 
    
    with tqdm(total=n_points_target, desc="Accept-Reject Sampling") as pbar:
        while len(X_valid) < n_points_target:
            # Generate another batch when the initial samples are exhausted
            if n_generated % n_to_sample_initially == 0:
                 uniform_samples = sobol_engine.random(n=n_to_sample_initially)
                 X_scaled = qmc.scale(uniform_samples, lower_bounds, upper_bounds)
            
            # Take one sample from the current batch
            idx_in_batch = n_generated % n_to_sample_initially
            sample = X_scaled[idx_in_batch]

            x_hat, y, L, cx_rel, cy, r = sample
            
            # Compute physical coordinates
            x = x_hat * L
            cx = cx_rel * L # cx is relative position
            
            # Acceptance test: is the point outside the hole?
            if (x - cx)**2 + (y - cy)**2 >= r**2:
                X_valid.append([x, y, L, cx, cy, r])
                pbar.update(1)

            n_generated += 1
            # Avoid infinite loops when the hole is too large
            if n_generated > n_points_target * 50: 
                print("\nWarning: Rejection rate is very high. Check geometry ranges.")
                break
    print(f"\nGenerated {len(X_valid)} valid data points from {n_generated} candidates.")
    return np.array(X_valid)


# --- Uniform random sampling: interior points of the perforated plate (acceptance-rejection) ---
def make_data_Random_plate_with_hole(config):
    """
    Sample candidate points uniformly at random in parameter space,
    then keep only the ones outside the circular hole via acceptance-rejection.

    Variables:
      - x_hat ∈ [0,1], actual x = x_hat * L
      - y ∈ [0,1]
      - L ∈ L_range
      - cx_rel ∈ cx_rel_range, actual cx = cx_rel * L
      - cy ∈ cy_range
      - r ∈ r_range
    """
    n_points_target = config["n_domain"]
    L_range = config["L_range"]
    cx_rel_range = config["cx_rel_range"]  # cx range relative to L
    cy_range = config["cy_range"]
    r_range = config["r_range"]

    print("Generating training data for plate with hole using Random Uniform...")

    # Random number generator
    rng = np.random.default_rng(config['seed'])

    # Parameter bounds used for single scaling step
    lower_bounds = [0, 0, L_range[0], cx_rel_range[0], cy_range[0], r_range[0]]
    upper_bounds = [1, 1, L_range[1], cx_rel_range[1], cy_range[1], r_range[1]]

    X_valid = []
    n_generated = 0
    n_to_sample_initially = int(n_points_target * 2.0)  # Batch candidates to reduce loop overhead

    with tqdm(total=n_points_target, desc="Accept-Reject Sampling (Random)") as pbar:
        while len(X_valid) < n_points_target:
            # Generate candidate parameters in batch
            if n_generated % n_to_sample_initially == 0:
                U = rng.random((n_to_sample_initially, 6))
                X_scaled = qmc.scale(U, lower_bounds, upper_bounds)

            # Take one sample from the current batch
            idx_in_batch = n_generated % n_to_sample_initially
            sample = X_scaled[idx_in_batch]

            x_hat, y, L, cx_rel, cy, r = sample
            x = x_hat * L
            cx = cx_rel * L  # cx is relative position

            # Acceptance-rejection: keep points outside the hole
            if (x - cx)**2 + (y - cy)**2 >= r**2:
                X_valid.append([x, y, L, cx, cy, r])
                pbar.update(1)

            n_generated += 1
            # Prevent abnormal parameter ranges from causing infinite loops
            if n_generated > n_points_target * 50:
                print("\nWarning: Rejection rate is very high. Check geometry ranges.")
                break

    print(f"\nGenerated {len(X_valid)} valid data points from {n_generated} candidates (Random).")
    return np.array(X_valid)


# --- Main function  ---
def main(config):
    save_dir = config["save_dir"]
    total_start = time.time()
    # Sampling method
    if config["sample_method"] =="make_data_Sobol_plate_with_hole":
        X_star = make_data_Sobol_plate_with_hole(config)
    if config["sample_method"] =="make_data_Random_plate_with_hole":
        X_star = make_data_Random_plate_with_hole(config)

    model = PINN(X_star, config)
    
    model.train(config)
    model.train_lbfgs(config)
    torch.save(model.net.state_dict(), os.path.join(save_dir, "model.pth"))
    total_time = time.time()-total_start
    print("Training time:", total_time)
    config["total_time"] = total_time

    # --- Save configuration and loss ---
    config["min_X"] = model.min_X.tolist()
    config["max_X"] = model.max_X.tolist()
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    loss_df = pd.DataFrame({"loss_f": model.loss_f_log})
    loss_df.to_csv(os.path.join(save_dir, "loss_log.csv"), index=False)
    print(f"\nModel and logs saved to: {save_dir}")
    
    # # --- Evaluate and visualize several test cases ---
    # Parameters include aspect ratio L_test, hole center (cx, cy), and radius r ? all handled by one model!
    if config.get("run_eval", True):
        model.evaluate(L_test=1, cx_test=0.5, cy_test=0.5, r_test=0.2)
        model.evaluate(L_test=2, cx_test=0.5, cy_test=0.5, r_test=0.2)
        model.evaluate(L_test=3, cx_test=0.5, cy_test=0.5, r_test=0.2)
        model.evaluate(L_test=4, cx_test=0.5, cy_test=0.5, r_test=0.2)



if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("results_hole_ndomain", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    config = {
        "save_dir": save_dir,
        "seed": 1234,  
        "n_domain": 2**17, 
        "sample_method" :"make_data_Sobol_plate_with_hole",

        # Geometry and defect parameter ranges ---
        "L_range": (0.9, 4.1),       # Plate length L range
        "cx_rel_range": (0.1, 0.9),  # Hole center x relative range
        "cy_range": (0.1, 0.9),      # Hole center y range
        "r_range": (0.01, 0.41),       # Hole radius range
        "model_type": 'GAPINN',
        "BC_b": "clamped",  # support  free  # clamped, simply supported, free
        "BC_hole": "clamped",  #  support  free
        "constraint_type": 'hard',

        #  Input layer has six features (x, y, L, cx, cy, r)
        # "layers": [6] + 4*[50] + [1], 
        "layers": [6] + 3*[30] + [1], 
        
        "epochs": 200, 
        "epochs_lbfgs": 200,
        "print_every": 10,
        "lr": 1e-3, 
    }
    print("Config:", config)
    
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    main(config=config)
    print("Config:", config)
