import os
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch import optim
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
import datetime
import pandas as pd
from tqdm import tqdm
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")


def get_fem_data(L,excel_path="FEM_clamped.xlsx"):
    sheet_name = f"L_{L:.1f}".replace('.', '_')
    df = pd.read_excel(excel_path,sheet_name=sheet_name)
    u_fem = df["w"].values
    x_pred = df['x'].values[:, None] 
    y_pred = df['y'].values[:, None]  
    L_pred = df['L'].values[:, None] 
    return u_fem,x_pred.flatten(),y_pred.flatten(),L_pred.flatten()



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
    
class GAPINN:
    def __init__(self, X, config):
        self.constraint_type = config['constraint_type']
        self.shell_case = config["shell_case"]
        self.save_dir = config.get("save_dir", "outputs")
        self.fem_excel_path = config.get("fem_excel_path", "FEM_clamped.xlsx")
        
        self.max_X = X.max(axis=0)
        self.min_X = X.min(axis=0)
        # The dimension of the network input layer is 3.   B=1 , so the aspect is L/1 = L
        self.x = torch.tensor(X[:, 0:1], dtype=torch.float32, device=device)
        self.y = torch.tensor(X[:, 1:2], dtype=torch.float32, device=device)
        self.L = torch.tensor(X[:, 2:3], dtype=torch.float32, device=device)

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
        # LBFGS 
        self.lbfgs = optim.LBFGS(params,
                                 lr=1.0,
                                 max_iter=50, 
                                 max_eval=80, 
                                 history_size=100,
                                 line_search_fn='strong_wolfe')
        self.loss_c_log = []
        self.loss_f_log = []

    def normalize(self, inp):
        # X:(x, y, L)
        min_X = torch.tensor(self.min_X, dtype=torch.float32, device=device)
        max_X = torch.tensor(self.max_X, dtype=torch.float32, device=device)
        return 2.0 * (inp - min_X) / (max_X - min_X + 1e-12) - 1.0

    def net_forward(self, x, y, L):
        xyl = torch.cat([x, y, L], dim=1)
        xyl_n = self.normalize(xyl)
        # --- Hard constraint functions depend on L to adapt to changing domains ---
        # Hard constrain on boundary: u=0, du/dn=0 on x=0, x=L, y=0, y=1
        if self.constraint_type=="clamped":
            boundary_func = (x * (L - x) * y * (1.0 - y)) ** 2
        if self.constraint_type=="supported":
             boundary_func = (x * (L - x) * y * (1.0 - y)) 
        u = boundary_func * self.net(xyl_n)
        return u

    def compute_derivatives(self, x, y, L):
        x_clone = x.clone().detach().requires_grad_(True)
        y_clone = y.clone().detach().requires_grad_(True)
        L_clone = L.clone().detach()
        
        u = self.net_forward(x_clone, y_clone, L_clone)
        
        ones_u = torch.ones_like(u)

        # first derivative   (for x, y)
        u_x, u_y = grad(u, (x_clone, y_clone), grad_outputs=ones_u, create_graph=True)
        
        # second derivative 
        u_xx = grad(u_x, x_clone, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y_clone, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_xy = grad(u_x, y_clone, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        derivatives = {'u': u, 'u_x': u_x, 'u_y': u_y, 'u_xx': u_xx, 'u_yy': u_yy, 'u_xy': u_xy}
        return derivatives

    def loss_terms(self):
        # ---  loss_f (total potential energy) ---
        derivs_f = self.compute_derivatives(self.x, self.y, self.L)
        
        u, u_xx, u_yy, u_xy = derivs_f['u'], derivs_f['u_xx'], derivs_f['u_yy'], derivs_f['u_xy']
        k = torch.cat([-u_xx, -u_yy, -2.0 * u_xy], dim=1)
        M = torch.matmul(k, self.D)
        
        # strain energe density: U_density 
        strain_energy_density = 0.5 * torch.sum(M * k, dim=1, keepdim=True)
        # external work density: W_density
        external_work_density = self.q * u
        
        # total potential energy density: Π_density = U_density - W_density
        potential_energy_density = strain_energy_density - external_work_density
        total_potential_energy = potential_energy_density / self.L 
        loss_f = total_potential_energy.mean()

        return loss_f

    def train(self, config):
        epochs = config["epochs"]
        print_every = config["print_every"]
        start_time = time.time()

        for it in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            loss_f = self.loss_terms()
            loss =  loss_f
            loss.backward()
            self.optimizer.step()
            self.loss_f_log.append(loss_f.item())

            if it % print_every == 0 or it == epochs - 1:
                elapsed = time.time() - start_time
                print(f'It: {it:d}, Loss_f: {loss_f.item():.3e}, Time: {elapsed:.2f}')
                start_time = time.time()

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

    def predict(self, x_np, y_np, L_np):
        self.net.eval()
        x = torch.tensor(x_np, dtype=torch.float32, device=device, requires_grad=True)
        y = torch.tensor(y_np, dtype=torch.float32, device=device, requires_grad=True)
        L = torch.tensor(L_np, dtype=torch.float32, device=device) 
        
        derivs = self.compute_derivatives(x, y, L)
        u = derivs['u']
        u_xx, u_yy, u_xy = derivs['u_xx'], derivs['u_yy'], derivs['u_xy']
        
        k = torch.cat([-u_xx, -u_yy, -2.0 * u_xy], dim=1)
        M = torch.matmul(k, self.D)
        
        self.net.train()
        return u.detach().cpu().numpy(), k.detach().cpu().numpy(), M.detach().cpu().numpy()
    
    # --- evaluation ---
    def evaluate(self, L_test):
        if not Path(self.fem_excel_path).exists():
            print(f"Skip FEM comparison because '{self.fem_excel_path}' was not found.")
            return

        sheet_name = "L_" + str(L_test).replace(".", "_")
        data_fem = pd.read_excel(self.fem_excel_path, sheet_name=sheet_name)

        x_pred = data_fem['x'].values[:, None]   # (N, 1)
        y_pred = data_fem['y'].values[:, None]   # (N, 1)
        L_pred = data_fem['L'].values[:, None]   # (N, 1)
        u_fem = data_fem['w'].values             # (N, )

        # 2. Predict w M
        u_pred, k_pred, M_pred = self.predict(x_pred, y_pred, L_pred)
        u_pred = u_pred.flatten()  # (N, )

        # 3. visualization 
        fig_1 = plt.figure(1, figsize=(20, 5), dpi=300)
        # GAPINN
        plt.subplot(1, 3, 1)
        contour1 = plt.tricontourf(x_pred.flatten(), y_pred.flatten(), u_pred.flatten(), levels=30, cmap=plt.get_cmap('Spectral'))
        cbar1 = plt.colorbar(contour1, format='%.2f')
        cbar1.set_label(label='u(x, y)', size=12)
        plt.xlabel('x', size=12)
        plt.ylabel('y', size=12)
        plt.axis('equal')
        plt.title('PINN', size=14, weight='bold')

        # FEM
        plt.subplot(1, 3, 2)
        contour2 = plt.tricontourf(x_pred.flatten(), y_pred.flatten(), u_fem.flatten(), levels=30, cmap=plt.get_cmap('Spectral'))
        cbar2 = plt.colorbar(contour2, format='%.2f')
        cbar2.set_label(label='u(x, y)', size=12)
        plt.xlabel('x', size=12)
        plt.ylabel('y', size=12)
        plt.axis('equal')
        plt.title('FEM', size=14, weight='bold')

        error = u_pred - u_fem
        MAE = np.mean(np.abs(error))  
        RMSE = np.sqrt(np.mean(error**2))  
        MAPE = (np.mean(np.abs(error)) /np.mean(np.abs(u_fem)))* 100 
        R_L2_error = relative_l2_error(u_pred,u_fem) * 100
        R2 = 1 - np.sum(error**2) / np.sum((u_fem - np.mean(u_fem))**2) 

        # Error
        plt.subplot(1, 3, 3)
        contour3 = plt.tricontourf(x_pred.flatten(), y_pred.flatten(), error.flatten(), levels=30, cmap=plt.get_cmap('RdBu_r'), vmin=-np.max(np.abs(error)), vmax=np.max(np.abs(error)))
        cbar3 = plt.colorbar(contour3, format='%.1e')
        cbar3.set_label(label='Error', size=12)
        plt.xlabel('x', size=12)
        plt.ylabel('y', size=12)
        plt.axis('equal')

        error_text = f'MAE: {MAE:.2e}\nRMSE: {RMSE:.2e}\nR_L2_error: {R_L2_error:.2f}%\nR²: {R2:.8f}'
        plt.text(0.05, 0.95, error_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                fontsize=10)

        plt.title('Error Analysis', size=16, weight='bold')
        # plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"PINN-FEM-L{L_test}.png"), bbox_inches='tight', pad_inches=0.1)
        plt.show()
        # return R_L2_error 

def relative_l2_error(y_true, y_pred):
    numerator = np.sqrt(np.sum((y_pred - y_true) ** 2))
    denominator = np.sqrt(np.sum(y_true ** 2))
    return numerator / denominator


from scipy.stats import qmc 
def make_data_Sobol(config):
    """
    generate collotation points at [0,L]x[0,1]x[L_min,L_max] by sobol
    """
    n_points = config["n_domain"]
    L_min, L_max = config["L_range"]
    lower_bounds = [0, 0, L_min]
    upper_bounds = [1, 1, L_max]
    
    # 2. Create a 3D Sobol Sequence Generator
    sobol_engine = qmc.Sobol(d=3, scramble=True, seed=config['seed'])
    
    # 3. Sample n_points from the unit cube of [0, 1]^3
    uniform_samples = sobol_engine.random(n=n_points)
    
    # 4. Linear mapping of the points of the unit cube to the target space using qmc.scale.
    X_transformed = qmc.scale(uniform_samples, lower_bounds, upper_bounds)
    
    # 5. Separating x_hat, y, L from the transformed coordinates
    x_hat = X_transformed[:, 0:1]
    y_vals = X_transformed[:, 1:2]
    L_vals = X_transformed[:, 2:3]
    
    # 6.Calculate the true physical coordinate x from x_hat and L
    x_vals = x_hat * L_vals
    
    # 7. concat
    X_domain = np.hstack((x_vals, y_vals, L_vals))
    return X_domain


def make_data_random(config):

    n_points = config["n_domain"]
    L_min, L_max = config["L_range"]
    L_vals = np.random.uniform(L_min, L_max, size=(n_points, 1))
    x_vals = np.random.rand(n_points, 1) * L_vals
    y_vals = np.random.rand(n_points, 1)
    X_domain = np.hstack((x_vals, y_vals, L_vals))
    return X_domain


def main(config):
    save_dir = config["save_dir"]
    total_start = time.time()
    # prapare data
    data_start = time.time()
    if config["make_data_method"] =="make_data_Sobol":
        X_star = make_data_Sobol(config)
    elif config["make_data_method"] =="make_data_random":
        X_star = make_data_random(config)
    else:
        pass  
    
    model = GAPINN(X_star, config)
    data_time = time.time() - data_start
    
    # train
    train_start = time.time()
    model.train(config)
    model.train_lbfgs(config)
     # --- save model.pth ---
    torch.save(model.net.state_dict(), os.path.join(save_dir, "model.pth"))

    train_time = time.time() - train_start
    total_time = time.time() - total_start
    
    print(f"time of preparing data: {data_time:.2f} s")
    print(f"time of training: {train_time:.2f} s")
    print(f"total time: {total_time:.2f} s")
    
    config["train_time"] = train_time
    config["data_time"] = data_time
    config["total_time"] = total_time
    config["min_X"] = model.min_X.tolist()
    config["max_X"] = model.max_X.tolist()

    config_save_path = os.path.join(save_dir, "config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)

    
    # save log
    loss_df = pd.DataFrame({"loss_f": model.loss_f_log})
    loss_df.to_csv(os.path.join(save_dir, "loss_log.csv"), index=False)
    if config.get("run_eval", True):
        for L in config.get("eval_L_values", [1.0, 2.5, 4.5]):
            model.evaluate(L)

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(r"results", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    config = {
        "save_dir": save_dir,
        "seed": 230258995,  
        "n_domain": 2**16, 
        "L_range":(0.9,5.1),

        "constraint_type": 'clamped', # supported
        "make_data_method":'make_data_Sobol',  # make_data_Sobol
        "layers": [3] + 3 * [20] + [1],   # ---  (x, y, L) ---
        
        "epochs": int(0), 
        "epochs_lbfgs": int(100),
        "print_every": 1,
        "lr": 1e-3, 
    }
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    main(config=config)
