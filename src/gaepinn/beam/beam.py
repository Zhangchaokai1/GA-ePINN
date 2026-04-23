import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
from torch.autograd import grad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import json
from tqdm import tqdm
import datetime

# Check whether a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Neural network definition
class FeedForward(nn.Module):
    def __init__(self, layers):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            # Xavier initialization
            nn.init.xavier_normal_(modules[-1].weight, gain=1.0)
            nn.init.zeros_(modules[-1].bias)
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

# 2. ml-PINN 
class mlPINN():
    def __init__(self, X_domain, X_boundary, config):
        self.config = config
        self.beam_case = config['beam_case']
        self.constraint_type = 'soft'

        # Physical constants
        self.EI = config['EI']
        self.q_val = config['q']

        # Normalization parameters
        self.max_X, self.min_X = X_domain.max(0), X_domain.min(0)

        # Data points
        self.x = torch.tensor(X_domain, dtype=torch.float32, device=device)
        self.q = torch.full_like(self.x, self.q_val)
        self.x_b = torch.tensor(X_boundary, dtype=torch.float32, device=device)

        # --- Initialize four separate neural networks ---
        #  [1] -> 3x[10] -> [1]
        layers_sub = [1] + 3 * [10] + [1]
        self.net_w = FeedForward(layers_sub).to(device)     # for deflection w(x)
        self.net_theta = FeedForward(layers_sub).to(device) # for angle theta(x)
        self.net_k = FeedForward(layers_sub).to(device)     # for curvature k(x)
        self.net_Q = FeedForward(layers_sub).to(device)     # for shear force Q(x)
        
        # The optimizer must include parameters from all four networks
        params = (list(self.net_w.parameters()) + 
                  list(self.net_theta.parameters()) + 
                  list(self.net_k.parameters()) + 
                  list(self.net_Q.parameters()))
        self.optimizer = Adam(params, lr=config['lr'])
        self.lbfgs = optim.LBFGS(params,
                                 lr=1.0,
                                 max_iter=50,
                                 max_eval=80,
                                 history_size=100,
                                 line_search_fn='strong_wolfe')

        self.loss_c_log = []
        self.loss_f_log = []

    def normalize(self, inp):
        return 2.0 * (inp - self.min_X[0]) / (self.max_X[0] - self.min_X[0] + 1e-12) - 1.0

    def _get_phys_vars(self, x):
        """辅助函数：获取所有网络在给定点x处的输出"""
        xn = self.normalize(x)
        w = self.net_w(xn)
        theta = self.net_theta(xn)
        k = self.net_k(xn)
        Q = self.net_Q(xn)
        M = -self.EI * k  # 本构关系 M = -EI*k (k=-w'')
        return w, theta, k, M, Q

    def loss_terms(self):
        """计算边界损失和物理损失"""
        # --- 边界损失 loss_c (软约束) ---
        w_b, theta_b, k_b, M_b, _ = self._get_phys_vars(self.x_b)
        w_b0, w_b1 = w_b[0], w_b[1]
        theta_b0, theta_b1 = theta_b[0], theta_b[1]
        M_b0, M_b1 = M_b[0], M_b[1]

        if self.beam_case == 1: # 简支梁: w=0, M=0
            loss_c = (w_b0.pow(2) + w_b1.pow(2) + M_b0.pow(2) + M_b1.pow(2))
        elif self.beam_case == 2: # 固端梁: w=0, θ=0
            loss_c = (w_b0.pow(2) + w_b1.pow(2) + theta_b0.pow(2) + theta_b1.pow(2))
        elif self.beam_case == 3: # w(0)=0, θ(0)=0, w(1)=1, θ(1)=0
            loss_c = (w_b0.pow(2) + theta_b0.pow(2) + (w_b1 - 1.0).pow(2) + theta_b1.pow(2))
        elif self.beam_case == 4: # w(0)=0, θ(0)=0, w(1)=0, θ(1)=1
            loss_c = (w_b0.pow(2) + theta_b0.pow(2) + w_b1.pow(2) + (theta_b1 - 1.0).pow(2))
        
        # --- 物理损失 loss_f ---
        x_f = self.x.clone().detach().requires_grad_(True)
        w, theta, k, M, Q = self._get_phys_vars(x_f)
        
        # 计算所需的一阶导数
        w_x = grad(w.sum(), x_f, create_graph=True)[0]
        theta_x = grad(theta.sum(), x_f, create_graph=True)[0]
        M_x = grad(M.sum(), x_f, create_graph=True)[0]
        Q_x = grad(Q.sum(), x_f, create_graph=True)[0]

        # 基于一阶PDE系统计算残差
        f_geom_1 = w_x - theta      # Geometric: dw/dx = θ  # 原来tf的是：f_a_k = a_x + k    
        f_geom_2 = theta_x - k      # Geometric: dθ/dx = k
        f_equil_1 = M_x - Q         # Equilibrium: dM/dx = Q
        f_equil_2 = Q_x + self.q    # Equilibrium: dQ/dx = -q
        
        loss_f = (f_geom_1.pow(2).mean() + 
                  f_geom_2.pow(2).mean() +
                  f_equil_1.pow(2).mean() +
                  f_equil_2.pow(2).mean())
        
        return loss_c, loss_f

    def train(self):
        """训练循环"""
        epochs = self.config["epochs"]
        print_every = self.config["print_every"]
        start_time = time.time()

        for it in tqdm(range(epochs), desc="Training Progress"):
            self.optimizer.zero_grad()
            loss_c, loss_f = self.loss_terms()
            loss = loss_c + 1.0 * loss_f 
            loss.backward()
            self.optimizer.step()
            
            self.loss_c_log.append(loss_c.item())
            self.loss_f_log.append(loss_f.item())

            if it % print_every == 0 or it == epochs - 1:
                elapsed = time.time() - start_time
                print(f'It: {it:d}, Loss_c: {loss_c.item():.3e}, Loss_f: {loss_f.item():.3e}, Time: {elapsed:.2f}s')
                start_time = time.time()
    def train_lbfgs(self):
        config = self.config
        epochs = config["epochs_lbfgs"]
        print_every = config["print_every"]
        start_time = time.time()

        def closure():
            self.lbfgs.zero_grad()
            loss_c, loss_f = self.loss_terms()
            loss = loss_c + loss_f
            loss.backward()
            return loss

        for it in tqdm(range(epochs)):
            self.lbfgs.step(closure)
            loss_c, loss_f = self.loss_terms()
            self.loss_c_log.append(loss_c.item())
            self.loss_f_log.append(loss_f.item())

            if it % print_every == 0 or it == epochs - 1:
                elapsed = time.time() - start_time
                print(f'It(LBFGS): {it:d}, Loss_c: {loss_c.item():.3e}, Loss_f: {loss_f.item():.3e}, Time: {elapsed:.2f}')
                start_time = time.time()

    def predict_w(self, x_np):
        """只预测挠度w，用于绘图"""
        x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(1)
        xn = self.normalize(x)
        self.net_w.eval()
        with torch.no_grad():
            w_pred = self.net_w(xn)
        self.net_w.train()
        return w_pred.cpu().numpy()

# --- 3 BeamPINN求解器 ---
class BeamPINN:
    """
    使用 cPINN (强形式) 或 ePINN (能量法) 求解一维梁弯曲问题
    """
    def __init__(self, X_domain, X_boundary, config):
        self.config = config
        self.model_type = config['model_type']
        self.constraint_type = config['constraint_type']
        self.beam_case = config['beam_case']

        # 物理常量
        self.EI = config['EI']
        self.q_val = config['q']

        # 归一化参数
        self.max_X = X_domain.max(axis=0)
        self.min_X = X_domain.min(axis=0)

        # 配位点 (内部/全局)
        self.x = torch.tensor(X_domain, dtype=torch.float32, device=device)
        self.q = torch.full_like(self.x, self.q_val) # 荷载

        # 边界点
        self.x_b = torch.tensor(X_boundary, dtype=torch.float32, device=device)

        # 网络和优化器
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

    def normalize(self, inp):
        """将输入归一化到[-1, 1]"""
        return 2.0 * (inp - self.min_X[0]) / (self.max_X[0] - self.min_X[0] + 1e-12) - 1.0

    def net_forward(self, x):
        """
        试函数(Trial Function)的前向传播，硬约束实现
        输入: x (tensor) - 空间坐标
        输出: w (tensor) - 挠度/位移
        """
        xn = self.normalize(x)
        
        if self.constraint_type == 'hard':
            # --- 硬约束实现 ---
            # 基础修正项 G(x) = x * (1-x)，在x=0和x=1处为0
            # G(x)^2 则保证在边界处函数值和一阶导数均为0
            G_x = x * (1.0 - x)
            
            if self.beam_case == 1:  # 简支梁: w(0)=0, w(1)=0
                return G_x * self.net(xn)
            
            if self.beam_case == 2:  # 固端梁: w(0)=w(1)=0, w'(0)=w'(1)=0
                return G_x**2 * self.net(xn)
            
            if self.beam_case == 3:  # 一端固支一端指定位移: w(0)=w'(0)=0, w(1)=1, w'(1)=0
                # u_trial 是一个满足所有4个边界条件的已知函数（三次埃尔米特多项式）
                u_trial = x**2 * (3.0 - 2.0*x) 
                return u_trial + G_x**2 * self.net(xn)

            if self.beam_case == 4:  # 一端固支一端指定转角: w(0)=w'(0)=0, w(1)=0, w'(1)=1
                # u_trial 是满足这4个边界条件的已知函数
                u_trial = x**2 * (x - 1.0)
                return u_trial + G_x**2 * self.net(xn)

        else:  # 'soft' 软约束: 网络直接输出位移
            return self.net(xn)

    def compute_derivatives(self, x, order=4):
        """按需计算导数，cPINN需要到4阶"""
        x_clone = x.clone().detach().requires_grad_(True)
        w = self.net_forward(x_clone)
        
        derivatives = {'w': w}
        
        # 逐阶计算导数
        w_prev = w
        for i in range(1, order + 1):
            w_curr = grad(w_prev.sum(), x_clone, create_graph=True)[0]
            derivatives[f'w_{"x"*i}'] = w_curr
            w_prev = w_curr
            
        return derivatives

    def loss_terms(self):
        """计算边界损失和物理损失"""
        # --- 边界损失 loss_c ---
        if self.constraint_type == 'hard':
            # 硬约束下，边界条件被精确满足，损失为0
            loss_c = torch.tensor(0.0, device=device)
        else: # 'soft'
            # 软约束下，显式计算边界误差
            derivs_b = self.compute_derivatives(self.x_b, order=2)
            w_b, w_x_b, w_xx_b = derivs_b['w'], derivs_b['w_x'], derivs_b['w_xx']
            
            # 边界点 x=0 和 x=1
            w_b0, w_b1 = w_b[0], w_b[1]
            w_x_b0, w_x_b1 = w_x_b[0], w_x_b[1]
            w_xx_b0, w_xx_b1 = w_xx_b[0], w_xx_b[1]
            
            if self.beam_case == 1: # 简支梁: w=0, M=EIw''=0
                loss_w = w_b0.pow(2) + w_b1.pow(2)
                loss_M = w_xx_b0.pow(2) + w_xx_b1.pow(2)
                loss_c = loss_w + loss_M
            
            elif self.beam_case == 2: # 固端梁: w=0, w'=0
                loss_w = w_b0.pow(2) + w_b1.pow(2)
                loss_theta = w_x_b0.pow(2) + w_x_b1.pow(2)
                loss_c = loss_w + loss_theta
                
            elif self.beam_case == 3: # w(0)=0, w'(0)=0, w(1)=1, w'(1)=0
                loss_b0 = w_b0.pow(2) + w_x_b0.pow(2)
                loss_b1 = (w_b1 - 1.0).pow(2) + w_x_b1.pow(2)
                loss_c = loss_b0 + loss_b1
            
            elif self.beam_case == 4: # w(0)=0, w'(0)=0, w(1)=0, w'(1)=1
                loss_b0 = w_b0.pow(2) + w_x_b0.pow(2)
                loss_b1 = w_b1.pow(2) + (w_x_b1 - 1.0).pow(2)
                loss_c = loss_b0 + loss_b1

        # --- 物理损失 loss_f ---
        # cPINN 需要4阶导数，ePINN 只需要2阶
        required_order = 4 if self.model_type == 'cPINN' else 2
        derivs_f = self.compute_derivatives(self.x, order=required_order)
        w = derivs_f['w']

        if self.model_type == 'cPINN':
            # 基于强形式PDE: EI * w'''' - q = 0
            w_xxxx = derivs_f['w_xxxx']
            f_residual = self.EI * w_xxxx - self.q
            loss_f = f_residual.pow(2).mean()
            
        elif self.model_type == 'ePINN':
            # 基于最小势能原理: Π = ∫(0.5*EI*(w'')² - q*w)dx
            w_xx = derivs_f['w_xx']
            strain_energy_density = 0.5 * self.EI * w_xx.pow(2)
            external_work_density = self.q * w
            potential_energy_density = strain_energy_density - external_work_density
            loss_f = potential_energy_density.mean() # 蒙特卡洛积分

        return loss_c, loss_f

    def train(self):
        """训练循环"""
        epochs = self.config["epochs"]
        print_every = self.config["print_every"]
        start_time = time.time()

        for it in tqdm(range(epochs), desc="Training Progress"):
            self.optimizer.zero_grad()
            loss_c, loss_f = self.loss_terms()
            loss = loss_c + loss_f
            loss.backward()
            self.optimizer.step()
            
            self.loss_c_log.append(loss_c.item())
            self.loss_f_log.append(loss_f.item())

            if it % print_every == 0 or it == epochs - 1:
                elapsed = time.time() - start_time
                print(f'It: {it:d}, Loss_c: {loss_c.item():.3e}, Loss_f: {loss_f.item():.3e}, Time: {elapsed:.2f}s')
                start_time = time.time()

    def train_lbfgs(self):
        config = self.config
        epochs = config["epochs_lbfgs"]
        print_every = config["print_every"]
        start_time = time.time()

        def closure():
            self.lbfgs.zero_grad()
            loss_c, loss_f = self.loss_terms()
            loss = loss_c + loss_f
            loss.backward()
            return loss

        for it in tqdm(range(epochs)):
            self.lbfgs.step(closure)
            loss_c, loss_f = self.loss_terms()
            self.loss_c_log.append(loss_c.item())
            self.loss_f_log.append(loss_f.item())

            if it % print_every == 0 or it == epochs - 1:
                elapsed = time.time() - start_time
                print(f'It(LBFGS): {it:d}, Loss_c: {loss_c.item():.3e}, Loss_f: {loss_f.item():.3e}, Time: {elapsed:.2f}')
                start_time = time.time()

    def predict(self, x_np):
        """预测给定点上的位移"""
        self.net.eval()
        x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(1)
        with torch.no_grad():
            w_pred = self.net_forward(x)
        self.net.train()
        return w_pred.cpu().numpy()

# --- 4. 辅助函数和主执行 ---
def get_analytical_solution(config, x):
    case = config['beam_case']
    EI = config['EI']
    q_abs = config['q']
    L = 1.0
    if case == 1:
        return (q_abs / (24 * EI)) * (x**4 - 2 * L * x**3 + L**3 * x)
    elif case == 2:
        return (q_abs / (24 * EI)) * (x**4 - 2 * L * x**3 + L**2 * x**2)
    elif case == 3:
        return x**2 * (3.0 - 2.0 * x)
    elif case == 4:
        return x**2 * (x - 1.0)
    else:
        return np.zeros_like(x)
    
def plot_results(model, config):
    save_dir = config["save_dir"]
    x_fine = np.linspace(0, 1, 501)

    # 预测
    if config['model_type'] == 'mlPINN':
        w_pred = model.predict_w(x_fine)
    else:
        w_pred = model.predict(x_fine)
    # # 把结果存一下
    # df_w_pred = pd.DataFrame({"w_pred":w_pred.flatten()})
    # df_w_pred.to_csv(os.path.join(save_dir,"w_pred.csv"),index=False)

    # 设置全局字体和样式
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",   # Computer Modern 字体，类似 LaTeX
        "font.size": 16,
        "axes.linewidth": 1.2
    })

    # 绘图
    plt.figure(figsize=(6, 2.5))
    plt.plot(x_fine, w_pred, label=fr'PINN ({config["model_type"]})',
             color='red', linewidth=2.2)

    w_analytical = get_analytical_solution(config, x_fine)
    if np.any(w_analytical) or np.any(w_pred):
        plt.plot(x_fine, w_analytical, '--',
                 label='Analytical Solution',
                 color='blue', linewidth=1.8)
        error_l2 = np.linalg.norm(w_pred.flatten() - w_analytical.flatten()) / \
                   (np.linalg.norm(w_analytical.flatten()) + 1e-8)
        plt.title(fr'Case {config["beam_case"]} ({config["model_type"]})'
                  f' - L2 Rel. Error: {error_l2:.2e}',
                  fontsize=18, pad=12)
        print(f'L2 Rel. Error: {error_l2:.2e}')
    else:
        plt.title(fr'Case {config["beam_case"]} ({config["model_type"]})',
                  fontsize=18, pad=12)
        print(f'L2 Rel. Error: {error_l2:.2e}')

    # 美化坐标轴
    plt.xlabel(r"$x$", fontsize=20)
    plt.ylabel(r"$w$", fontsize=20)
    plt.tick_params(direction="in", length=5, width=1.2, top=True, right=True)

    plt.legend(frameon=False, fontsize=14)
    plt.tight_layout()

    # 保存 & 显示
    plt.savefig(os.path.join(save_dir, "deflection_plot.png"), dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()


def main(config):
    # --- 创建保存目录 ---
    save_dir = config.get("save_dir")
    if not save_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        case_name = f"case{config['beam_case']}_{config['model_type']}"
        if config['model_type'] != 'mlPINN':
            case_name += f"_{config['constraint_type']}"
        save_dir = os.path.join("outputs", "beam", f"{timestamp}_{case_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    print(json.dumps(config, indent=2))
    print(f"结果保存在: {save_dir}")

    # --- 数据准备 ---
    X_domain = np.linspace(0, 1, config['n_domain'])[:, None]
    X_boundary = np.array([0.0, 1.0])[:, None]

    # --- 模型创建与训练 ---
    if config['model_type'] == 'mlPINN':
        model = mlPINN(X_domain, X_boundary, config)
    else:
        model = BeamPINN(X_domain, X_boundary, config)
        
    start_time = time.time()
    if config["epochs"]!=0:
        model.train()
    if config['epochs_lbfgs']!=0:
        model.train_lbfgs()
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    config["training_time"] = total_time
    config["save_dir"] = save_dir

    # --- 保存配置和结果 ---
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
        
    loss_df = pd.DataFrame({"loss_f": model.loss_f_log, "loss_c": model.loss_c_log})
    loss_df.to_csv(os.path.join(save_dir, "loss_log.csv"), index=False)
    
    # --- 结果可视化 ---
    plot_results(model, config)


if __name__ == "__main__":
    config = {
        # --- Problem Definition ---
        "beam_case": 1,           # 1:简支梁, 2:固端梁, 3:一端位移为1, 4:一端转角为1
        "EI": 1.0,                # 抗弯刚度
        
        # --- PINN Model Configuration ---
        "model_type": 'ePINN',     # 'cPINN', 'ePINN', 或 'mlPINN'
        "constraint_type": 'hard',  # 'soft' 或 'hard' (mlPINN会强制使用soft)
        
        # --- Network and Training Parameters ---
        # 'layers' 仅用于 cPINN/ePINN。mlPINN的网络结构是硬编码的。
        "layers": [1] + 3 * [20] + [1],
        "epochs": 100,
        "lr": 1e-3,
        "n_domain": 100,
        "epochs": int(50), 
        "epochs_lbfgs": int(50),
        
        # --- Housekeeping ---
        "print_every": 10,
        "seed": 1234
    }
    
    # 根据工况自动设置荷载 q
    if config["beam_case"] in [1, 2]:
        config["q"] = 1.0
    else:
        config["q"] = 0.0

    # 设置随机种子
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
        
    main(config=config)
