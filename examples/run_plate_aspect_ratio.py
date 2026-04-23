import datetime
import os

from gaepinn.plate.aspect_ratio import main


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("outputs", "aspect_ratio", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    config = {
        "save_dir": save_dir,
        "seed": 230258995,
        "n_domain": 2**12,
        "L_range": (0.9, 5.1),
        "constraint_type": "clamped",
        "make_data_method": "make_data_Sobol",
        "layers": [3, 20, 20, 20, 1],
        "epochs": 0,
        "epochs_lbfgs": 10,
        "print_every": 1,
        "lr": 1e-3,
    }
    main(config)
