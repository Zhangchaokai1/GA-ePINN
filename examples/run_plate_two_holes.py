import datetime
import os

from gaepinn.plate.two_holes import main


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("outputs", "two_holes", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    config = {
        "save_dir": save_dir,
        "seed": 1234,
        "n_domain": 2**12,
        "sample_method": "make_data_Sobol_plate_with_two_holes",
        "L_fixed": 2.5,
        "cx1_rel_range": (0.10, 0.90),
        "cy1_range": (0.10, 0.90),
        "r1_range": (0.01, 0.25),
        "cx2_rel_range": (0.10, 0.90),
        "cy2_range": (0.10, 0.90),
        "r2_range": (0.01, 0.25),
        "model_type": "GAPINN",
        "constraint_type": "hard",
        "hole_c": "clamped",
        "layers": [8, 50, 50, 50, 50, 1],
        "epochs": 0,
        "epochs_lbfgs": 10,
        "print_every": 1,
        "lr": 1e-3,
    }
    main(config)
