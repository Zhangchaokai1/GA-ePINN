import datetime
import os

from gaepinn.plate.hole_parametric import main


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("outputs", "hole_parametric", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    config = {
        "save_dir": save_dir,
        "seed": 1234,
        "n_domain": 2**12,
        "sample_method": "make_data_Sobol_plate_with_hole",
        "L_range": (0.9, 4.1),
        "cx_rel_range": (0.1, 0.9),
        "cy_range": (0.1, 0.9),
        "r_range": (0.01, 0.41),
        "model_type": "GAPINN",
        "BC_b": "clamped",
        "BC_hole": "clamped",
        "constraint_type": "hard",
        "layers": [6, 30, 30, 30, 1],
        "epochs": 0,
        "epochs_lbfgs": 10,
        "print_every": 1,
        "lr": 1e-3,
    }
    main(config)
