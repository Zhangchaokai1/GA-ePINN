import datetime
import os

from gaepinn.plate.hole_fixed import main


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("outputs", "hole_fixed", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    config = {
        "save_dir": save_dir,
        "seed": 230258995,
        "n_domain": 2**12,
        "sample_method": "make_data_Sobol_plate_with_hole",
        "L_fixed": 2.5,
        "cx_rel_range": (0.1, 0.9),
        "cy_range": (0.1, 0.9),
        "r_range": (0.061, 0.361),
        "model_type": "GAPINN",
        "constraint_type": "hard",
        "hole_c": "free",
        "layers": [5, 80, 80, 1],
        "epochs": 0,
        "epochs_lbfgs": 10,
        "print_every": 1,
        "lr": 1e-3,
    }
    main(config)
