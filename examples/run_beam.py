from gaepinn.beam import run_beam


if __name__ == "__main__":
    run_beam(
        {
            "beam_case": 1,
            "EI": 1.0,
            "model_type": "ePINN",
            "constraint_type": "hard",
            "layers": [1, 20, 20, 20, 1],
            "n_domain": 100,
            "epochs": 0,
            "epochs_lbfgs": 10,
            "print_every": 1,
            "lr": 1e-3,
            "seed": 1234,
            "q": 1.0,
        }
    )
