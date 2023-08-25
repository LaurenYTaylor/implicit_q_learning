def get_config(train_type=None):
    config = {"actor_lr": 3e-4, "value_lr": 3e-4, "critic_lr": 3e-4, "hidden_dims": (256, 256), "discount": 0.99,
              "expectile": 0.9, "temperature": 10.0, "dropout_rate": None, "tau": 0.005}

    if train_type == "online":
        config["opt_decay_schedule"] = None  # Don't decay optimizer lr

    return config
