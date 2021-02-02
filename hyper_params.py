hyper_params = {
    "nb_time" : 200,
    "n_window" : 5, # 5 windows by 300 frame time
    "d_m":128,
    "d_ff":512,
    # "n_mels": 40,
    "numcep": 13,
    "batch_size": 64,
    "num_epochs": 100,
    "lr": 0.001,
    "weight_decay": 0.001,
    "num_workers" : 12,
    "comet": False,
    "log": False,
    "dev": True,
}
