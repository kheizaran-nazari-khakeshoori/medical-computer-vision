"""Configuring hyperparameter search with optuna."""
try:
    import optuna
    def objective(trial, train_fn):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        bs = trial.suggest_categorical("batch_size", [8,16,32])
        return train_fn(lr=lr, batch_size=bs)
    def run_search(train_fn, n_trials=10):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, train_fn), n_trials=n_trials)
        return study.best_params
except ImportError:
    def run_search(*args, **kwargs): print("optuna not installed"); return {}
