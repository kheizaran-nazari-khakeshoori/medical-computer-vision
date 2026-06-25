"""Assembling training pipeline with wandb logging."""
try:
    import wandb
    def init_wandb(project="radiology"):
        wandb.init(project=project)
    def log_metrics(metrics: dict):
        wandb.log(metrics)
except ImportError:
    def init_wandb(*a, **k): pass
    def log_metrics(*a, **k): pass
