# Adapted from CProMG GitHub repository

class EarlyStopping:
    """
    Early stops the training if metrics do not improve after a given patience.
    
    Args:
        mode (str): Mode of metric optimisation. Default: 'min'
        patience (int): How long to wait after last time metric improves
        delta (float): Minimum change in the monitored quantity to qualify as an improvement 
    """
    def __init__(
        self, 
        mode:str = 'min', 
        patience:int = 10, 
        delta:float = 0.0,
    ):
        super().__init__()
        assert mode in ['min', 'max'], "Accepted modes are 'min' or 'max'"
        self.mode = mode
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
        
    def __call__(self, metric):
        improved = (self.mode == 'min' and metric < self.best_score - self.delta) or \
                   (self.mode == 'max' and metric > self.best_score + self.delta)
        
        if improved:
            self.best_score = metric
            self.counter = 0
            update = True

        else:
            self.counter += 1
            update = False
            if self.counter >= self.patience:
                self.early_stop = True
        
        return update, self.best_score, self.counter
