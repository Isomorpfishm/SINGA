import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log

from typing import Any, Optional, Callable, Tuple, Dict, Sequence, NamedTuple
from tqdm import tqdm

from common import EquivariantGNN
from vanillaHelper import *
from vanillaModel import GAT, Net, Discriminator
from vanillaGenerate import dataset, data_module, QM9DataModule



EPOCHS = 25
LR = 2e-4
WEIGHT_DECAY = 1e-8
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_CHANNELS = 64
NUM_LAYERS = 5

init_wandb(name='vanilla', 
           batch_size=BATCH_SIZE, 
           lr=LR,
           epochs=EPOCHS, 
           hidden_channels=1,
           num_layers=1, 
           device=DEVICE)

"""
for i in tqdm(range(len(graphs))):
    graphs[i].to(DEVICE) 

train_dataset = dataset[len(dataset) // 10:]
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

test_dataset = dataset[:len(dataset) // 10]
test_loader = DataLoader(test_dataset, BATCH_SIZE)

#model = GAT(dataset=train_loader).to(DEVICE)
model = Net(in_channels=dataset.num_features, 
            hidden_channels=HIDDEN_CHANNELS, 
            out_channels=dataset.num_classes,
            num_layers=NUM_LAYERS
            ).to(DEVICE)
"""
 
model = EquivariantGNN(hidden_channels=HIDDEN_CHANNELS, num_mp_layers=NUM_LAYERS)

#model = Discriminator(in_channels=dataset.num_features, 
#                      hidden_channels=HIDDEN_CHANNELS, 
#                      out_channels=dataset.num_classes,
#                      num_layers=NUM_LAYERS
#                      ).to(DEVICE)            

optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.BCELoss()


def train():
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    total_correct = 0
    
    for data in loader:
        data = data.to(DEVICE)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    
    return total_correct / len(loader.dataset)

"""
for epoch in range(1, EPOCHS+1):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
"""

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable[[Tensor, Tensor], Tensor],
    pbar: Optional[Any] = None,
    optim: Optional[torch.optim.Optimizer] = None,
):
    """Run a single epoch.

    Parameters
    ----------
    model : nn.Module
        the NN used for regression
    loader : DataLoader
        an iterable over data batches
    criterion : Callable[[Tensor, Tensor], Tensor]
        a criterion (loss) that is optimized
    pbar : Optional[Any], optional
        a tqdm progress bar, by default None
    optim : Optional[torch.optim.Optimizer], optional
        a optimizer that is optimizing the criterion, by default None
    """

    def step(
        data_batch: Data,
    ) -> Tuple[float, float]:
        """Perform a single train/val step on a data batch.

        Parameters
        ----------
        data_batch : Data

        Returns
        -------
        Tuple[float, float]
            Loss (mean squared error) and validation critierion (absolute error).
        """
        pred = model.forward(data_batch)
        target = data_batch.y
        loss = criterion(pred, target)
        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()
        return loss.detach().item(), total_absolute_error(pred.detach(), target.detach())

    if optim is not None:
        model.train()
        # This enables pytorch autodiff s.t. we can compute gradients
        model.requires_grad_(True)
    else:
        model.eval()
        # disable autodiff: when evaluating we do not need to track gradients
        model.requires_grad_(False)

    total_loss = 0
    total_mae = 0
    for data in loader:
        loss, mae = step(data)
        total_loss += loss * data.num_graphs
        total_mae += mae
        if pbar is not None:
            pbar.update(1)

    return total_loss / len(loader.dataset), total_mae / len(loader.dataset)


def train_model(
    data_module: QM9DataModule,
    model: nn.Module,
    num_epochs: int = 30,
    lr: float = 3e-4,
    batch_size: int = 32,
    weight_decay: float = 1e-8,
    best_model_path: str = "trained_model.pth",
    ) -> Dict[str, Any]:
    """Takes data and model as input and runs training, collecting additional validation metrics
    while doing so.

    Parameters
    ----------
    data_module : QM9DataModule
        a data module as defined earlier
    model : nn.Module
        a gnn model
    num_epochs : int, optional
        number of epochs to train for, by default 30
    lr : float, optional
        "learning rate": optimizer SGD step size, by default 3e-4
    batch_size : int, optional
        number of examples used for one training step, by default 32
    weight_decay : float, optional
        L2 regularization parameter, by default 1e-8
    best_model_path : Path, optional
        path where the model weights with lowest val. error should be stored
        , by default DATA.joinpath("trained_model.pth")

    Returns
    -------
    Dict[str, Any]
        a training result, ie statistics and info about the model
    """
    # create data loaders
    train_loader = data_module.train_loader(batch_size=batch_size)
    val_loader = data_module.val_loader(batch_size=batch_size)

    # setup optimizer and loss
    optim = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-8)
    loss_fn = nn.MSELoss()

    # keep track of the epoch with the best validation mae
    # st we can save the "best" model weights
    best_val_mae = float("inf")

    # Statistics that will be plotted later on
    # and model info
    result = {
        "model": model,
        "path_to_best_model": best_model_path,
        "train_loss": np.full(num_epochs, float("nan")),
        "val_loss": np.full(num_epochs, float("nan")),
        "train_mae": np.full(num_epochs, float("nan")),
        "val_mae": np.full(num_epochs, float("nan")),
    }

    # Auxiliary functions for updating and reporting
    # Training progress statistics
    def update_statistics(i_epoch: int, **kwargs: float):
        for key, value in kwargs.items():
            result[key][i_epoch] = value

    def desc(i_epoch: int) -> str:
        return " | ".join(
            [f"Epoch {i_epoch + 1:3d} / {num_epochs}"]
            + [
                f"{key}: {value[i_epoch]:8.2f}"
                for key, value in result.items()
                if isinstance(value, np.ndarray)
            ]
        )

    # main training loop
    for i_epoch in range(0, num_epochs):
        progress_bar = tqdm(total=len(train_loader) + len(val_loader))
        try:
            # tqdm for reporting progress
            progress_bar.set_description(desc(i_epoch))

            # training epoch
            train_loss, train_mae = run_epoch(model, train_loader, loss_fn, progress_bar, optim)
            # validation epoch
            val_loss, val_mae = run_epoch(model, val_loader, loss_fn, progress_bar)

            update_statistics(
                i_epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_mae=train_mae,
                val_mae=val_mae,
            )

            progress_bar.set_description(desc(i_epoch))

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), best_model_path)
        finally:
            progress_bar.close()

    return result


@torch.no_grad()
def test_model(model: nn.Module, data_module: QM9DataModule) -> Tuple[float, Tensor, Tensor]:
    """
    Test a model.

    Parameters
    ----------
    model : nn.Module
        a trained model
    data_module : QM9DataModule
        a data module as defined earlier
        from which we'll get the test data

    Returns
    -------
    _Tuple[float, Tensor, Tensor]
        Test MAE, and model predictions & targets for further processing
    """
    test_mae = 0
    preds, targets = [], []
    loader = data_module.test_loader()
    for data in loader:
        pred = model(data)
        target = data.y
        preds.append(pred)
        targets.append(target)
        test_mae += total_absolute_error(pred, target).item()

    test_mae = test_mae / len(data_module.test_split)
    return test_mae, torch.cat(preds, dim=0), torch.cat(targets, dim=0)


egnn_train_result = train_model(
    data_module=data_module,
    model=model,
    num_epochs=EPOCHS,
    lr=LR,
    batch_size=BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
    best_model_path="trained_egnn.pth",
)
