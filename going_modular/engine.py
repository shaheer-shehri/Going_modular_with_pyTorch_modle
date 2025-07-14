"""Contains functions for training and testing a PyTorch model."""

import torch

from tqdm.auto import tqdm


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn=torch.nn.Module,
    optimizer=torch.optim.Optimizer,
    device=torch.device,
):
    """Trains a PyTorch model for a single epoch.

        Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """

    # put the model to train mode
    model.train()

    # setup train loss and accuracy to zero
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)

        # calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        # optimizer zero grad
        optimizer.zero_grad()

        # loss backwards
        loss.backward()

        # optimizer step
        optimizer.step()

        # calculate and accumulate the accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / (len(data_loader))
    train_acc = train_acc / len(data_loader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn=torch.nn.Module,
    optimizer=torch.optim.Optimizer,
    device=torch.device,
):
    """Tests a PyTorch model for a single epoch.
    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
    """

    model.eval()

    test_acc, test_loss = 0, 0
    for batch, (X, y) in enumerate(data_loader):
         X, y = X.to(device), y.to(device)
        # forward pass
        y_pred_logits = model(X)

        loss = loss_fn(y_pred_logits, y)
        test_loss += loss.item()
        y_pred_prob = torch.softmax(y_pred_logits, dim=1)
        y_pred_labels = torch.argmax(y_pred_prob, dim=1)

        test_acc += (y_pred_labels == y).sum().item() / len(y_pred_labels)

    test_loss = test_loss / len(data_loader)
    test_acc = test_acc / len(data_loader)
    return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
):
    """Train and test a PyTorch model.
    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for
        each epoch.
        In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]}
        For example if training for epochs=2:
                    {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]}
    """

    results = {"test_acc": [], "train_acc": [], "test_loss": [], "train_loss": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            data_loader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        test_loss, test_acc = test_step(
            model=model,
            data_loader=test_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        print(
            f"Epoch: {epoch + 1} |"
            f"train_loss {train_loss} |"
            f"train_acc: {train_acc} |"
            f"test_loss: {test_loss} |"
            f"test_acc: {test_acc}"
        )

        results["test_acc"].append(test_acc)
        results["test_loss"].append(test_loss)
        results["train_acc"].append(train_acc)
        results["train_loss"].append(train_loss)
    return results
