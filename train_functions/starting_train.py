import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, tqdm_notebook


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss() #loss function 

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        curLoss = 0.0
        # Loop over each batch in the dataset
        for step, batch in enumerate(tqdm(train_loader), 0):
            # TODO: redo validation split
            statements, labels = batch
            optimizer.zero_grad()

            statements = statements.type(torch.float)
            output = model.forward(statements)
            
            
            # TODO: Backpropagation and gradient descent
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            # Periodically evaluate our model + log to Tensorboard
            curLoss += loss.item()
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                print('Loss after mini-batch %5d: %.3f' %
                (step + 1, curLoss / 500))
                curLoss = 0
                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard. 
                # Don't forget to turn off gradient calculations!
                evaluate(val_loader, model, loss_fn)


        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    pass
