import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval, device):
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

    step = 0
    
    writer = SummaryWriter()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            # TODO: Forward propagate
            outputs = model(features)
            # TODO: Backpropagation and gradient descent
            loss = loss_fn(outputs, labels)
            loss.backwards()
            optimizer.step()
            optimizer.zero_grad()

            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                predictions = torch.argmax(outputs, dim=1)
                acc = compute_accuracy(predictions, labels)
                writer.add_scalar('Loss/train', loss, epoch)
                writer.add_scalar('Accuracy/train', acc, epoch)

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard. 
                # Don't forget to turn off gradient calculations!
                loss, acc = evaluate(val_loader, model, loss_fn, device) 
                writer.add_scalar('Loss/val', loss, epoch)
                writer.add_scalar('Accuracy/val', acc, epoch)
                #turn on training, evaluate turns off training
                model.train()

            step += 1

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


def evaluate(val_loader, model, loss_fn, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    #turn off training
    model.eval()

    features, labels = val_loader
    features = features.to(device)
    labels = labels.to(device)
    
    outputs = model(features)
    
    loss = loss_fn(outputs, labels)
    predictions = torch.argmax(outputs, dim=1)

    return loss, compute_accuracy(outputs, labels)
