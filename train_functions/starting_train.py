import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    loss_fn = nn.BCELoss()

    step = 0
    loss_history = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        # Loop over each batch in the dataset
        for batch in enumerate(train_loader):
            # TODO: Forward propagate

            #print("Loop Begin")
            _, (sequences, labels) = batch
            sequences = sequences.float()
            # Move inputs over to GPU
            sequences = sequences.to(device)
            labels = labels.to(device)
            # Forward propagation
            outputs = model(sequences) # Same thing as model.forward(sequences)

            # TODO: Backpropagation and gradient descent
            loss = loss_fn(outputs.squeeze(), labels.type(torch.float32))
            loss.backward()       # Compute gradients
            loss_history.append(loss.cpu().item())
            optimizer.step()      # Update all the weights with the gradients you just calculated
            optimizer.zero_grad() # Clear gradients before next iteration
            print('Epoch:', epoch, 'Loss:', loss.item())


            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard. 
                # Don't forget to turn off gradient calculations!
                evaluate(val_loader, model, loss_fn)

            step += 1
    return loss_history


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """
    print(outputs)
    print(labels)
    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    #print("correct is: " + str(n_correct))
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    
    """
    model.eval()
    with torch.no_grad():
        for batch in enumerate(val_loader):
            # TODO: Forward propagate
            #print("Loop Begin")
            _, (sequences, labels) = batch
            sequences = sequences.float()
            # Move inputs over to GPU
            sequences = sequences.to(device)
            labels = labels.to(device)
            # Forward propagation
            outputs = model(sequences) # Same thing as model.forward(sequences)

            # TODO: Backpropagation and gradient descent
            # loss = loss_fn(outputs.squeeze(), labels.type(torch.float32))
        
            acc = compute_accuracy(outputs, labels)/64 #percentage that's correct per
            print("accuracy is: " + str(acc))
    model.train()
    pass
