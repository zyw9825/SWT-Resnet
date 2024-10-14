import torch

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, start_epoch=0):
    """
    Train and validate the model for a specified number of epochs.

    Parameters:
    - train_loader: DataLoader for the training data.
    - val_loader: DataLoader for the validation data.
    - model: The neural network model to be trained.
    - loss_fn: Loss function to evaluate the model's performance.
    - optimizer: Optimization algorithm used for training.
    - scheduler: Learning rate scheduler.
    - n_epochs: Total number of epochs for training.
    - start_epoch: The starting epoch (default is 0).
    """
    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Training phase
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        train_loss /= len(train_loader)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)

        # Validation phase
        val_loss = test_epoch(val_loader, model, loss_fn)
        val_loss /= len(val_loader)
        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, val_loss)


def train_epoch(train_loader, model, loss_fn, optimizer):
    """
    Train the model for one epoch.

    Parameters:
    - train_loader: DataLoader for the training data.
    - model: The neural network model to be trained.
    - loss_fn: Loss function to evaluate the model's performance.
    - optimizer: Optimization algorithm used for training.

    Returns:
    - train_loss: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        target = target.cuda()  # Move target to GPU
        optimizer.zero_grad()  # Reset gradients
        outputs = model(data)  # Forward pass
        loss = loss_fn(outputs, target)  # Compute loss
        total_loss += loss.item()
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

    train_loss = total_loss / len(train_loader)
    return train_loss


def test_epoch(val_loader, model, loss_fn):
    """
    Evaluate the model on the validation set and calculate accuracy.

    Parameters:
    - val_loader: DataLoader for the validation data.
    - model: The neural network model to be evaluated.
    - loss_fn: Loss function to evaluate the model's performance.

    Returns:
    - val_loss: The average validation loss for the epoch.
    - accuracy: The accuracy of the model on the validation set.
    """
    with torch.no_grad():  # Disable gradient calculation for evaluation
        model.eval()  # Set model to evaluation mode
        total_loss = 0
        correct = 0  # Track number of correct predictions
        total = 0    # Track total number of samples

        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.cuda()  # Move data to GPU if using CUDA
            target = target.cuda()  # Move target to GPU if using CUDA

            outputs = model(data)  # Forward pass
            loss = loss_fn(outputs, target)  # Compute loss
            total_loss += loss.item()  # Accumulate loss

            # Get predicted class and calculate number of correct predictions
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            correct += (predicted == target).sum().item()  # Count correct predictions
            total += target.size(0)  # Count total samples

        val_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total if total > 0 else 0  # Calculate accuracy

        print(f'Validation Loss: {val_loss:.6f}, Accuracy: {accuracy:.2f}%')
        return val_loss, accuracy

