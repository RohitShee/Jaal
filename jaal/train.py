def fit(model, loss_fn, optimizer, loader, epochs, print_every=10):
    history = []
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0

        for xb, yb in loader:
            optimizer.zero_grad()          # 1. clear old gradients
            pred = model(xb)               # 2. forward
            loss = loss_fn(pred, yb)
            model.backward(loss_fn.backward())   # 3. backward
            optimizer.step()               # 4. update weights

            n = len(xb)
            total_loss += loss * n         # weight by batch size
            total_samples += n

        epoch_loss = total_loss / total_samples
        history.append(epoch_loss)

        if epoch % print_every == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  loss {epoch_loss:.6f}")

    return history
