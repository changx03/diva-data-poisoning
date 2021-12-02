import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linprog
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .torch_utils import train_model
from .utils import flip_binary_label


def solveLPNN(alpha, tau, y_true, eps):
    n = len(alpha)

    c = alpha - tau

    # constraint: <=
    A_ub = np.copy(y_true).astype(float)
    A_ub[A_ub == 1] = -1
    A_ub[A_ub == 0] = 1
    A_ub = A_ub.reshape(1, -1)
    b_ub = np.dot(A_ub, y_true) + n * eps

    # Boundary
    bounds = np.array([[0., 1.]] * n)

    # The initial value must be valid.
    x0 = np.copy(y_true)
    idx = np.random.choice(n, size=int(np.floor(n * eps)), replace=False)
    x0 = flip_binary_label(y_true, idx)

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, x0=x0,
        bounds=bounds, method='revised simplex')
    y_poison = result.x
    return y_poison, result.message


def get_dual_loss(model, X, device):
    model.eval()
    with torch.no_grad():
        output = model(X.to(device))
        log_p = F.log_softmax(output, dim=1).cpu().detach().numpy()

    assert log_p.shape == (len(X), 2), \
        f'log_softmax expected dim: {(len(X), 2)}, received: {log_p.shape}'
    loss_dual = -log_p[:, 1] + log_p[:, 0]
    return loss_dual


def alfa_nn(model,
            X_train,
            y_train,
            eps,
            max_epochs,
            optimizer,
            loss_fn,
            batch_size,
            device,
            steps=3):
    # Compute the initial output
    X_train_tensor = torch.from_numpy(X_train).type(torch.float32)
    tau = get_dual_loss(model, X_train_tensor, device)
    alpha = np.zeros_like(tau)
    y_poison = np.copy(y_train).astype(int)

    pbar = tqdm(range(steps), ncols=100)
    for step in pbar:
        y_poison_next, msg = solveLPNN(alpha, tau, y_true=y_train, eps=eps)
        y_poison_next = np.round(y_poison_next).astype(int)
        pbar.set_postfix({'Optimizer': msg})

        if step > 1 and np.all(y_poison_next == y_poison):
            print('Poison labels are converged. Break.')
            break
        y_poison = y_poison_next

        # Update model
        dataset = TensorDataset(torch.from_numpy(X_train).type(torch.float32),
                                torch.from_numpy(y_poison).type(torch.int64))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_model(model,
                    dataloader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device,
                    max_epochs=max_epochs)
        alpha = get_dual_loss(model, X_train_tensor, device)
    return y_poison
