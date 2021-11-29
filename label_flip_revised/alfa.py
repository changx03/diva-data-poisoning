import numpy as np
from scipy.optimize import linprog
from sklearn.svm import SVC
from tqdm import tqdm


def get_flip_labels(y, q, C):
    n = len(y)
    C = int(np.floor(n * C))
    idx_flip = np.argsort(q[n:])[::-1][:C]
    y_flip = y.copy()
    y_flip[idx_flip] = -y_flip[idx_flip]
    return y_flip


def solveLP(eps, psi, C):
    n2 = len(eps)  # U is duplicated X_train with inverted y_train.
    n = int(n2 / 2)

    # Objective function coefficient
    coef = eps - psi

    # Inequality constraint: <=
    A_ub = np.array([[0.]*n + [1.]*n])
    # We consider the weight c_i for each sample are the same.
    b_ub = [n * C]  # C controls the rate of training data we can change.

    # Equality constraint: =
    A_eq = np.hstack((np.identity(n), np.identity(n)))
    b_eq = np.ones(n)

    # Boundary condition
    q_bound = np.array([[0., 1.]]*n2)

    # The optimization is NOT guaranteed to be solvable.
    result = linprog(
        coef,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=q_bound,
        method='interior-point',
        # options={'sym_pos': False, 'lstsq': True, 'presolve': True}
    )
    q = result.x
    return q, result.message


def solveQP(q, U_X, U_y, C, svc_params):
    n2 = len(U_y)
    n = int(n2 / 2)
    eps = np.zeros(n2)
    X_train = U_X[:n]
    y_adv = get_flip_labels(U_y[:n], q, C)

    clf = SVC(**svc_params)
    clf.fit(X_train, y_adv)
    f_U_X = clf.decision_function(U_X)
    eps = 1 - f_U_X * U_y
    eps[eps < 0] = 0
    return eps


def alfa(X_train, y_train, budget, svc_params, max_iter=5):
    clf = SVC(**svc_params)
    clf.fit(X_train, y_train)

    U_X = np.vstack((X_train, X_train))
    y_invert = -y_train
    U_y = np.concatenate((y_train, y_invert))

    f_U_X = clf.decision_function(U_X)
    psi = 1 - f_U_X * U_y
    psi[psi < 0] = 0
    eps = np.zeros_like(U_y)

    pbar = tqdm(range(max_iter), ncols=100)
    for _ in pbar:
        q, msg = solveLP(eps, psi, C=budget)
        pbar.set_postfix({'Optimizer': msg})
        eps = solveQP(q, U_X, U_y, C=budget, svc_params=svc_params)

    y_flip = get_flip_labels(y_train, q, budget)
    return y_flip
