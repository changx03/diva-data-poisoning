import numpy as np
from scipy.optimize import linprog
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from tqdm import tqdm


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

    result = linprog(
        coef,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=q_bound,
        method='interior-point',
        options={'sym_pos': False, 'lstsq': True, 'presolve': True}
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


def get_flip_labels(y, q, C):
    n = len(y)
    C = int(np.floor(n * C))
    idx_flip = np.argsort(q[n:])[::-1][:C]
    y_flip = y.copy()
    y_flip[idx_flip] = -y_flip[idx_flip]
    return y_flip


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


def evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)
    print('Train acc: {:.2f} Test acc: {:.2f}'.format(
        acc_train*100, acc_test*100))


def run_demo():
    SEED = 1234
    # 500 for training and 500 for testing
    N_SAMPLES = 1000
    N_FEATURES = 2
    POISON_RATE = 0.1

    X, y = datasets.make_blobs(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        centers=[[-0.5, 0.5], [0.5, -0.5]],
        cluster_std=0.4,
        random_state=SEED,
    )

    scaler = MinMaxScaler((-1, 1))
    X = scaler.fit_transform(X)

    # Labels should be {-1, 1}
    idx_0 = np.where(y == 0)[0]
    y[idx_0] = -1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=500, random_state=SEED)

    svc_params = {'C': 100, 'kernel': 'linear', 'random_state': SEED}
    y_poison = alfa(X_train, y_train, budget=POISON_RATE,
                    svc_params=svc_params)

    print('Poison rate: {:.2f}'.format(np.mean(y_poison != y_train)))

    clf = SVC(**svc_params)
    print('Trained on clean data')
    evaluate(clf, X_train, y_train, X_test, y_test)

    clf = SVC(**svc_params)
    print('Trained on poisoned data')
    evaluate(clf, X_train, y_poison, X_test, y_test)


if __name__ == '__main__':
    run_demo()
