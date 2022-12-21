import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def SGD(X, y, lam=0, epochs=1000, l_rate=0.01, sgd_type="practical"):
    np.random.seed(2)
    m = X.shape[0]
    d = X.shape[1]
    w = np.random.uniform(size=d)
    b = np.random.uniform(size=1)

    if sgd_type == "theory":
        w_list = [w]
        b_list = [b]
        for _ in range(m * epochs):
            idx = np.random.randint(low=0, high=m - 1, size=1)
            sample = X[idx]
            v_w, v_b = sub_gradient(w, b, lam, sample, y[idx])

            w -= l_rate * v_w
            b -= l_rate * v_b

            w_list.append(w)
            b_list.append(b)
        w_mean = np.average(w_list)
        b_mean = np.average(b_list)
        return w_mean, b_mean
    else:
        for _ in range(epochs):
            permutation = np.random.permutation(np.arange(m))
            for i in permutation:
                v_w, v_b = sub_gradient(w, b, lam, X[i], y[i])
                w -= l_rate * v_w
                b -= l_rate * v_b
        return w, b


def sub_gradient(w, b, lam, x, y):
    if 0 > 1 - y * (np.inner(w, x) + b):
        return 2 * lam * w, 0
    else:
        v_w = (-y * x) + (2 * lam * w)
        v_b = -y
        return v_w, v_b


# %%
def calculate_error(w, bias, X, y):
    y_pred = [np.sign(np.inner(x, w) + bias) for x in X]
    # print(f'y pred = {y_pred}')
    # print(f'y true = {y}')
    return np.average(y_pred != y)  # error-rate


def main(name):
    X, y = load_iris(return_X_y=True)
    X = X[y != 0]
    y = y[y != 0]
    y[y == 2] = -1
    X = X[:, 2:4]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)
    lam_list = np.array([0, 0.05, 0.1, 0.2, 0.5])

    models = [SGD(X=X_train, y=y_train, lam=l) for l in lam_list]
    margins = [1 / np.linalg.norm(m[0]) for m in models]
    train_error_rates = [calculate_error(w=m[0], bias=m[1][0], X=X_train, y=y_train) for m in models]
    val_error_rates = [calculate_error(w=m[0], bias=m[1][0], X=X_val, y=y_val) for m in models]

    # fig, axs = plt.subplots(2, 1)
    # fig.tight_layout()
    barWidth = 0.4
    bar = np.arange(len(lam_list))

    plt.bar(bar - 0.2, train_error_rates, width=barWidth, label='train error',
            edgecolor='black', align='center')
    plt.bar(bar + 0.2, val_error_rates, width=barWidth, label='val error',
            edgecolor='black', align='center')
    plt.xlabel('\u03BB')
    plt.ylabel('Error Rate')
    plt.title("Train and validation Error rate per lambda value")
    plt.legend()
    plt.show()

    plt.bar(bar, margins, edgecolor='black', align='center', width=barWidth)
    plt.xlabel('\u03BB')
    plt.ylabel('Margin width')
    plt.title("Margin  per lambda value")

    plt.show()


if __name__ == '__main__':
    main('PyCharm')
