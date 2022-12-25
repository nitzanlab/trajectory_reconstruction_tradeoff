import numpy as np
from scipy import optimize
from sklearn.linear_model import LinearRegression


eps = 1e-2
class ZeroLinearModel():
    def __init__(self) -> None:
        pass

    def fit(self, X, y):

        # with zero region
        X = np.array(X)
        def zero_linear(X, x0, y0, k2): # TODO: handle multi dimensions
            x = X[:, 0]
            return np.piecewise(x, [x < x0], [lambda x: eps, lambda x:k2*x + y0-k2*x0])
    
        coef_ , e = optimize.curve_fit(zero_linear, X, y) 

        def predict(X_new):
            return zero_linear(np.array(X_new), *coef_)
        
        def score(X, y):
            y_pred = predict(X)
            u = ((y - y_pred)** 2).sum()
            v = ((y - y.mean()) ** 2).sum()
            R2 = (1 - u/v)
            return R2

        self.predict = predict
        self.score = score
        self.x0_ = coef_[0]
        self.intercept_ = coef_[1]
        self.coef_ = coef_[2]

        # without zero region
        model_l = LinearRegression()
        model_l.fit(X, y)
        
        if model_l.score(X, y) > self.score(X, y):
            self.predict = model_l.predict
            self.score = model_l.score
            self.x0_ = None
            self.intercept_ = model_l.intercept_
            self.coef_ = model_l.coef_

        return self


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    X = np.array([[0], [1], [2], [3], [4]]).reshape(-1, 1)
    y = np.array([0, 0, 1, 3, 5])

    model_zl = ZeroLinearModel()
    model_zl.fit(X, y)
    print(model_zl.predict(X))
    print(model_zl.score(X, y))
    x = X
    plt.plot(x, y, 'o')
    plt.plot(x, model_zl.predict(X), label='zero_linear')

    from sklearn.linear_model import LinearRegression
    model_l = LinearRegression()
    model_l.fit(X, y)
    print(model_l.predict(X))
    print(model_l.score(X, y))
    plt.plot(x, model_l.predict(X), label='linear')


    plt.show()