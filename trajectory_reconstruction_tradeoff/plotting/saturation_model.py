import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.linear_model import LinearRegression, HuberRegressor


eps = 1e-2
class SaturationModel():
    """
    Saturation model with two linear models
    """
    def __init__(self, thr_saturation=0.01, model_type='huber') -> None:
        self.thr_saturation = thr_saturation
        self.model_type = model_type
        self.linear_model1_ = HuberRegressor() if model_type == 'huber' else LinearRegression()
        

    def fit(self, X, y):

        # with zero region
        X = np.array(X).flatten() # do we need to sort X?
        y = np.array(y).flatten() 
        assert len(X) == len(y)
        
        # use mean values for duplicate X
        df = pd.DataFrame({'X': X, 'y': y}).groupby('X').mean().reset_index()
        X, y = df['X'].values, df['y'].values

        x_max_id = X.argmax() # least sampled point
        # x_max_y = y[X == x_max] # error at least sampled point
        # x_max_y = x_max_y.mean() if len(x_max_y) > 1 else x_max_y[0]
        x_max_y = y[x_max_id] # error at least sampled point
        x_max_y_thr = x_max_y + self.thr_saturation # permitted error
        x_above_thr = X[y > x_max_y_thr] # points with error above permitted error
        if len(x_above_thr) <= 1 or len(x_above_thr) == len(X):
            # default to linear model
            self.linear_model1_.fit(X, y)
            self.x0_ = np.inf
        else:
            self.x0_ = x_above_thr.max() # threshold between the models
            
            idx1 = np.where(X < self.x0_)[0]
            if len(idx1) == 0:
                self.linear_model1_.predict = lambda x: np.full(len(x), x_max_y)
                self.linear_model1_.coef_ = np.array([0])
                self.linear_model1_.intercept_ = x_max_y
            else:
                sX1, sy1 = X[idx1], y[idx1]
                self.linear_model1_.fit(sX1.reshape((-1, 1)), sy1)

            # idx2 = np.where(X >= self.x0_)[0]
            self.y0_ = np.maximum(self.linear_model1_.predict(self.x0_.reshape((-1, 1)))[0], x_max_y)
            # if len(idx2) == 0:
            #     self.linear_model2_.predict = lambda x: np.full(len(x), x_max_y)
            #     self.linear_model2_.coef_ = np.array([0])
            #     self.linear_model2_.intercept_ = x_max_y
            # else:
            #     sX2, sy2 = X[idx2], y[idx2]
            #     self.linear_model2_.fit(sX2.reshape((-1, 1)), sy2)
        
        
        def predict(X_new):
            X_new = np.array(X_new)
            X_new = X_new.flatten()
            idx_new1 = np.where(X_new < self.x0_)[0]
            sX_new1 = X_new[idx_new1]

            idx_new2 = np.where(X_new >= self.x0_)[0]
            # sX_new2 = X_new[idx_new2]

            ynew = np.zeros(len(X_new))
            if len(idx_new1) > 0:
                ynew[idx_new1] = self.linear_model1_.predict(sX_new1.reshape((-1, 1)))
            if len(idx_new2) > 0:
                # ynew[idx_new2] = self.linear_model2_.predict(sX_new2.reshape((-1, 1)))
                ynew[idx_new2] = np.full(len(idx_new2), self.y0_)
            ynew[ynew < self.y0_] = self.y0_ # TODO: check this
            return ynew
        
        def score(X, y):
            y_pred = predict(X)
            return np.corrcoef(np.array(y).flatten(), y_pred)[0,1]**2
            # u = ((y - y_pred)** 2).sum()
            # v = ((y - y.mean()) ** 2).sum()
            # R2 = (1 - u/v)
            # return R2

        self.predict = predict
        self.score = score
        self.intercept_ = self.intercept1_ = self.linear_model1_.intercept_
        self.coef_ = self.coef1_ = self.linear_model1_.coef_

        # self.intercept2_ = self.linear_model2_.intercept_
        # self.coef2_ = self.linear_model2_.coef_

        return self


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    X = np.array([[0], [1], [2], [3], [4]]).reshape(-1, 1)
    y = np.array([0, 0, 1, 3, 5])[::-1]

    model_zl = SaturationModel()
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