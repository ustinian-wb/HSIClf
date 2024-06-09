# --*-- conding:utf-8 --*--
# @Time  : 2024/6/9
# @Author: weibo
# @Email : csbowei@gmail.com
# @File  : elm.py
# @Description:

import numpy as np
from hpelm import ELM


class ELMClassifier:
    def __init__(self, input_dim, output_dim, hidden_neurons=1000, activation_function="sigm"):
        # self.elm = ELM(input_dim, output_dim, classification="c", batch=1000, accelerator="GPU", norm=1)
        self.elm = ELM(input_dim, output_dim, classification="c", batch=1000, norm=1)
        self.elm.add_neurons(hidden_neurons, activation_function)

    def train(self, data, regularization_params):
        X, Y = data
        for lam in regularization_params:
            self.elm.train(X, Y, "c", "LOO", k=10, lambda_=lam)

    def classify(self, X):
        Y_pred = self.elm.predict(X)
        return Y_pred
