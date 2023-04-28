import numpy as np
from criteria import rank_loss


class PCC:
    def __init__(self, cost, base_learner, params={}, n_sample=100):
        self.cost = cost
        self.base_learner = base_learner
        self.params = params
        self.n_sample = n_sample

    def fit(self, x_train, y_train):
        self.K = y_train.shape[1]
        self.clfs = [self.base_learner(**self.params) for i in range(self.K)]
        for i in range(self.K):
            self.clfs[i].fit(
                np.concatenate((x_train, y_train[:, :i]), axis=1), y_train[:, i]
            )

    def predict(self, x_test):
        pb_test = self.predict_prob(x_test)
        pred_test = np.zeros((x_test.shape[0], self.K), dtype=int)
        # shape = (n_sample, K)
        # print(r_test.shape, p_test.shape)
        for i in range(x_test.shape[0]):
            pred_test[i, :] = self.predict_one(x_test[i, :], pb_test[i, :])
        return pred_test

    def predict_prob(self, x_test):
        r_test = np.zeros((x_test.shape[0], self.K))
        for i in range(self.K):
            r_test[:, i] = (
                1.0
                - self.clfs[i].predict_proba(
                    np.concatenate((x_test, (r_test[:, :i] > 0.5).astype(int)), axis=1)
                )[:, 0]
            )
        return r_test

    def predict_one(self, x, pb):
        if self.cost == "ham":
            return (pb > 0.5).astype(int)
        # self.n_sample is to select n_sample random samples from the distribution
        # prob = np.repeat(pb, self.n_sample).reshape((pb.shape[0], self.n_sample)).T
        prob = pb
        # print(prob.shape, pb.shape)
        y_sample = (np.random.random((self.n_sample, self.K)) < prob).astype(int)
        # print(y_sample.shape, y_sample[0])
        # TODO: why did they need to < prob ?
        if self.cost == "rank":
            thr = 0.0
            pred = (pb > thr).astype(int)
            p_sample = (
                np.repeat(pred, self.n_sample).reshape((pred.shape[0], self.n_sample)).T
            )
            score = rank_loss(y_sample, p_sample).mean()
            for p in pb:
                pred = (pb > p).astype(int)
                p_sample = (
                    np.repeat(pred, self.n_sample)
                    .reshape((pred.shape[0], self.n_sample))
                    .T
                )
                score_t = rank_loss(y_sample, p_sample).mean()
                if score_t < score:
                    score = score_t
                    thr = p
            return (pb > thr).astype(int)
        elif self.cost == "f1":
            # self.n_sample is the number of point of data (sample).
            s_idxs = y_sample.sum(axis=1)
            # print(s_idxs)
            P = np.zeros((self.K, self.K))
            for i in range(self.K):
                P[i, :] = (
                    y_sample[s_idxs == (i + 1), :].sum(axis=0) * 1.0 / self.n_sample
                )
            # print(P.shape)
            # print(P[0])
            W = 1.0 / (
                np.cumsum(np.ones((self.K, self.K)), axis=1)
                + np.cumsum(np.ones((self.K, self.K)), axis=0)
            )
            F = P * W
            idxs = (-F).argsort(axis=1)
            H = np.zeros((self.K, self.K), dtype=int)
            for i in range(self.K):
                H[i, idxs[i, : i + 1]] = 1
            scores = (F * H).sum(axis=1)
            # print(H.shape)
            # print(scores)
            pred = H[scores.argmax(), :]
            # if (s_idxs==0).mean() > 2*scores.max():
            # 	pred = np.zeros((self.K, ), dtype=int)
            # print(pred.shape)

            return pred
        elif self.cost == "mar":
            pb.sort(reversed=True)
            f = np.zeros((self.K, self.K))
            sum_pb = sum([p for p in pb])

            f[self.K - 1] = 1 + (1 / self.K) * sum_pb
            for idx in range(0, self.K - 1):
                l = idx + 1
                f[idx] = (
                    1
                    - (1 / (self.K - 1)) * sum_pb
                    + (self.K / ((self.K - l) * l)) * sum([p for p in pb[:l]])
                )

            l_max = f.argmax()
            return (pb > pb[l_max]).astype(int)
        elif self.cost == "inf":
            # f = [0] * self.K
            # f[self.K - 1] =
            pass
