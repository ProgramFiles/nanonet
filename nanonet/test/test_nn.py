import unittest

import nanonet.nn as nn
import numpy as np

class ANNTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print '* ANN'
        np.random.seed(0xdeadbeef)
        self._NSTEP = 10
        self._NFEATURES = 4
        self._SIZE = 5

        self.W = np.random.normal(size=(self._NFEATURES, self._SIZE))
        self.b = np.random.normal(size=self._SIZE)
        self.x = np.random.normal(size=(self._NSTEP, self._NFEATURES))
        self.res = self.x.dot(self.W) + self.b

    def test_000_single_layer_linear(self):
        network = nn.layer(self.W, self.b, nn.linear)
        self.assertEqual(network.in_size(), self._NFEATURES)
        self.assertEqual(network.out_size(), self._SIZE)
        np.testing.assert_array_equal(network.run(self.x), self.res)

    def test_001_single_layer_tanh(self):
        network = nn.layer(self.W, self.b, nn.tanh)
        self.assertEqual(network.in_size(), self._NFEATURES)
        self.assertEqual(network.out_size(), self._SIZE)
        np.testing.assert_array_equal(network.run(self.x), np.tanh(self.res))

    def test_002_parallel_layers(self):
        l1 = nn.layer(self.W, self.b, nn.tanh)
        l2 = nn.layer(self.W, self.b, nn.tanh)
        network = nn.parallel([l1, l2])
        self.assertEqual(network.in_size(), self._NFEATURES)
        self.assertEqual(network.out_size(), 2 * self._SIZE)

        res = network.run(self.x)
        np.testing.assert_array_equal(res[:,:self._SIZE], res[:,self._SIZE:])

    def test_003_simple_serial(self):
        W2 = np.random.normal(size=(self._SIZE, self._SIZE))
        res = self.x.dot(self.W).dot(W2)

        l1 = nn.layer(self.W, fun=nn.linear)
        l2 = nn.layer(W2, fun=nn.linear)
        network = nn.serial([l1, l2])
        self.assertEqual(network.in_size(), self._NFEATURES)
        self.assertEqual(network.out_size(), self._SIZE)

        np.testing.assert_array_equal(network.run(self.x), res)

    def test_004_reverse(self):
        network1 = nn.layer(self.W, self.b, nn.tanh)
        res1 = network1.run(self.x)
        network2 = nn.reverse(network1)
        res2 = network2.run(self.x)
        self.assertEqual(network1.in_size(), network2.in_size())
        self.assertEqual(network1.out_size(), network2.out_size())

        np.testing.assert_array_equal(res1, res2)

    def test_005_poormans_birnn(self):
        layer1 = nn.layer(self.W, self.b, nn.tanh)
        layer2 = nn.layer(self.W, self.b, nn.tanh)
        network = nn.birnn(layer1, layer2)

        res = network.run(self.x)
        np.testing.assert_array_equal(res[:,:self._SIZE], res[:,self._SIZE:])

    def test_006_softmax(self):
        network = nn.softmax(self.W, self.b)

        res = network.run(self.x)
        res_sum = res.sum(axis=1)
        self.assertTrue(np.allclose(res_sum, 1.0))

    def test_007_rnn_no_state(self):
        W1 = np.vstack((np.zeros((self._SIZE, self._SIZE)), self.W))
        network = nn.rnn_layer(W1, b=self.b, fun=nn.linear)

        res = network.run(self.x)
        np.testing.assert_almost_equal(res, self.res)

    def test_008_rnn_no_input(self):
        W1 = np.random.normal(size=(self._SIZE, self._SIZE))
        W2 = np.vstack((W1, np.zeros((self._NFEATURES, self._SIZE))))
        network = nn.rnn_layer(W2, fun=nn.linear)

        res = network.run(self.x)
        np.testing.assert_almost_equal(res, 0.0)

    def test_009_rnn_no_input_with_bias(self):
        W1 = np.random.normal(size=(self._SIZE, self._SIZE))
        W2 = np.vstack((W1, np.zeros((self._NFEATURES, self._SIZE))))
        network = nn.rnn_layer(W2, b=self.b, fun=nn.linear)

        res = network.run(self.x)
        res2 = np.zeros(self._SIZE)
        for i in xrange(self._NSTEP):
            res2 = res2.dot(W1) + self.b
            np.testing.assert_almost_equal(res[i], res2)

    def test_010_birnn_no_input_with_bias(self):
        W1 = np.random.normal(size=(self._SIZE, self._SIZE))
        W2 = np.vstack((W1, np.zeros((self._NFEATURES, self._SIZE))))
        layer1 = nn.rnn_layer(W2, b=self.b)
        layer2 = nn.rnn_layer(W2, b=self.b)
        network = nn.birnn(layer1, layer2)

        res = network.run(self.x)
        np.testing.assert_almost_equal(res[:,:self._SIZE], res[::-1,self._SIZE:])
