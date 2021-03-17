import sys
import numpy as np

sys.path.append('..')

from parker import Network, ActivationMethods


def test_back_prop():
    EXPECTED_VALUE = 0.53639513

    test_network = Network(2,[2],1, ActivationMethods.SIGMOID)
    
    test_network.weight_matrix = [np.array([[3,6],[4,5]]),
                                np.array([[2],[4]])]
    test_network.bias_matrix = [np.array([[1, -6]]), np.array([[-3.92]])]
    
    for i in range(2):  # train twice
        test_network.train(np.array([[1,0]]), np.array([[1]]))
    
    assert abs(test_network.feed_forward([[1,0]])[1][0][0]-EXPECTED_VALUE) < 0.00000001  # check if its close (double issues)

