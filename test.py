import unittest

import torch

from model import GestureClassifier


class TestModel(unittest.TestCase):

    def test_pointer_network_in_out(self):
        model = GestureClassifier(20)

        gesture = model(torch.Tensor(1, 20), torch.Tensor(1, 20))

        self.assertTrue((1, 5) == gesture.size())


if __name__ == "__main__":
    unittest.main()
