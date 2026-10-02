import unittest
import torch
from cosp.model import CoSP, training_components, train_step


class ModelTests(unittest.TestCase):
    def test_forward_backward_and_paper_optimizer(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        model = CoSP()
        optimizer, loss = training_components(model)
        self.assertIs(type(optimizer), torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-5)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0)
        x = torch.rand(2,1,120,109)
        before = model.head[-1].weight.detach().clone()
        value = train_step(model, optimizer, loss, x, torch.tensor([0.,1.]))
        self.assertTrue(value > 0)
        self.assertFalse(torch.equal(before, model.head[-1].weight))
        model.eval()
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(tuple(logits.shape), (2,1))
        self.assertTrue(torch.isfinite(logits).all())
        with self.assertRaises(ValueError):
            model(torch.rand(2,1,109,120))

