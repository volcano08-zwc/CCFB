"""Smoke checks for V7's AdamW plus warmup/cosine training change."""

from types import SimpleNamespace

import torch

from DBConformer_LOSO import build_optimizer, learning_rate_at_step


def main():
    network = torch.nn.Linear(3, 2)
    args = SimpleNamespace(lr=1e-3, weight_decay=1e-4)
    optimizer = build_optimizer(network, args)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]['weight_decay'] == 1e-4

    total, warmup = 100, 5
    assert learning_rate_at_step(1e-3, 1, total, warmup) == 2e-4
    assert learning_rate_at_step(1e-3, 5, total, warmup) == 1e-3
    assert 0.0 < learning_rate_at_step(1e-3, 50, total, warmup) < 1e-3
    assert abs(learning_rate_at_step(1e-3, total, total, warmup)) < 1e-12

    print('PASS: production optimizer is AdamW with weight_decay=1e-4')
    print('PASS: learning rate warms up for five epochs then decays by cosine')


if __name__ == '__main__':
    main()
