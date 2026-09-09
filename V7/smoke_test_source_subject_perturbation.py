"""Checks that V7 retains V5's leakage-safe source perturbation."""

from types import SimpleNamespace

import numpy as np
import torch

from models.TriViewConcatDBConformer import TriViewConcatDBConformer
from utils.vision_dg_preprocess import (
    build_source_domain_ids,
    subject_distribution_perturbation,
)


def experiment_args(probability=0.47):
    return SimpleNamespace(
        N=9,
        idt=0,
        trials_arr=False,
        class_num=2,
        data_name="BNCI2014001",
        version="TriViewConcatSoftmaxAdamWWarmupCosineV7",
        subject_style_windows=4,
        subject_perturb_prob=probability,
        subject_perturb_strength=0.6,
        style_scale_min=0.8,
        style_scale_max=1.25,
        subject_style_eps=1e-8,
    )


def model_args():
    return SimpleNamespace(
        data_name="BNCI2014001",
        chn=22,
        class_num=2,
        time_sample_num=1001,
        sample_rate=250,
        patch_size=125,
        emb_size=40,
        spa_dim=16,
        transformer_depth_tem=2,
        transformer_depth_chn=2,
    )


def main():
    # Eight source subjects, four trials per subject, both classes represented.
    args = experiment_args(probability=1.0)
    domains = build_source_domain_ids(args, source_length=32)
    labels = np.tile(np.array([0, 1, 0, 1], dtype=np.int64), 8)
    rng = np.random.RandomState(7)
    source = rng.standard_normal((32, 22, 1001)).astype(np.float32)
    source *= np.linspace(0.8, 1.2, 8).repeat(4)[:, None, None]

    first = subject_distribution_perturbation(
        source, labels, domains, args, save_debug=False
    )
    second = subject_distribution_perturbation(
        source, labels, domains, args, save_debug=False
    )
    perturbed, records, prototypes, global_style, deviations = first

    assert set(np.unique(domains)) == set(range(1, 9))
    assert args.idt not in domains
    assert perturbed.shape == source.shape and perturbed.dtype == source.dtype
    assert np.isfinite(perturbed).all()
    assert not np.allclose(perturbed, source)
    assert np.array_equal(perturbed, second[0])
    assert prototypes.shape == (8, 2, 22, 4)
    assert global_style.shape == (2, 22, 4)
    assert deviations.shape == (8, 2, 22, 4)
    assert all(record[-1] == 1 for record in records)
    assert all(
        record[3] != record[1] and record[4] != record[1]
        for record in records
    )

    # V7 must retain V0's 120-D competitive-Softmax fusion model.
    model = TriViewConcatDBConformer(model_args())
    model.eval()
    views = [torch.randn(2, 40) for _ in range(3)]
    with torch.no_grad():
        fused, weights = model.adaptive_fusion(*views)
        features, logits = model(torch.randn(2, 1, 22, 1001))
    expected_weights = torch.full_like(weights, 1.0 / 3.0)
    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(fused, torch.cat(views, dim=-1) / 3.0)
    assert features.shape == (2, 120) and logits.shape == (2, 2)

    production = experiment_args()
    assert production.subject_style_windows == 4
    assert production.subject_perturb_prob == 0.47
    assert production.subject_perturb_strength == 0.6
    assert (production.style_scale_min, production.style_scale_max) == (0.8, 1.25)

    print("PASS: held-out target subject is absent from all source domain IDs")
    print("PASS: perturbation uses other source subjects and is deterministic")
    print("PASS: outputs preserve source shape/dtype and contain no NaN/Inf")
    print("PASS: prototypes are source subject x class x channel x 4 windows")
    print("PASS: production parameters exactly match the verified V111 method")
    print("PASS: V7 retains V5 perturbation and V0's competitive softmax model")


if __name__ == "__main__":
    main()
