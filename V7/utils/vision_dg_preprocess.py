"""EEG-specific source-subject distribution perturbation."""
import csv
from pathlib import Path

import numpy as np


def build_source_domain_ids(args, source_length):
    subjects = [s for s in range(args.N) if s != args.idt]
    raw = getattr(args, 'trials_arr', False)
    if not isinstance(raw, (bool, np.bool_)) and raw is not None:
        counts = np.asarray(raw, dtype=int).reshape(-1)
        if len(counts) != args.N:
            raise ValueError('trials_arr must contain one count per original subject')
        counts = [int(counts[s]) for s in subjects]
    else:
        if source_length % len(subjects):
            raise ValueError('equal-trial source data is not divisible by source subjects')
        counts = [source_length // len(subjects)] * len(subjects)
    domain = np.concatenate([np.full(n, s, dtype=np.int64) for s, n in zip(subjects, counts)])
    if len(domain) != source_length:
        raise ValueError('source domain IDs do not match source trial count')
    return domain


def temporal_windows(length, count):
    return np.array_split(np.arange(length), count)


def trial_log_rms_style(trial, windows, eps):
    trial = np.asarray(trial, dtype=np.float64)
    return np.stack([np.log(np.sqrt(np.mean(trial[:, w] ** 2, axis=1) + eps) + eps)
                     for w in windows], axis=1)


def interpolate_log_scale(log_scale_window, windows, length):
    centers = np.asarray([np.mean(w) for w in windows], dtype=np.float64)
    time = np.arange(length, dtype=np.float64)
    return np.stack([np.interp(time, centers, row, left=row[0], right=row[-1])
                     for row in log_scale_window], axis=0)


def _debug_dir(args):
    path = Path('vision_dg_debug') / args.data_name / ('S' + str(args.idt))
    path.mkdir(parents=True, exist_ok=True)
    return path


def subject_distribution_perturbation(X, labels, domains, args, save_debug=True):
    base = np.asarray(X).copy()
    work = base.astype(np.float64, copy=False)
    labels, domains = np.asarray(labels).reshape(-1), np.asarray(domains).reshape(-1)
    if not (len(work) == len(labels) == len(domains)):
        raise ValueError('source data, labels, and domains must have equal length')
    windows = temporal_windows(work.shape[-1], args.subject_style_windows)
    styles = np.stack([trial_log_rms_style(trial, windows, args.subject_style_eps)
                       for trial in work])
    subject_ids = np.asarray(sorted(np.unique(domains)), dtype=int)
    prototypes = np.full((len(subject_ids), args.class_num, work.shape[1], len(windows)),
                          np.nan, dtype=np.float64)
    pos = {int(s): i for i, s in enumerate(subject_ids)}
    prototype_rng = np.random.RandomState(41000 + 1009 * args.idt + 97)
    for s in subject_ids:
        for k in range(args.class_num):
            indices = np.where((domains == s) & (labels == k))[0]
            if len(indices):
                shuffled = indices.copy()
                prototype_rng.shuffle(shuffled)
                group_count = 5 if len(indices) >= 5 else 3
                group_means = [np.mean(styles[group], axis=0)
                               for group in np.array_split(shuffled, group_count) if len(group)]
                prototypes[pos[int(s)], k] = np.median(
                    np.stack(group_means), axis=0
                )
    global_style = np.nanmean(prototypes, axis=0)
    deviations = prototypes - global_style[None, ...]
    rng = np.random.RandomState(41000 + 1009 * args.idt)
    output = work.copy(); records = []
    lower, upper = np.log(args.style_scale_min), np.log(args.style_scale_max)
    for i in range(len(work)):
        s, k = int(domains[i]), int(labels[i])
        candidates = [int(d) for d in subject_ids if d != s and
                      np.isfinite(prototypes[pos[int(d)], k]).all()]
        applied = bool(candidates) and rng.rand() < args.subject_perturb_prob
        a = b = -1; beta = np.nan
        if applied:
            if len(candidates) >= 2:
                a, b = map(int, rng.choice(candidates, size=2, replace=False))
            else:
                a = b = candidates[0]
            beta = float(rng.uniform(0.0, 1.0))
            virtual_deviation = beta * deviations[pos[a], k] + (1.0 - beta) * deviations[pos[b], k]
            virtual_style = global_style[k] + virtual_deviation
            log_scale = args.subject_perturb_strength * (virtual_style - prototypes[pos[s], k])
            log_scale = np.clip(log_scale, lower, upper)
            curve = np.exp(interpolate_log_scale(log_scale, windows, work.shape[-1]))
            output[i] = work[i] * curve
        records.append((i, s, k, a, b, beta, int(applied)))
    if not np.isfinite(output).all():
        raise FloatingPointError(args.version + ' produced NaN or Inf')
    if save_debug:
        debug = _debug_dir(args)
        np.save(debug / 'subject_class_style_prototypes.npy', prototypes)
        np.save(debug / 'class_global_style.npy', global_style)
        np.save(debug / 'subject_style_deviations.npy', deviations)
        np.save(debug / 'source_subject_ids.npy', subject_ids)
        with (debug / 'virtual_style_records.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle); writer.writerow(
                ['recipient_trial', 'subject', 'class', 'virtual_source_a',
                 'virtual_source_b', 'beta', 'applied'])
            writer.writerows(records)
    return output.astype(base.dtype, copy=False), records, prototypes, global_style, deviations
