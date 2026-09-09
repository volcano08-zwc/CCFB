'''
=================================================
coding:utf-8
@Time:      2025/5/6 10:58
@File:      DBConformer_LOSO.py
@Author:    Ziwei Wang
@Function: Leave-One-Subject-Out (LOSO) scenario
=================================================
'''
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.network import backbone_net_triview_concat_dbconformer as backbone_net_dbconformer
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar
from utils.utils import fix_random_seed, cal_acc_comb, data_loader
from utils.vision_dg_preprocess import build_source_domain_ids
import gc
import sys
import csv
import math
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def learning_rate_at_step(base_lr, step, max_steps, warmup_steps):
    """Linear warmup followed by cosine decay; ``step`` is one-based."""
    if max_steps <= 0:
        raise ValueError('max_steps must be positive')
    warmup_steps = max(0, min(int(warmup_steps), int(max_steps)))
    step = max(1, min(int(step), int(max_steps)))
    if warmup_steps and step <= warmup_steps:
        return float(base_lr) * step / warmup_steps
    decay_steps = max(max_steps - warmup_steps, 1)
    progress = (step - warmup_steps) / decay_steps
    return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(network, args):
    return optim.AdamW(
        network.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )


def save_epoch_history(history, args, make_plot=False):
    version = getattr(args, 'version', 'V5')
    out_dir = Path('metrics') / args.data_name / version / ('seed' + str(args.SEED))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = 'S' + str(args.idt) + '_epoch_metrics'
    csv_path = out_dir / (stem + '.csv')
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'epoch', 'train_loss', 'train_accuracy', 'test_accuracy'])
        writer.writeheader()
        writer.writerows(history)
    if make_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        epochs = [row['epoch'] for row in history]
        fig, loss_axis = plt.subplots(figsize=(8, 5))
        acc_axis = loss_axis.twinx()
        loss_axis.plot(epochs, [row['train_loss'] for row in history],
                       color='#d95f02', label='Train loss')
        acc_axis.plot(epochs, [row['train_accuracy'] for row in history],
                      color='#1b9e77', label='Train accuracy')
        acc_axis.plot(epochs, [row['test_accuracy'] for row in history],
                      color='#7570b3', label='Test accuracy')
        loss_axis.set(xlabel='Epoch', ylabel='Cross-entropy loss')
        acc_axis.set_ylabel('Accuracy (%)')
        loss_axis.grid(alpha=0.25)
        lines = loss_axis.lines + acc_axis.lines
        loss_axis.legend(lines, [line.get_label() for line in lines], loc='best')
        fig.tight_layout()
        fig.savefig(out_dir / (stem + '.png'), dpi=180)
        plt.close(fig)


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    domain_src = build_source_domain_ids(args, len(X_src))
    assert len(X_src) == len(y_src) == len(domain_src)
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    dset_loaders = data_loader(
        X_src, y_src, X_tar, y_tar, args, domain_src=domain_src
    )
    netF = backbone_net_dbconformer(args)
    if args.data_env != 'local':
        netF = netF.cuda()
    base_network = netF
    optimizer_f = build_optimizer(netF, args)
    if args.class_num == 2:
        class_weight = torch.tensor([1., args.weight], dtype=torch.float32)
        if args.data_env != 'local':
            class_weight = class_weight.cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
        criterion = nn.CrossEntropyLoss()
    max_iter = args.max_epoch * len(dset_loaders["source"])
    interval_iter = max_iter // args.max_epoch
    args.max_iter = max_iter
    warmup_iters = args.warmup_epochs * len(dset_loaders["source"])
    iter_num = 0
    history = []
    epoch_loss_sum = 0.0
    epoch_correct = 0
    epoch_samples = 0
    base_network.train()
    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source = next(iter_source)
        if inputs_source.size(0) == 1:
            continue
        iter_num += 1
        inputs_source = inputs_source.unsqueeze_(3)
        inputs_source = inputs_source.permute(0, 3, 1, 2)
        features_source, outputs_source = base_network(inputs_source)
        classifier_loss = criterion(outputs_source, labels_source)
        current_lr = learning_rate_at_step(
            args.lr, iter_num, max_iter, warmup_iters
        )
        for param_group in optimizer_f.param_groups:
            param_group['lr'] = current_lr
        batch_samples = labels_source.size(0)
        epoch_loss_sum += classifier_loss.item() * batch_samples
        epoch_correct += (outputs_source.argmax(dim=1) == labels_source).sum().item()
        epoch_samples += batch_samples
        optimizer_f.zero_grad()
        classifier_loss.backward()
        optimizer_f.step()
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()
            acc_t_te, _ = cal_acc_comb(dset_loaders["target-online"], base_network, args=args)  # TODO target-online
            epoch = int(iter_num // len(dset_loaders["source"]))
            train_loss = epoch_loss_sum / epoch_samples
            train_acc = 100.0 * epoch_correct / epoch_samples
            history.append({'epoch': epoch, 'train_loss': train_loss,
                            'train_accuracy': train_acc, 'test_accuracy': acc_t_te})
            save_epoch_history(history, args)
            log_str = ('Task: {}, Epoch:{}/{}; Loss = {:.6f}; Train Acc = {:.2f}%; '
                       'Test Acc = {:.2f}%; LR = {:.8f}').format(
                args.task_str, epoch, args.max_epoch, train_loss, train_acc,
                acc_t_te, current_lr)
            args.log.record(log_str)
            epoch_loss_sum = 0.0
            epoch_correct = 0
            epoch_samples = 0
            base_network.train()
    save_epoch_history(history, args, make_plot=True)
    print('Test Acc = {:.2f}%'.format(acc_t_te))
    print('saving model...')
    if not os.path.exists('./runs/' + str(args.data_name) + '/'):
        os.makedirs('./runs/' + str(args.data_name) + '/')
    if args.align:
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '.ckpt')
    else:
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_noEA' + '.ckpt')
    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()
    return acc_t_te


if __name__ == '__main__':
    cpu_num = 8
    torch.set_num_threads(cpu_num)
    data_name_list = ['BNCI2014001']  # BCI Competition IV 2a, left-hand vs right-hand LOSO
    for data_name in data_name_list:
        weight = 1
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 9, 22, 2, 1001, 250, 144
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 14, 15, 2, 2561, 512, 100
        if data_name == 'BNCI2014004': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 9, 3, 2, 1126, 250, 120
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 12, 13, 2, 2561, 512, 200
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 9, 22, 4, 1001, 250, 288
        if data_name == 'MI1-7': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 7, 59, 2, 750, 250, 200
        if data_name == 'MI1': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, dim_e, dim_p = 'MI', 5, 59, 2, 750, 250, 200, 184, 750
        if data_name == 'BNCI2014008': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, weight = 'ERP', 8, 8, 2, 256, 256, 4200, 64,
        if data_name == 'BNCI2015003': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, weight = 'ERP', 10, 8, 2, 206, 256, 2520, 64
        if data_name == 'Zhou2016': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 4, 14, 2, 1251, 250, -1
        if data_name == 'Zhou2016_3': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 4, 14, 3, 1251, 250, -1
        args = argparse.Namespace(trial_num=trial_num,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, paradigm=paradigm, data_name=data_name,
                                  weight=weight)

        args.backbone = 'TriViewConcatAdamWCosineDBConformer'
        args.version = 'TriViewConcatSoftmaxAdamWWarmupCosineV7'
        args.method = args.backbone + '_' + data_name
        # Source-only subject-distribution perturbation inherited from V111.
        args.subject_style_windows = 4
        args.subject_perturb_prob = 0.47
        args.subject_perturb_strength = 0.6
        args.style_scale_min = 0.8
        args.style_scale_max = 1.25
        args.subject_style_eps = 1e-8
        # DBConformer parameters
        args.gate_flag = False
        args.posemb_flag = True
        args.chn_atten_flag = True
        args.branch = 'all'  # [all, temporal]
        args.emb_size = 40
        args.spa_dim = 16
        if data_name == 'BNCI2014001' or data_name == 'BNCI2014004':
            args.transformer_depth_tem = 2
            args.transformer_depth_chn = 2
        else:
            args.transformer_depth_tem = 6
            args.transformer_depth_chn = 6
        if data_name == 'BNCI2015001' or data_name == 'BNCI2014002':
            args.patch_size = 128
        else:
            args.patch_size = 125
        # whether to use EA
        args.align = True
        args.dropoutRate = 0.25
        # learning rate
        args.lr = 0.001
        args.weight_decay = 1e-4
        args.warmup_epochs = 5
        # train batch size
        args.batch_size = 32
        # training epochs
        args.max_epoch = 100
        # GPU device id
        try:
            device_id = str(sys.argv[1])
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
        except:
            args.data_env = 'local'
        total_acc = []
        for s in [1, 2, 3, 4, 5]:
            args.SEED = s
            fix_random_seed(args.SEED)
            torch.backends.cudnn.deterministic = True
            args.data = data_name
            print(args.data)
            print(args.method)
            print(args.SEED)
            print(args)
            args.local_dir = './data/' + str(data_name) + '/'
            args.result_dir = './logs/'
            my_log = LogRecord(args)
            my_log.log_init()
            my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)
            sub_acc_all = np.zeros(N)
            for idt in range(N):
                args.idt = idt
                source_str = 'Except_S' + str(idt)
                target_str = 'S' + str(idt)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                print(info_str)
                my_log.record(info_str)
                args.log = my_log
                sub_acc_all[idt] = train_target(args)
            print('Sub acc: ', np.round(sub_acc_all, 3))
            print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
            total_acc.append(sub_acc_all)
            acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
            acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
            args.log.record("\n==========================================")
            args.log.record(acc_sub_str)
            args.log.record(acc_mean_str)
        args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)
        print(str(total_acc))
        args.log.record(str(total_acc))
        subject_mean = np.round(np.average(total_acc, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_acc)), 5)
        total_std = np.round(np.std(np.average(total_acc, axis=1)), 5)
        print(subject_mean)
        print(args.method)
        print(total_mean)
        print(total_std)
        args.log.record(str(subject_mean))
        args.log.record(str(total_mean))
        args.log.record(str(total_std))
