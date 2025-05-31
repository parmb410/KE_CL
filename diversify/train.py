
# Fully integrated train.py with Automated K Estimation, Warm-up Pre-Training, and Curriculum Learning

import time
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
from datautil.getdataloader_single import get_act_dataloader

def automated_k_estimation(features, k_min=2, k_max=10):
    best_k = k_min
    best_score = -1
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(features)
        labels = kmeans.labels_
        score = silhouette_score(features, labels)
        if score > best_score:
            best_k = k
            best_score = score
    print("[INFO] Optimal K determined as {} (Silhouette Score: {:.4f})".format(best_k, best_score))
    return best_k

def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)

    print_environ()
    print(s)

    train_loader, train_loader_noshuffle, valid_loader, target_loader, _, _, _ = get_act_dataloader(args)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')

    print("\n==== Warm-up Pre-Training (1-2 epochs) ====")
    algorithm.train()
    warmup_epochs = 2
    for epoch in range(warmup_epochs):
        print("Warm-up Epoch {}/{}".format(epoch + 1, warmup_epochs))
        for step in range(args.local_epoch):
            for data in train_loader:
                _ = algorithm.update_a(data, opta)
    print("Warm-up Pre-Training done.")

    algorithm.eval()
    feature_list = []
    with torch.no_grad():
        for batch in train_loader:
            data = batch[0].cuda() if isinstance(batch, (list, tuple)) else batch.cuda()
            features = algorithm.featurizer(data)
            feature_list.append(features.cpu().numpy())
    all_features = np.concatenate(feature_list, axis=0)
    optimal_k = automated_k_estimation(all_features)
    args.latent_domain_num = optimal_k
    print("Using automated latent_domain_num (K): {}".format(args.latent_domain_num))

    if args.latent_domain_num < 6:
        args.batch_size = 32 * args.latent_domain_num
    else:
        args.batch_size = 16 * args.latent_domain_num

    sample_losses = torch.zeros(len(train_loader.dataset)).cuda()
    def compute_sample_losses():
        new_losses = torch.zeros(len(train_loader.dataset)).cuda()
        algorithm.eval()
        with torch.no_grad():
            for i, (data, labels, idx) in enumerate(train_loader):
                data, labels = data.cuda(), labels.cuda()
                outputs = algorithm.clf(data)
                loss = torch.nn.functional.cross_entropy(outputs, labels, reduction='none')
                for j, sample_id in enumerate(idx):
                    new_losses[sample_id] = loss[j]
        return new_losses

    best_valid_acc, target_acc = 0, 0

    for round in range(args.max_epoch):
        print("\n========ROUND {}========".format(round))
        new_losses = compute_sample_losses()
        sample_losses = 0.9 * sample_losses + 0.1 * new_losses
        sorted_indices = torch.argsort(sample_losses)
        curriculum_size = int((round + 1) / args.max_epoch * len(sorted_indices))
        curriculum_subset = torch.utils.data.Subset(train_loader.dataset, sorted_indices[:curriculum_size])
        train_loader_curriculum = torch.utils.data.DataLoader(curriculum_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        print("[CL] Curriculum size: {}/{}".format(curriculum_size, len(train_loader.dataset)))

        algorithm.train()
        print('====Feature update====')
        loss_list = ['class']
        print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)
        for step in range(args.local_epoch):
            for data in train_loader_curriculum:
                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step]+[loss_result_dict[item] for item in loss_list], colwidth=15)

        print('====Latent domain characterization====')
        loss_list = ['total', 'dis', 'ent']
        print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)
        for step in range(args.local_epoch):
            for data in train_loader_curriculum:
                loss_result_dict = algorithm.update_d(data, optd)
            print_row([step]+[loss_result_dict[item] for item in loss_list], colwidth=15)

        algorithm.set_dlabel(train_loader_curriculum)
        print('====Domain-invariant feature learning====')
        loss_list = alg_loss_dict(args)
        eval_dict = train_valid_target_eval_names(args)
        print_key = ['epoch']
        print_key.extend([item+'_loss' for item in loss_list])
        print_key.extend([item+'_acc' for item in eval_dict.keys()])
        print_key.append('total_cost_time')
        print_row(print_key, colwidth=15)
        sss = time.time()
        for step in range(args.local_epoch):
            for data in train_loader_curriculum:
                step_vals = algorithm.update(data, opt)
            results = {'epoch': step}
            results['train_acc'] = modelopera.accuracy(algorithm, train_loader_noshuffle, None)
            acc = modelopera.accuracy(algorithm, valid_loader, None)
            results['valid_acc'] = acc
            acc = modelopera.accuracy(algorithm, target_loader, None)
            results['target_acc'] = acc
            for key in loss_list:
                results[key+'_loss'] = step_vals[key]
            if results['valid_acc'] > best_valid_acc:
                best_valid_acc = results['valid_acc']
                target_acc = results['target_acc']
            results['total_cost_time'] = time.time() - sss
            print_row([results[key] for key in print_key], colwidth=15)

    print("Target acc: {:.4f}".format(target_acc))

if __name__ == '__main__':
    args = get_args()
    main(args)
