import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from load_dataset import load_dataset
from ogb.nodeproppred import Evaluator

from cogdl.models.nn.correct_smooth import CorrectSmooth

def evaluate(y_pred, y_true, idx, evaluator):
    return evaluator.eval({
        'y_true': y_true[idx],
        'y_pred': y_pred[idx]
    })['acc']


def main(args):
    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)

    with torch.no_grad():
        data = load_dataset(args.dataset, device, args, return_nid=True)
    if args.dataset == 'ogbn-products':
        graph, train_nid, val_nid, test_nid, evaluator = data
    train_node_nums = len(train_nid)
    valid_node_nums = len(val_nid)
    test_node_nums = len(test_nid)

    graph.train_mask = torch.full((graph.num_nodes,), False, dtype=torch.bool)
    graph.val_mask = torch.full((graph.num_nodes,), False, dtype=torch.bool)
    graph.test_mask = torch.full((graph.num_nodes,), False, dtype=torch.bool)

    graph.train_mask[train_nid] = True
    graph.val_mask[val_nid] = True
    graph.test_mask[test_nid] = True

    graph.to(device)
    labels = graph.y.to(torch.long)

    print("--------------------load_data---------------------")
    
    file_name = f'./output/{args.dataset}/'+ args.file_name +'.pt'

    predict_prob = torch.load(file_name)
    predict_prob = predict_prob.softmax(dim=1).to(device)

    print("This history model Train ACC is {}".format(
        evaluator(labels[:train_node_nums], predict_prob[:train_node_nums].argmax(dim=-1, keepdim=True).cpu())))

    print("This history model Valid ACC is {}".format(
        evaluator(labels[train_node_nums:train_node_nums + valid_node_nums],
                  predict_prob[train_node_nums:train_node_nums + valid_node_nums].argmax(dim=-1, keepdim=True).cpu())))

    print("This history model Test ACC is {}".format(
        evaluator(labels[train_node_nums + valid_node_nums:train_node_nums + valid_node_nums + test_node_nums],
                  predict_prob[
                  train_node_nums + valid_node_nums:train_node_nums + valid_node_nums + test_node_nums].argmax(dim=-1,
                                                                                                               keepdim=True).cpu())))
    labels = labels.reshape(-1, 1)

    print('---------- Correct & Smoothing ----------')

    if args.search : 
        cs_search = CorrectSmoothSearch(graph, predict_prob, args)
        cs_search.run(args.n_trials)
    else :
        cs = CorrectSmooth(correct_alpha=args.correction_alpha,
                                smooth_alpha=args.smoothing_alpha,
                                num_correct_prop=args.num_correction_layers,
                                num_smooth_prop=args.num_smoothing_layers,
                                autoscale=args.autoscale,
                                correct_norm="sym",
                                smooth_norm="sym",
                                scale=args.scale,)
                                
        y_soft = cs(graph, predict_prob)
        y_pred = y_soft.argmax(dim=-1, keepdim=True)

        labels = graph.y.reshape(-1, 1)

        evaluator = Evaluator(name=args.dataset)
        valid_acc = evaluate(y_pred, labels, graph.val_nid, evaluator)
        test_acc = evaluate(y_pred, labels, graph.test_nid, evaluator)
        print(f'Valid acc: {valid_acc:.4f} | Test acc: {test_acc:.4f}')

class CorrectSmoothSearch(object):
    def __init__(self, graph, predict_prob, args):
        self.graph = graph
        self.predict_prob = predict_prob
        self.dataset = args.dataset
        self.best_val_acc = 0
        self.best_test_acc = 0
        self.cs = CorrectSmooth(correct_alpha=0.1,
                                smooth_alpha=0.1,
                                num_correct_prop=args.num_correction_layers,
                                num_smooth_prop=args.num_smoothing_layers,
                                autoscale=args.autoscale,
                                correct_norm="sym",
                                smooth_norm="sym",
                                scale=args.scale,)

    def objective(self, trial):
        correction_alpha = trial.suggest_float('correction-alpha', 0.4, 1.0)
        smoothing_alpha = trial.suggest_float('smoothing-alpha', 0.4, 1.0)
        self.cs.op_dict["correct_alpha"] = correction_alpha
        self.cs.op_dict["smooth_alpha"] = smoothing_alpha

        y_soft = self.cs(self.graph, self.predict_prob)
        y_pred = y_soft.argmax(dim=-1, keepdim=True)

        labels = self.graph.y.reshape(-1, 1)

        evaluator = Evaluator(name=self.dataset)
        valid_acc = evaluate(y_pred, labels, self.graph.val_nid, evaluator)
        test_acc = evaluate(y_pred, labels, self.graph.test_nid, evaluator)
        print(f'Valid acc: {valid_acc:.4f} | Test acc: {test_acc:.4f}')

        if valid_acc > self.best_val_acc:
            self.best_val_acc = valid_acc
            self.best_test_acc = test_acc
            self.best_trial = trial

        return valid_acc

    def run(self, n_trials):
        study = optuna.create_study(direction='maximize')

        study.optimize(
            lambda trial: self.objective(trial), n_trials=n_trials, n_jobs=1)
        print('Number of finished trials: ', len(study.trials))
        print('Best trial:')
        trial = self.best_trial
        print(f'  Best Valid acc: {self.best_val_acc:.4f} | Best Test acc: {self.best_test_acc:.4f}')
        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hyperparameters for Correct&Smooth postprocessing")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--root", type=str, default='')
    parser.add_argument("--file_name", type=str, default='',
                        help="pt filename")

    # C & S
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument('--num-correction-layers', type=int, default=50)
    parser.add_argument('--correction-alpha', type=float, default=0)
    parser.add_argument('--num-smoothing-layers', type=int, default=50)
    parser.add_argument('--smoothing-alpha', type=float, default=0)
    parser.add_argument("--autoscale", action="store_true",
                        help="Whether to use autoscale in Correction operation")
    parser.add_argument('--scale', type=float, default=20.)
    args = parser.parse_args()
    print(args)
    main(args)
