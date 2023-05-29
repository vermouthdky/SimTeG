import argparse
import copy
import gc
import math
import os
import random
import time
import uuid
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .load_dataset import *
from .model import EMA
from .utils import *


def evaluate(y_pred, y_true, idx, evaluator):
    return evaluator.eval({"y_true": y_true[idx], "y_pred": y_pred[idx]})["acc"]


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def run(args, device):
    dirs = "./"
    checkpt_file = dirs

    with torch.no_grad():
        data = prepare_data(device, args)
    if args.dataset == "ogbn-products":
        (
            graph,
            feats,
            labels,
            in_size,
            num_classes,
            train_node_nums,
            valid_node_nums,
            test_node_nums,
            evaluator,
        ) = data

    for stage, epochs in enumerate(args.stages):
        if stage > 0 and args.use_rlu:
            predict_prob = (
                torch.load(os.path.join(args.output_dir, checkpt_file + "_{}_{}.pt".format((stage - 1), args.method)))
                / args.temp
            )
            predict_prob = predict_prob.softmax(dim=1)

            print(
                "This history model Train ACC is {}".format(
                    evaluator(
                        labels[:train_node_nums], predict_prob[:train_node_nums].argmax(dim=-1, keepdim=True).cpu()
                    )
                )
            )

            print(
                "This history model Valid ACC is {}".format(
                    evaluator(
                        labels[train_node_nums : train_node_nums + valid_node_nums],
                        predict_prob[train_node_nums : train_node_nums + valid_node_nums]
                        .argmax(dim=-1, keepdim=True)
                        .cpu(),
                    )
                )
            )

            print(
                "This history model Test ACC is {}".format(
                    evaluator(
                        labels[train_node_nums + valid_node_nums : train_node_nums + valid_node_nums + test_node_nums],
                        predict_prob[
                            train_node_nums + valid_node_nums : train_node_nums + valid_node_nums + test_node_nums
                        ]
                        .argmax(dim=-1, keepdim=True)
                        .cpu(),
                    )
                )
            )

            confident_nid = torch.arange(len(predict_prob))[predict_prob.max(1)[0] > args.threshold]
            extra_confident_nid = confident_nid[confident_nid >= train_node_nums]
            print(f"Stage: {stage}, confident nodes: {len(extra_confident_nid)}")
            enhance_idx = extra_confident_nid
            if len(extra_confident_nid) > 0:
                enhance_loader = torch.utils.data.DataLoader(
                    enhance_idx,
                    batch_size=int(args.batch_size * len(enhance_idx) / (len(enhance_idx) + train_node_nums)),
                    shuffle=True,
                    drop_last=False,
                )

            if args.consis == True:
                confident_nid_cons = torch.arange(len(predict_prob))[predict_prob.max(1)[0] > args.threshold + 0.145]
                extra_confident_nid_cons = confident_nid_cons[confident_nid_cons >= train_node_nums]
                print(f"Stage: {stage}, confident_cons nodes: {len(extra_confident_nid_cons)}")
                enhance_idx_cons = extra_confident_nid_cons
                if len(extra_confident_nid_cons) > 0:
                    enhance_loader_cons = torch.utils.data.DataLoader(
                        enhance_idx_cons,
                        batch_size=int(args.batch_size * len(enhance_idx_cons) / (len(enhance_idx) + train_node_nums)),
                        shuffle=True,
                        drop_last=False,
                    )

            teacher_probs = torch.zeros(predict_prob.shape[0], predict_prob.shape[1])
            teacher_probs[enhance_idx, :] = predict_prob[enhance_idx, :].cpu()
        else:
            teacher_probs = None

        label_emb = None
        if args.use_rlu:
            label_emb = prepare_label_emb(args, graph, labels, num_classes, train_node_nums, teacher_probs)

        if stage == 0:
            train_loader = torch.utils.data.DataLoader(
                torch.arange(train_node_nums), batch_size=args.batch_size_first, shuffle=True, drop_last=False
            )
        elif stage != 0:
            train_loader = torch.utils.data.DataLoader(
                torch.arange(train_node_nums),
                batch_size=int(args.batch_size * train_node_nums / (len(enhance_idx) + train_node_nums)),
                shuffle=True,
                drop_last=False,
            )

        val_loader = torch.utils.data.DataLoader(
            torch.arange(train_node_nums, train_node_nums + valid_node_nums),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.arange(train_node_nums + valid_node_nums, train_node_nums + valid_node_nums + test_node_nums),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
        all_loader = torch.utils.data.DataLoader(
            torch.arange(train_node_nums + valid_node_nums + test_node_nums),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        if args.method == "R_GAMLP":
            print("not use rlu")
            model = gen_model(args, in_size, num_classes)
        elif args.method == "R_GAMLP_RLU":
            print("GAMLP")
            model = gen_model_rlu(args, in_size, num_classes)
        elif args.method == "SAGN":
            print("SAGN")
            label_in_feats = label_emb.shape[1] if label_emb is not None else n_classes
            model = gen_model_sagn(args, in_size, label_in_feats, num_classes)
        model = model.to(device)

        if args.mean_teacher == True:
            print("use mean_teacher")
            if args.method == "R_GAMLP":
                print("not use rlu")
                teacher_model = gen_model(args, in_size, num_classes)
            elif args.method == "R_GAMLP_RLU":
                print("teacher-GAMLP")
                teacher_model = gen_model_rlu(args, in_size, num_classes)
            elif args.method == "SAGN":
                print("teacher-SAGN")
                label_in_feats = label_emb.shape[1] if label_emb is not None else n_classes
                teacher_model = gen_model_sagn(args, in_size, label_in_feats, num_classes)
            teacher_model = teacher_model.to(device)
            for param in teacher_model.parameters():
                param.detach_()

        if args.ema == True and stage == 0:
            print("use ema")
            ema = EMA(model, args.decay)
            ema.register()
        else:
            ema = None
        print("# Params:", get_n_params(model))

        loss_fcn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_factor)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if (args.giant or args.use_bert_x) and args.method == "SAGN":
            # Start training
            best_epoch = 0
            best_val = 0
            best_val_loss = 1e9
            best_test = 0
            num_epochs = epochs
            count = 0
            for epoch in range(1, num_epochs + 1):
                start = time.time()
                if args.mean_teacher == False:
                    train_sagn(device, model, feats, label_emb, labels, loss_fcn, optimizer, train_loader, args, ema)
                else:
                    if epoch < (args.warm_up + 1):
                        train_sagn(
                            device, model, feats, label_emb, labels, loss_fcn, optimizer, train_loader, args, ema
                        )
                    else:
                        if epoch == (args.warm_up + 1):
                            print("start mean teacher")
                        if (epoch - 1) % args.gap == 0 or epoch == (args.warm_up + 1):
                            preds = gen_output_torch(model, feats, all_loader, device, label_emb, args)
                            predict_prob = preds.softmax(dim=1)

                            threshold = args.top - (args.top - args.down) * epoch / num_epochs

                            confident_nid_cons = torch.arange(len(predict_prob))[predict_prob.max(1)[0] > threshold]
                            extra_confident_nid_cons = confident_nid_cons[confident_nid_cons >= train_node_nums]
                            enhance_idx_cons = extra_confident_nid_cons
                            enhance_loader_cons = torch.utils.data.DataLoader(
                                enhance_idx_cons,
                                batch_size=int(
                                    args.batch_size * len(enhance_idx_cons) / (len(enhance_idx_cons) + train_node_nums)
                                ),
                                shuffle=True,
                                drop_last=False,
                            )
                            train_loader_with_pseudos = torch.utils.data.DataLoader(
                                train_nid,
                                batch_size=int(
                                    args.batch_size * train_node_nums / (len(enhance_idx_cons) + train_node_nums)
                                ),
                                shuffle=True,
                                drop_last=False,
                            )
                        train_mean_teacher_sagn(
                            device,
                            model,
                            teacher_model,
                            feats,
                            label_emb,
                            labels,
                            loss_fcn,
                            optimizer,
                            train_loader_with_pseudos,
                            enhance_loader_cons,
                            args,
                            ema,
                        )
                med = time.time()

                if epoch % args.eval_every == 0:
                    with torch.no_grad():
                        train_nid = torch.arange(train_node_nums)
                        val_nid = torch.arange(train_node_nums, train_node_nums + valid_node_nums)
                        test_nid = torch.arange(
                            train_node_nums + valid_node_nums, train_node_nums + valid_node_nums + test_node_nums
                        )

                        acc = test_sagn(
                            device,
                            model,
                            feats,
                            label_emb,
                            labels,
                            loss_fcn,
                            val_loader,
                            all_loader,
                            evaluator,
                            train_nid,
                            val_nid,
                            test_nid,
                            args,
                            ema,
                        )
                    end = time.time()

                    if acc[1] > best_val:
                        best_epoch = epoch
                        best_val = acc[1]
                        best_test = acc[2]
                        best_val_loss = acc[3]
                        best_model = deepcopy(model)
                        count = 0
                    else:
                        count += args.eval_every
                        if count >= args.patience:
                            break
                    log = "Epoch {}, Time(s): {:.4f} {:.4f}, ".format(epoch, med - start, acc[-1])
                    log += "Best Val loss: {:.4f}, Accs: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best Val: {:.4f}, Best Test: {:.4f}".format(
                        best_val_loss, acc[0], acc[1], acc[2], best_val, best_test
                    )
                    print(log)
            preds = gen_output_torch(best_model, feats, all_loader, device, label_emb, args, ema)
            torch.save(preds, os.path.join(args.output_dir, checkpt_file + f"_{stage}_{args.method}.pt"))
            print("Stage: {}, Best Epoch {}, Val {:.4f}, Test {:.4f}".format(stage, best_epoch, best_val, best_test))
        else:
            # Start training
            best_epoch = 0
            best_val = 0
            best_test = 0
            count = 0
            if args.mean_teacher == True:
                global_step = 0

            for epoch in range(epochs):
                gc.collect()
                start = time.time()

                if args.mean_teacher == False:
                    if stage == 0:
                        loss, acc = train(
                            model,
                            feats,
                            labels,
                            device,
                            loss_fcn,
                            optimizer,
                            train_loader,
                            label_emb,
                            evaluator,
                            args,
                            ema,
                        )
                    elif stage != 0 and args.consis == True:
                        loss, acc = train_rlu_consis(
                            model,
                            train_loader,
                            enhance_loader,
                            optimizer,
                            evaluator,
                            device,
                            feats,
                            labels,
                            label_emb,
                            predict_prob,
                            args,
                            enhance_loader_cons,
                        )
                    elif stage != 0 and args.consis == False:
                        loss, acc = train_rlu(
                            model,
                            train_loader,
                            enhance_loader,
                            optimizer,
                            evaluator,
                            device,
                            feats,
                            labels,
                            label_emb,
                            predict_prob,
                            args.gama,
                            args,
                        )
                else:
                    if epoch < (args.warm_up + 1):
                        if stage == 0:
                            loss, acc = train(
                                model,
                                feats,
                                labels,
                                device,
                                loss_fcn,
                                optimizer,
                                train_loader,
                                label_emb,
                                evaluator,
                                args,
                                ema,
                            )
                    else:
                        if epoch == (args.warm_up + 1):
                            print("start mean teacher")

                        if (epoch - 1) % args.gap == 0 or epoch == (args.warm_up + 1):
                            preds = gen_output_torch(model, feats, all_loader, device, label_emb, args)
                            prob_teacher = preds.softmax(dim=1)

                            threshold = args.top - (args.top - args.down) * epoch / epochs

                            confident_nid = torch.arange(len(prob_teacher))[prob_teacher.max(1)[0] > threshold]
                            extra_confident_nid = confident_nid[confident_nid >= train_node_nums]
                            enhance_idx = extra_confident_nid
                            train_loader = torch.utils.data.DataLoader(
                                torch.arange(train_node_nums),
                                batch_size=int(
                                    args.batch_size * train_node_nums / (len(enhance_idx) + train_node_nums)
                                ),
                                shuffle=True,
                                drop_last=False,
                            )
                            enhance_loader = torch.utils.data.DataLoader(
                                enhance_idx,
                                batch_size=int(
                                    args.batch_size * len(enhance_idx) / (len(enhance_idx) + train_node_nums)
                                ),
                                shuffle=True,
                                drop_last=False,
                            )

                        loss, acc = train_mean_teacher(
                            model,
                            teacher_model,
                            feats,
                            labels,
                            device,
                            loss_fcn,
                            optimizer,
                            train_loader,
                            enhance_loader,
                            label_emb,
                            evaluator,
                            args,
                            global_step,
                            ema,
                        )
                        global_step += 1
                end = time.time()

                log = "Epoch {}, Time(s): {:.4f}, Train loss: {:.4f}, Train acc: {:.4f} ".format(
                    epoch, end - start, loss, acc * 100
                )
                if epoch % args.eval_every == 0 and epoch > args.train_num_epochs[stage]:
                    with torch.no_grad():
                        acc = test(model, feats, labels, device, val_loader, evaluator, label_emb, args, ema)
                    val_end = time.time()

                    log += "\nValidation: Time(s): {:.4f}, ".format(val_end - end)
                    log += "Val {:.4f}, ".format(acc)
                    if acc > best_val:
                        best_epoch = epoch
                        best_val = acc
                        best_model = copy.deepcopy(model)

                        best_test = test(model, feats, labels, device, test_loader, evaluator, label_emb, args, ema)
                        preds = gen_output_torch(model, feats, all_loader, device, label_emb, args, ema)

                        torch.save(preds, os.path.join(args.output_dir, checkpt_file + f"_{stage}_{args.method}.pt"))

                        count = 0
                    else:
                        count = count + args.eval_every
                        if count >= args.patience:
                            break
                    log += "Best Epoch {}, Val {:.4f}, Test {:.4f}".format(best_epoch, best_val, best_test)
                print(log)
            print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(best_epoch, best_val, best_test))
    return best_val, best_test


def main(args):
    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)

    val_accs = []
    test_accs = []
    for i in range(args.num_runs):
        print(f"Run {i} start training")
        set_seed(args.seed + i)
        best_val, best_test = run(args, device)
        # np.save(f"output/{args.dataset}/output_{i}.npy", preds.numpy())
        val_accs.append(best_val)
        test_accs.append(best_test)
        print(f"Current Average val accuracy: {np.mean(val_accs):.4f}, " f"std: {np.std(val_accs):.4f}")
        print(f"Current Average test accuracy: {np.mean(test_accs):.4f}, " f"std: {np.std(test_accs):.4f}")

    print(f"Average val accuracy: {np.mean(val_accs):.4f}, " f"std: {np.std(val_accs):.4f}")
    print(f"Average test accuracy: {np.mean(test_accs):.4f}, " f"std: {np.std(test_accs):.4f}")

    return np.mean(test_accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAMLP/SAGN")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--num_hops", type=int, default=5, help="number of hops")
    parser.add_argument("--label_num_hops", type=int, default=9, help="number of hops for label")
    parser.add_argument("--seed", type=int, default=0, help="the seed used in the training")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout on activation")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=50000)
    parser.add_argument("--batch_size_first", type=int, default=50000)
    parser.add_argument("--n-layers-1", type=int, default=4, help="number of feed-forward layers")
    parser.add_argument("--n-layers-2", type=int, default=4, help="number of feed-forward layers")
    parser.add_argument("--n-layers-3", type=int, default=4, help="number of feed-forward layers")
    parser.add_argument("--num-runs", type=int, default=10, help="number of times to repeat the experiment")
    parser.add_argument("--patience", type=int, default=100, help="early stop of times of the experiment")
    parser.add_argument("--alpha", type=float, default=0.5, help="initial residual parameter for the model")
    parser.add_argument("--temp", type=float, default=1, help="temperature of the output prediction")
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="the threshold for the node to be added into the model"
    )
    parser.add_argument("--input-drop", type=float, default=0, help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.5, help="attention dropout of model")
    parser.add_argument("--label-drop", type=float, default=0.5, help="label feature dropout of model")
    parser.add_argument("--gama", type=float, default=0.5, help="parameter for the KL loss")
    parser.add_argument(
        "--pre-process", action="store_true", default=False, help="whether to process the input features"
    )
    parser.add_argument("--residual", action="store_true", default=False, help="whether to connect the input features")
    parser.add_argument("--act", type=str, default="relu", help="the activation function of the model")
    parser.add_argument("--method", type=str, default="R_GAMLP_RLU", help="the model to use")
    parser.add_argument("--use-emb", type=str, default=True)
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="../cogdl_data")
    parser.add_argument(
        "--use-rlu", action="store_true", default=True, help="whether to use the reliable data distillation"
    )
    parser.add_argument(
        "--train-num-epochs", nargs="+", type=int, default=[100, 100], help="The Train epoch setting for each stage."
    )
    parser.add_argument("--stages", nargs="+", type=int, default=[300, 300], help="The epoch setting for each stage.")
    parser.add_argument(
        "--pre-dropout", action="store_true", default=False, help="whether to process the input features"
    )
    parser.add_argument("--bns", action="store_true", default=False, help="whether to process the input features")
    # EMA
    parser.add_argument("--ema", action="store_true", default=False, help="Whether to use ema in the first stage")
    parser.add_argument("--decay", type=float, default=0.9)

    # consistency loss
    parser.add_argument("--consis", action="store_true", default=False, help="Whether to use consistency loss")
    parser.add_argument("--tem", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=0.1)

    # sagn
    parser.add_argument("--weight-style", type=str, default="attention")
    parser.add_argument("--zero-inits", action="store_true", help="Whether to initialize hop attention vector as zeros")
    parser.add_argument("--position-emb", action="store_true")
    parser.add_argument("--focal", type=str, default="first")
    parser.add_argument("--mlp-layer", type=int, default=2, help="number of MLP layers")
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--label-residual", action="store_true")
    parser.add_argument("--label-mlp-layer", type=int, default=4, help="number of label MLP layers")

    # mean_teacher
    parser.add_argument("--mean_teacher", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--adap", action="store_true")
    parser.add_argument("--sup_lam", type=float, default=1.0)
    parser.add_argument("--kl", action="store_true")
    parser.add_argument("--kl_lam", type=float, default=0.2)
    parser.add_argument("--top", type=float, default=0.9)
    parser.add_argument("--down", type=float, default=0.7)
    parser.add_argument("--warm_up", type=int, default=60)
    parser.add_argument("--gap", type=int, default=20)

    parser.add_argument("--giant", action="store_true")
    parser.add_argument("--use_bert_x", action="store_true")
    parser.add_argument("--lm_model_type", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--label_smoothing_factor", type=float, default=0.0)

    args = parser.parse_args()
    print(args)
    main(args)
