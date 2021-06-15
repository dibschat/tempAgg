# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from dataset_recognition import SequenceDataset
from os.path import join
import torch
from torch.utils.data import DataLoader
from utils import ValueMeter, topk_accuracy, topk_accuracy_save_validation_pred, topk_recall
from utils import get_marginal_indexes, marginalize, softmax, predictions_to_json
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from network import Network
from torch.optim import lr_scheduler
from torch import nn
import copy
import pickle as pkl

device = 'cuda' if torch.cuda.is_available() else 'cpu'

COMP_PATH = 'tempAgg_ant_rec/'

pd.options.display.float_format = '{:05.2f}'.format

parser = ArgumentParser(description="Training for Action Recognition")
parser.add_argument('--mode', type=str, default='train', choices=['train', 'validate', 'test', 'validate_json'],
                    help="Whether to perform training, validation or test. If test/validate_json is selected, "
                         "--json_directory must be used to provide a directory in which to save the generated jsons.")
parser.add_argument('--path_to_data', type=str, default=COMP_PATH + 'DATA_EPIC_ALL/',
                    help="Path to the data folder, containing all LMDB datasets")
parser.add_argument('--path_to_models', type=str, default=COMP_PATH + '/models_recognition/',
                    help="Path to the directory where to save all models")

parser.add_argument('--json_directory', type=str, default=COMP_PATH + '/models_recognition/',
                    help='Directory in which to save the generated jsons.')
parser.add_argument('--task', type=str, default='action_recognition',
                    choices=['action_anticipation', 'action_recognition'],
                    help='Task to tackle: anticipation or recognition')

parser.add_argument('--img_tmpl', type=str, default='frame_{:010d}.jpg',
                    help='Template to use to load the representation of a given frame')
parser.add_argument('--resume', action='store_true', help='Whether to resume suspended training')
parser.add_argument('--best_model', type=str, default='best', choices=['best', 'last'], help='')

parser.add_argument('--modality', type=str, default='obj', choices=['rgb', 'flow', 'obj', 'roi', 'late_fusion'],
                    help="Modality. rgb/flow/obj/roi represent single branches or late fusion of all.")
parser.add_argument('--weight_rgb', type=float, default=0.5, help='')
parser.add_argument('--weight_flow', type=float, default=0.5, help='')
parser.add_argument('--weight_obj', type=float, default=0.5, help='')
parser.add_argument('--weight_roi', type=float, default=0.5, help='')

parser.add_argument('--num_workers', type=int, default=0, help="Number of parallel thread to fetch the data")
parser.add_argument('--display_every', type=int, default=10, help="Display every n iterations")

parser.add_argument('--schedule_on', type=int, default=1, help='')
parser.add_argument('--schedule_epoch', type=int, default=10, help='')

parser.add_argument('--num_class', type=int, default=2513, help='Number of classes')
parser.add_argument('--verb_class', type=int, default=125, help='')
parser.add_argument('--noun_class', type=int, default=352, help='')
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
parser.add_argument('--latent_dim', type=int, default=512, help='')
parser.add_argument('--linear_dim', type=int, default=512, help='')
parser.add_argument('--dropout_rate', type=float, default=0.3, help='')
parser.add_argument('--scale_factor', type=float, default=-.5, help='')
parser.add_argument('--scale', type=bool, default=True, help='')
parser.add_argument('--batch_size', type=int, default=10, help="Batch Size")
parser.add_argument('--epochs', type=int, default=25, help="Training epochs")
parser.add_argument('--video_feat_dim', type=int, default=352, choices=[352, 1024], help='')
parser.add_argument('--past_attention', type=bool, default=True, help='')

# Spanning snippets
parser.add_argument('--spanning_sec', type=float, default=6.0, help='')
parser.add_argument('--span_dim1', type=int, default=5, help='')
parser.add_argument('--span_dim2', type=int, default=3, help='')
parser.add_argument('--span_dim3', type=int, default=2, help='')

# Recent snippets
parser.add_argument('--recent_dim', type=int, default=5, help='')
parser.add_argument('--recent_sec1', type=float, default=0.0, help='')
parser.add_argument('--recent_sec2', type=float, default=1.0, help='')
parser.add_argument('--recent_sec3', type=float, default=2.0, help='')
parser.add_argument('--recent_sec4', type=float, default=3.0, help='')

# Adding verb and noun loss
parser.add_argument('--verb_noun_scores', type=bool, default=True, help='')
parser.add_argument('--add_verb_loss', action='store_true', default=True, help='Whether to train with verb loss or not')
parser.add_argument('--add_noun_loss', action='store_true', default=True, help='Whether to train with verb loss or not')
parser.add_argument('--verb_loss_weight', type=float, default=1.0, help='')
parser.add_argument('--noun_loss_weight', type=float, default=1.0, help='')
parser.add_argument('--ek100', action='store_true', help="Whether to use EPIC-KITCHENS-100")
parser.add_argument('--trainval', type=bool, default=False, help='Whether to train on train+val or only train')

parser.add_argument('--topK', type=int, default=1, help='')

# Debugging True 
parser.add_argument('--debug_on', type=bool, default=False, help='')

args = parser.parse_args()


def make_model_name(arg_save):
    save_name = "arec_mod_{}_span_{}_s1_{}_s2_{}_s3_{}_recent_{}_r1_{}_r2_{}_r3_{}_r4_{}_bs_{}_drop_{}_lr_{}_dimLa_{}_" \
                "dimLi_{}_epoc_{}".format(arg_save.modality, arg_save.spanning_sec, arg_save.span_dim1,
                                          arg_save.span_dim2, arg_save.span_dim3, arg_save.recent_dim,
                                          arg_save.recent_sec1, arg_save.recent_sec2, arg_save.recent_sec3,
                                          arg_save.recent_sec4, arg_save.batch_size, arg_save.dropout_rate, arg_save.lr,
                                          arg_save.latent_dim, arg_save.linear_dim, arg_save.epochs)
    if arg_save.add_verb_loss:
        save_name = save_name + '_vb'
    if arg_save.add_noun_loss:
        save_name = save_name + '_nn'
    return save_name


def save_model(model, epoch, perf, best_perf, is_best=False):
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                'perf': perf, 'best_perf': best_perf}, join(args.path_to_models, exp_name + '.pth.tar'))
    if is_best:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'perf': perf, 'best_perf': best_perf}, join(
            args.path_to_models, exp_name + '_best.pth.tar'))


def get_validation_ids():
    unseen_participants_ids = pd.read_csv(join(args.path_to_data, 'validation_unseen_participants_ids.csv'), names=['id'], squeeze=True)
    tail_verbs_ids = pd.read_csv(join(args.path_to_data, 'validation_tail_verbs_ids.csv'), names=['id'], squeeze=True)
    tail_nouns_ids = pd.read_csv(join(args.path_to_data, 'validation_tail_nouns_ids.csv'), names=['id'], squeeze=True)
    tail_actions_ids = pd.read_csv(join(args.path_to_data, 'validation_tail_actions_ids.csv'), names=['id'], squeeze=True)

    return unseen_participants_ids, tail_verbs_ids, tail_nouns_ids, tail_actions_ids


def get_many_shot():
    """Get many shot verbs, nouns and actions for class-aware metrics (Mean Top-5 Recall)"""
    # read the list of many shot verbs
    many_shot_verbs = pd.read_csv(join(args.path_to_data, 'EPIC_many_shot_verbs.csv'))['verb_class'].values
    # read the list of many shot nouns
    many_shot_nouns = pd.read_csv(
        join(args.path_to_data, 'EPIC_many_shot_nouns.csv'))['noun_class'].values

    # read the list of actions
    actions = pd.read_csv(join(args.path_to_data, 'actions.csv'))
    # map actions to (verb, noun) pairs
    a_to_vn = {a[1]['id']: tuple(a[1][['verb', 'noun']].values)
               for a in actions.iterrows()}

    # create the list of many shot actions
    # an action is "many shot" if at least one
    # between the related verb and noun are many shot
    many_shot_actions = []
    for a, (v, n) in a_to_vn.items():
        if v in many_shot_verbs or n in many_shot_nouns:
            many_shot_actions.append(a)

    return many_shot_verbs, many_shot_nouns, many_shot_actions


def get_scores(model, loader, challenge=False):
    model.eval()
    predictions_act = []
    predictions_noun = []
    predictions_verb = []
    labels = []
    ids = []
    with torch.set_grad_enabled(False):
        for batch in tqdm(loader, 'Evaluating...', len(loader)):
            x_spanning = batch['spanning_features']
            x_recent = batch['recent_features']
            if type(x_spanning) == list:
                x_spanning = [xx.to(device) for xx in x_spanning]
                x_recent = [xx.to(device) for xx in x_recent]
            else:
                x_spanning = x_spanning.to(device)
                x_recent = x_recent.to(device)

            y_label = batch['label'].numpy()
            ids.append(batch['id'])

            pred_act1, pred_act2, pred_act3, pred_act4, pred_verb1, pred_verb2, pred_verb3, pred_verb4, \
            pred_noun1, pred_noun2, pred_noun3, pred_noun4 = model(x_spanning, x_recent)

            pred_ensemble_act = pred_act1.detach() + pred_act2.detach() + pred_act3.detach() + pred_act4.detach()
            pred_ensemble_act = pred_ensemble_act.cpu().numpy()
            pred_ensemble_verb = pred_verb1.detach() + pred_verb2.detach() + pred_verb3.detach() + pred_verb4.detach()
            pred_ensemble_verb = pred_ensemble_verb.cpu().numpy()
            pred_ensemble_noun = pred_noun1.detach() + pred_noun2.detach() + pred_noun3.detach() + pred_noun4.detach()
            pred_ensemble_noun = pred_ensemble_noun.cpu().numpy()

            predictions_act.append(pred_ensemble_act)
            predictions_verb.append(pred_ensemble_verb)
            predictions_noun.append(pred_ensemble_noun)
            labels.append(y_label)

    action_scores = np.concatenate(predictions_act)
    labels = np.concatenate(labels)
    ids = np.concatenate(ids)

    if args.verb_noun_scores:  # use the verb and noun scores
        verb_scores = np.concatenate(predictions_verb)
        noun_scores = np.concatenate(predictions_noun)
    else:  # marginalize the action scores to get the noun and verb scores
        actions = pd.read_csv(join(args.path_to_data, 'actions.csv'), index_col='id')
        vi = get_marginal_indexes(actions, 'verb')
        ni = get_marginal_indexes(actions, 'noun')
        action_prob = softmax(action_scores.reshape(-1, action_scores.shape[-1]))
        verb_scores = marginalize(action_prob, vi)  # .reshape( action_scores.shape[0], action_scores.shape[1], -1)
        noun_scores = marginalize(action_prob, ni)  # .reshape( action_scores.shape[0], action_scores.shape[1], -1)

    if labels.max() > 0 and not challenge:
        return verb_scores, noun_scores, action_scores, labels[:, 0], labels[:, 1], labels[:, 2], ids
    else:
        return verb_scores, noun_scores, action_scores, ids


def get_scores_late_fusion(models, loaders, challenge=False):
    verb_scores = []
    noun_scores = []
    action_scores = []
    outputs = []
    for model, loader in zip(models, loaders):
        outputs = get_scores(model, loader, challenge)
        verb_scores.append(outputs[0])
        noun_scores.append(outputs[1])
        action_scores.append(outputs[2])

    verb_scores[0] = verb_scores[0] * args.weight_rgb
    verb_scores[1] = verb_scores[1] * args.weight_flow
    verb_scores[2] = verb_scores[2] * args.weight_obj
    verb_scores[3] = verb_scores[3] * args.weight_roi

    noun_scores[0] = noun_scores[0] * args.weight_rgb
    noun_scores[1] = noun_scores[1] * args.weight_flow
    noun_scores[2] = noun_scores[2] * args.weight_obj
    noun_scores[3] = noun_scores[3] * args.weight_roi

    action_scores[0] = action_scores[0] * args.weight_rgb
    action_scores[1] = action_scores[1] * args.weight_flow
    action_scores[2] = action_scores[2] * args.weight_obj
    action_scores[3] = action_scores[3] * args.weight_roi

    verb_scores = sum(verb_scores)
    noun_scores = sum(noun_scores)
    action_scores = sum(action_scores)

    #return [verb_scores, noun_scores, action_scores] + list(outputs[3:])
    return [verb_scores, noun_scores, action_scores] + list(outputs[3:])


def log(mode, epoch, total_loss_meter, ensemble_accuracy_meter,
        action_loss_meter, verb_loss_meter, noun_loss_meter,
        accuracy_action1_meter, accuracy_action2_meter, accuracy_action3_meter, accuracy_action4_meter,
        best_perf=None, green=False):
    if green:
        print('\033[92m', end="")
    print(
        "[{}] Epoch: {:.2f}. ".format(mode, epoch),
        "Total Loss: {:.2f}. ".format(total_loss_meter.value()),
        "Act. Loss: {:.2f}. ".format(action_loss_meter.value()),
        "Verb Loss: {:.2f}. ".format(verb_loss_meter.value()),
        "Noun Loss: {:.2f}. ".format(noun_loss_meter.value()),
        "Acc. Act1: {:.2f}% ".format(accuracy_action1_meter.value()),
        "Acc. Act2: {:.2f}% ".format(accuracy_action2_meter.value()),
        "Acc. Act3: {:.2f}% ".format(accuracy_action3_meter.value()),
        "Acc. Act4: {:.2f}% ".format(accuracy_action4_meter.value()),
        "Ensemble Acc.: {:.2f}% ".format(ensemble_accuracy_meter.value()),
        end="")

    if best_perf:
        print("[best: {:.2f}]%".format(best_perf), end="")

    print('\033[0m')


def train_validation(model, loaders, optimizer, epochs, start_epoch, start_best_perf, schedule_on):
    """Training/Validation code"""

    best_perf = start_best_perf  # to keep track of the best performing epoch

    loss_act_TAB1 = nn.CrossEntropyLoss()
    loss_act_TAB2 = nn.CrossEntropyLoss()
    loss_act_TAB3 = nn.CrossEntropyLoss()
    loss_act_TAB4 = nn.CrossEntropyLoss()
    if args.add_verb_loss:
        print('Add verb losses')
        loss_verb_TAB1 = nn.CrossEntropyLoss()
        loss_verb_TAB2 = nn.CrossEntropyLoss()
        loss_verb_TAB3 = nn.CrossEntropyLoss()
        loss_verb_TAB4 = nn.CrossEntropyLoss()
    if args.add_noun_loss:
        print('Add noun losses')
        loss_noun_TAB1 = nn.CrossEntropyLoss()
        loss_noun_TAB2 = nn.CrossEntropyLoss()
        loss_noun_TAB3 = nn.CrossEntropyLoss()
        loss_noun_TAB4 = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, epochs):
        if schedule_on is not None:
            schedule_on.step()

        # define training and validation meters
        total_loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        action_loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        verb_loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        noun_loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}

        ensemble_accuracy_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_action1_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_action2_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_action3_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_action4_meter = {'training': ValueMeter(), 'validation': ValueMeter()}

        for mode in ['training', 'validation']:

            # enable gradients only if training
            with torch.set_grad_enabled(mode == 'training'):
                if mode == 'training':
                    model.train()
                else:
                    model.eval()

                for i, batch in enumerate(loaders[mode]):
                    x_spanning = batch['spanning_features']
                    x_recent = batch['recent_features']
                    if type(x_spanning) == list:
                        x_spanning = [xx.to(device) for xx in x_spanning]
                        x_recent = [xx.to(device) for xx in x_recent]
                    else:
                        x_spanning = x_spanning.to(device)
                        x_recent = x_recent.to(device)

                    y_label = batch['label'].to(device)
                    bs = y_label.shape[0]  # batch size

                    pred_act1, pred_act2, pred_act3, pred_act4, pred_verb1, pred_verb2, pred_verb3, pred_verb4, \
                    pred_noun1, pred_noun2, pred_noun3, pred_noun4 = model(x_spanning, x_recent)

                    loss = loss_act_TAB1(pred_act1, y_label[:, 2]) + \
                           loss_act_TAB2(pred_act2, y_label[:, 2]) + \
                           loss_act_TAB3(pred_act3, y_label[:, 2]) + \
                           loss_act_TAB4(pred_act4, y_label[:, 2])
                    action_loss_meter[mode].add(loss.item(), bs)

                    if args.add_verb_loss:
                        verb_loss = loss_verb_TAB1(pred_verb1, y_label[:, 0]) + \
                                    loss_verb_TAB2(pred_verb2, y_label[:, 0]) + \
                                    loss_verb_TAB3(pred_verb3, y_label[:, 0]) + \
                                    loss_verb_TAB4(pred_verb4, y_label[:, 0])
                        verb_loss_meter[mode].add(verb_loss.item(), bs)
                        loss = loss + args.verb_loss_weight * verb_loss
                    else:
                        verb_loss_meter[mode].add(-1, bs)

                    if args.add_noun_loss:
                        noun_loss = loss_noun_TAB1(pred_noun1, y_label[:, 1]) + \
                                    loss_noun_TAB2(pred_noun2, y_label[:, 1]) + \
                                    loss_noun_TAB3(pred_noun3, y_label[:, 1]) + \
                                    loss_noun_TAB4(pred_noun4, y_label[:, 1])
                        noun_loss_meter[mode].add(noun_loss.item(), bs)
                        loss = loss + args.noun_loss_weight * noun_loss
                    else:
                        noun_loss_meter[mode].add(-1, bs)

                    label_curr = y_label[:, 2].detach().cpu().numpy()
                    acc_future1 = topk_accuracy(pred_act1.detach().cpu().numpy(), label_curr, (args.topK,))[0] * 100
                    acc_future2 = topk_accuracy(pred_act2.detach().cpu().numpy(), label_curr, (args.topK,))[0] * 100
                    acc_future3 = topk_accuracy(pred_act3.detach().cpu().numpy(), label_curr, (args.topK,))[0] * 100
                    acc_future4 = topk_accuracy(pred_act4.detach().cpu().numpy(), label_curr, (args.topK,))[0] * 100
                    accuracy_action1_meter[mode].add(acc_future1, bs)
                    accuracy_action2_meter[mode].add(acc_future2, bs)
                    accuracy_action3_meter[mode].add(acc_future3, bs)
                    accuracy_action4_meter[mode].add(acc_future4, bs)

                    pred_ensemble = pred_act1.detach() + pred_act2.detach() + pred_act3.detach() + pred_act4.detach()
                    pred_ensemble = pred_ensemble.cpu().numpy()
                    acc_ensemble = topk_accuracy(pred_ensemble, label_curr, (args.topK,))[0] * 100

                    # store the values in the meters to keep incremental averages
                    total_loss_meter[mode].add(loss.item(), bs)
                    ensemble_accuracy_meter[mode].add(acc_ensemble, bs)

                    # if in training mode
                    if mode == 'training':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # log training during loop - avoid logging the very first batch. It can be biased.
                    if mode == 'training' and i != 0 and i % args.display_every == 0:
                        epoch_curr = epoch + i / len(loaders[mode])  # compute decimal epoch for logging
                        log(mode, epoch_curr, total_loss_meter[mode], ensemble_accuracy_meter[mode],
                            action_loss_meter[mode], verb_loss_meter[mode], noun_loss_meter[mode],
                            accuracy_action1_meter[mode], accuracy_action2_meter[mode],
                            accuracy_action3_meter[mode], accuracy_action4_meter[mode])

                # log at the end of each epoch
                log(mode, epoch + 1, total_loss_meter[mode], ensemble_accuracy_meter[mode],
                    action_loss_meter[mode], verb_loss_meter[mode], noun_loss_meter[mode],
                    accuracy_action1_meter[mode], accuracy_action2_meter[mode],
                    accuracy_action3_meter[mode], accuracy_action4_meter[mode],
                    max(ensemble_accuracy_meter[mode].value(), best_perf) if mode == 'validation' else None, green=True)

        if best_perf < ensemble_accuracy_meter['validation'].value():
            best_perf = ensemble_accuracy_meter['validation'].value()
            is_best = True
        else:
            is_best = False
        with open(args.path_to_models + '/' + exp_name + '.txt', 'a') as f:
            f.write("%d - %0.2f\n" % (epoch + 1, ensemble_accuracy_meter['validation'].value()))

        # save checkpoint at the end of each train/val epoch
        save_model(model, epoch + 1, ensemble_accuracy_meter['validation'].value(), best_perf, is_best=is_best)

    with open(args.path_to_models + '/' + exp_name + '.txt', 'a') as f:
        f.write("%d - %0.2f\n" % (epochs + 1, best_perf))


def load_checkpoint(model):
    model_add = '.pth.tar'
    if args.best_model == 'best':
        print('args.best_model == True')
        model_add = '_best.pth.tar'

    chk = torch.load(join(args.path_to_models, exp_name + model_add))
    epoch = chk['epoch']
    best_perf = chk['best_perf']
    perf = chk['perf']
    model.load_state_dict(chk['state_dict'])
    return epoch, perf, best_perf


def get_loader(mode, override_modality=None):
    if override_modality:
        path_to_lmdb = join(args.path_to_data, override_modality)
    else:
        path_to_lmdb = join(args.path_to_data, args.modality)

    if args.trainval:
        csv_file = 'trainval'
    else:
        csv_file = mode

    kargs = {
        'path_to_lmdb': path_to_lmdb,
        'path_to_csv': join(args.path_to_data, "{}.csv".format(csv_file)),
        'label_type': ['verb', 'noun', 'action'],
        'img_tmpl': args.img_tmpl,
        'challenge': 'test' in mode,
        'args': args
    }
    _set = SequenceDataset(**kargs)
    return DataLoader(_set, batch_size=args.batch_size, num_workers=args.num_workers,
                      pin_memory=True, shuffle=mode == 'training')


def get_model():
    if not args.modality == 'late_fusion':
        return Network(args)
    elif args.modality == 'late_fusion':
        obj_model = Network(args)
        rgb_model = Network(args_rgb)
        flow_model = Network(args_flow)
        roi_model = Network(args_roi)

        model_add = '.pth.tar'
        if args.best_model == 'best':
            print('args.best_model == True')
            model_add = '_best.pth.tar'
        

        checkpoint_rgb = torch.load(join(args.path_to_models, exp_rgb.replace(f'{args.modality}', 'rgb') + model_add))
        checkpoint_flow = torch.load(join(args.path_to_models, exp_flow.replace(f'{args.modality}', 'flow') + model_add))
        checkpoint_obj = torch.load(join(args.path_to_models, exp_name.replace(f'{args.modality}', 'obj') + model_add))
        checkpoint_roi = torch.load(join(args.path_to_models, exp_roi.replace(f'{args.modality}', 'roi') + model_add))

        print(f"Loaded checkpoint for model rgb. Epoch: {checkpoint_rgb['epoch']}. Perf: {checkpoint_rgb['perf']:.2f}.")
        print(f"Loaded checkpoint for model flow. Epoch: {checkpoint_flow['epoch']}. Perf: {checkpoint_flow['perf']:.2f}.")
        print(f"Loaded checkpoint for model obj. Epoch: {checkpoint_obj['epoch']}. Perf: {checkpoint_obj['perf']:.2f}.")
        print(f"Loaded checkpoint for model roi. Epoch: {checkpoint_roi['epoch']}. Perf: {checkpoint_roi['perf']:.2f}.")

        rgb_model.load_state_dict(checkpoint_rgb['state_dict'])
        flow_model.load_state_dict(checkpoint_flow['state_dict'])
        obj_model.load_state_dict(checkpoint_obj['state_dict'])
        roi_model.load_state_dict(checkpoint_roi['state_dict'])
        
        return [rgb_model, flow_model, obj_model, roi_model]


def main():
    model = get_model()
    if type(model) == list:
        model = [m.to(device) for m in model]
    else:
        model.to(device)

    if args.mode == 'train':
        loaders = {m: get_loader(m) for m in ['training', 'validation']}

        if args.resume:
            start_epoch, _, start_best_perf = load_checkpoint(model)
        else:
            start_epoch = 0
            start_best_perf = 0

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        schedule_on = None
        if args.schedule_on:
            schedule_on = lr_scheduler.StepLR(optimizer, args.schedule_epoch, gamma=0.1, last_epoch=-1)

        train_validation(model, loaders, optimizer, args.epochs, start_epoch, start_best_perf, schedule_on)

    elif args.mode == 'validate':
        if args.modality == 'late_fusion':
            loaders = [get_loader('validation', 'rgb'),
                       get_loader('validation', 'flow'),
                       get_loader('validation', 'obj'), 
                       get_loader('validation', 'roi')]
            verb_scores, noun_scores, action_scores, verb_labels, noun_labels, action_labels, ids = get_scores_late_fusion(
                model, loaders)
        else:
            epoch, perf, _ = load_checkpoint(model)
            print("Loaded checkpoint for model {}. Epoch: {}. Perf: {:0.2f}.".format(type(model), epoch, perf))

            loader = get_loader('validation')
            verb_scores, noun_scores, action_scores, verb_labels, noun_labels, action_labels, ids = get_scores(model, loader)

        verb_accuracies = topk_accuracy(verb_scores, verb_labels, (args.topK,))[0]
        noun_accuracies = topk_accuracy(noun_scores, noun_labels, (args.topK,))[0]
        action_accuracies = topk_accuracy(action_scores, action_labels, (args.topK,))[0]

        verb_accuracies_5 = topk_accuracy(verb_scores, verb_labels, (5,))[0]
        noun_accuracies_5 = topk_accuracy(noun_scores, noun_labels, (5,))[0]
        action_accuracies_5 = topk_accuracy(action_scores, action_labels, (5,))[0]

        many_shot_verbs, many_shot_nouns, many_shot_actions = get_many_shot()
        verb_recalls = topk_recall(verb_scores, verb_labels, k=args.topK, classes=many_shot_verbs)
        noun_recalls = topk_recall(noun_scores, noun_labels, k=args.topK, classes=many_shot_nouns)
        action_recalls = topk_recall(action_scores, action_labels, k=args.topK, classes=many_shot_actions)

        unseen, tail_verbs, tail_nouns, tail_actions = get_validation_ids()

        unseen_bool_idx = pd.Series(ids).isin(unseen).values
        tail_verbs_bool_idx = pd.Series(ids).isin(tail_verbs).values
        tail_nouns_bool_idx = pd.Series(ids).isin(tail_nouns).values
        tail_actions_bool_idx = pd.Series(ids).isin(tail_actions).values

        tail_verb_accuracies = topk_accuracy(verb_scores[tail_verbs_bool_idx], verb_labels[tail_verbs_bool_idx], (args.topK,))[0]
        tail_noun_accuracies = topk_accuracy(noun_scores[tail_nouns_bool_idx], noun_labels[tail_nouns_bool_idx], (args.topK,))[0]
        tail_action_accuracies = topk_accuracy(action_scores[tail_actions_bool_idx], action_labels[tail_actions_bool_idx], (args.topK,))[0]

        unseen_verb_accuracies = topk_accuracy(verb_scores[unseen_bool_idx], verb_labels[unseen_bool_idx], (args.topK,))[0]
        unseen_noun_accuracies = topk_accuracy(noun_scores[unseen_bool_idx], noun_labels[unseen_bool_idx], (args.topK,))[0]
        unseen_action_accuracies = topk_accuracy(action_scores[unseen_bool_idx], action_labels[unseen_bool_idx], (args.topK,))[0]

        print(f'Overall Top-1 Acc. (Verb) = {verb_accuracies*100:.2f}')
        print(f'Overall Top-1 Acc. (Noun) = {noun_accuracies*100:.2f}')
        print(f'Overall Top-1 Acc. (Action) = {action_accuracies*100:.2f}')
        print(f'Overall Top-5 Acc. (Verb) = {verb_accuracies_5*100:.2f}')
        print(f'Overall Top-5 Acc. (Noun) = {noun_accuracies_5*100:.2f}')
        print(f'Overall Top-5 Acc. (Action) = {action_accuracies_5*100:.2f}')
        print(f'Unseen Top-1 Acc. (Verb) = {unseen_verb_accuracies*100:.2f}')
        print(f'Unseen Top-1 Acc. (Noun) = {unseen_noun_accuracies*100:.2f}')
        print(f'Unseen Top-1 Acc. (Action) = {unseen_action_accuracies*100:.2f}')
        print(f'Tail Top-1 Acc. (Verb) = {tail_verb_accuracies*100:.2f}')
        print(f'Tail Top-1 Acc. (Noun) = {tail_noun_accuracies*100:.2f}')
        print(f'Tail Top-1 Acc. (Action) = {tail_action_accuracies*100:.2f}')

    elif args.mode == 'test':
        if args.ek100:
            mm = ['timestamps']
        else:
            mm = ['seen', 'unseen']

        for m in mm:
            if args.modality == 'late_fusion':
                loaders = [get_loader("test_{}".format(m), 'rgb'),
                           get_loader("test_{}".format(m), 'flow'),
                           get_loader("test_{}".format(m), 'obj'),
                           get_loader("test_{}".format(m), 'roi')]
                discarded_ids = loaders[0].dataset.discarded_ids
                verb_scores, noun_scores, action_scores, ids = get_scores_late_fusion(model, loaders)
            else:
                loader = get_loader("test_{}".format(m))
                epoch, perf, _ = load_checkpoint(model)
                print("Loaded checkpoint for model {}. Epoch: {}. Perf: {:.2f}.".format(type(model), epoch, perf))

                discarded_ids = loader.dataset.discarded_ids
                verb_scores, noun_scores, action_scores, ids = get_scores(model, loader)

            ids = list(ids) + list(discarded_ids)
            verb_scores = np.concatenate((verb_scores, np.zeros((len(discarded_ids), *verb_scores.shape[1:]))))
            noun_scores = np.concatenate((noun_scores, np.zeros((len(discarded_ids), *noun_scores.shape[1:]))))
            action_scores = np.concatenate((action_scores, np.zeros((len(discarded_ids), *action_scores.shape[1:]))))

            actions = pd.read_csv(join(args.path_to_data, 'actions.csv'))
            
            # map actions to (verb, noun) pairs
            a_to_vn = {a[1]['id']: tuple(a[1][['verb', 'noun']].values)
                       for a in actions.iterrows()}
            
            predictions = predictions_to_json(args.task, verb_scores, noun_scores, action_scores, ids, a_to_vn, version = '0.2' if args.ek100 else '0.1', sls=True)
            if args.ek100:
                with open(join(args.json_directory,exp_name+f"_test.json"), 'w') as f:
                    f.write(json.dumps(predictions, indent=4, separators=(',',': ')))
            else:
                with open(join(args.json_directory, exp_name + "_{}.json".format(m)), 'w') as f:
                    f.write(json.dumps(predictions, indent=4, separators=(',', ': ')))
            print('Printing done')


if __name__ == '__main__':

    if args.mode == 'test':
        assert args.json_directory is not None

    exp_name = make_model_name(args)
    print("Save file name ", exp_name)
    print("Printing Arguments ")
    print(args)

    # Considering args parameters from object model
    if args.modality == 'late_fusion':
        assert (args.mode != 'train')

        args_rgb = copy.deepcopy(args)
        args_rgb.video_feat_dim = 1024
        exp_rgb = make_model_name(args_rgb)

        args_flow = copy.deepcopy(args_rgb)
        exp_flow = make_model_name(args_flow)

        args_roi = copy.deepcopy(args_rgb)
        exp_roi = make_model_name(args_roi)

        # uncomment the next line when using TSM instead of TSN for rgb
        #args_rgb.video_feat_dim = 2048

    main()