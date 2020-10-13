import os
import json
import pickle
import numpy as np

import torch
import torch.nn as nn

from .models import *
from .attack import *
from .evaluation import *
from .curves import poly_chain, bezier_curve

def vanilla_train(model, train_loader, test_loader, attacker, epoch_num, epoch_ckpts, train_batches, optimizer, lr_func, eps_func,
    schedule_update_mode, out_folder, model_name, device, criterion, tosave, mask, **tricks):

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    acc_calculator = AverageCalculator()
    loss_calculator = AverageCalculator()

    for epoch_idx in range(epoch_num):

        acc_calculator.reset()
        loss_calculator.reset()

        model.train()
        for idx, (data_batch, label_batch) in enumerate(train_loader, 0):

            sys.stdout.write('Instance Idx: %d\r' % idx)
            epoch_batch_idx = epoch_idx + 1. / train_batches * idx if schedule_update_mode.lower() in ['batch',] else epoch_idx

            # Update the learning rate
            lr_this_batch = lr_func(epoch_batch_idx)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_batch
            if idx == 0:
                print('Learning rate = %1.2e' % lr_this_batch)

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch

            if attacker != None:
                if eps_func != None and (schedule_update_mode.lower() in ['batch'] or idx == 0):
                    next_threshold = eps_func(epoch_batch_idx)
                    attacker.adjust_threshold(next_threshold)
                model.eval()
                data_batch, label_batch = attacker.attack(model, optimizer, data_batch, label_batch, criterion)
                model.train()

            logits = model(data_batch)
            loss = criterion(logits, label_batch)
            acc = accuracy(logits.data, label_batch)
            optimizer.zero_grad()
            loss.backward()
            if mask != None:
                for n, p in model.named_parameters():
                    if p.grad is not None and n in mask:
                        p.grad.data = p.grad.data * mask[n].to(p.device)
            optimizer.step()
            optimizer.zero_grad()

            loss_calculator.update(loss.item(), data_batch.size(0))
            acc_calculator.update(acc.item(), data_batch.size(0))

        loss_this_epoch = loss_calculator.average
        acc_this_epoch = acc_calculator.average
        print('Train loss / acc after epoch %d: %.4f / %.2f%%' % (epoch_idx, loss_this_epoch, acc_this_epoch * 100.))
        tosave['train_loss'][epoch_idx] = loss_this_epoch
        tosave['train_acc'][epoch_idx] = acc_this_epoch

        loss_calculator.reset()
        acc_calculator.reset()

        model.eval()
        for idx, (data_batch, label_batch) in enumerate(test_loader, 0):

            sys.stdout.write('Instance Idx: %d\r' % idx)

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch

            if attacker != None:
                data_batch, label_batch = attacker.attack(model, optimizer, data_batch, label_batch, criterion)

            logits = model(data_batch)
            loss = criterion(logits, label_batch)
            acc = accuracy(logits.data, label_batch)

            loss_calculator.update(loss.item(), data_batch.size(0))
            acc_calculator.update(acc.item(), data_batch.size(0))

        loss_this_epoch = loss_calculator.average
        acc_this_epoch = acc_calculator.average
        print('Test loss / acc after epoch %d: %.4f / %.2f%%' % (epoch_idx, loss_this_epoch, acc_this_epoch * 100.))
        tosave['test_loss'][epoch_idx] = loss_this_epoch
        tosave['test_acc'][epoch_idx] = acc_this_epoch

        json.dump(tosave, open(os.path.join(out_folder, '%s.json' % model_name), 'w'))
        if (epoch_idx + 1) in epoch_ckpts:
            torch.save(model.state_dict(), os.path.join(out_folder, '%s_%d.ckpt' % (model_name, epoch_idx + 1)))

    torch.save(model.state_dict(), os.path.join(out_folder, '%s.ckpt' % model_name))

    return model, tosave

def attack(model, loader, attacker, optimizer, out_file, device, criterion, tosave, **tricks):

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    clean_acc_calculator = AverageCalculator()
    clean_loss_calculator = AverageCalculator()
    adv_acc_calculator = AverageCalculator()
    adv_loss_calculator = AverageCalculator()

    model.eval()
    for idx, (data_batch, label_batch) in enumerate(loader, 0):

        sys.stdout.write('Instance Idx: %d\r' % idx)

        clean_data_batch = data_batch.cuda(device) if use_gpu else data_batch
        clean_label_batch = label_batch.cuda(device) if use_gpu else label_batch

        adv_data_batch, adv_label_batch = attacker.attack(model, optimizer, clean_data_batch, clean_label_batch, criterion)

        clean_logits = model(clean_data_batch)
        clean_loss = criterion(clean_logits, clean_label_batch)
        clean_acc = accuracy(clean_logits.data, clean_label_batch)

        adv_logits = model(adv_data_batch)
        adv_loss = criterion(adv_logits, adv_label_batch)
        adv_acc = accuracy(adv_logits.data, adv_label_batch)

        clean_acc_calculator.update(clean_acc.item(), clean_data_batch.size(0))
        clean_loss_calculator.update(clean_loss.item(), clean_data_batch.size(0))
        adv_acc_calculator.update(adv_acc.item(), adv_data_batch.size(0))
        adv_loss_calculator.update(adv_loss.item(), adv_data_batch.size(0))

    clean_acc_this_epoch = clean_acc_calculator.average
    clean_loss_this_epoch = clean_loss_calculator.average
    adv_acc_this_epoch = adv_acc_calculator.average
    adv_loss_this_epoch = adv_loss_calculator.average

    print('Clean loss / acc: %.4f / %.2f%%' % (clean_loss_this_epoch, clean_acc_this_epoch * 100.))
    print('Adversarial loss / acc: %.4f / %.2f%%' % (adv_loss_this_epoch, adv_acc_this_epoch * 100.))

    tosave['clean_acc'] = clean_acc_this_epoch
    tosave['clean_loss'] = clean_loss_this_epoch
    tosave['adv_acc'] = adv_acc_this_epoch
    tosave['adv_loss'] = adv_loss_this_epoch

    if out_file != None:
        json.dump(tosave, open(out_file, 'w'))

    return clean_acc_this_epoch, clean_loss_this_epoch, adv_acc_this_epoch, adv_loss_this_epoch

def attack_list(model_list, loader, attacker, optimizer, out_file, device, criterion, tosave, **tricks):

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    clean_acc_calculator = AverageCalculator()
    clean_loss_calculator = AverageCalculator()
    adv_acc_calculator = AverageCalculator()
    adv_loss_calculator = AverageCalculator()

    for model in model_list:
        model.eval()

    for idx, (data_batch, label_batch) in enumerate(loader, 0):

        sys.stdout.write('Instance Idx: %d\r' % idx)

        clean_data_batch = data_batch.cuda(device) if use_gpu else data_batch
        clean_label_batch = label_batch.cuda(device) if use_gpu else label_batch

        adv_data_batch, adv_label_batch = attacker.attack_list(model_list, optimizer, clean_data_batch, clean_label_batch, criterion)

        clean_prob = 0.
        adv_prob = 0.
        for model in model_list:
            clean_logits_this_model = model(clean_data_batch)
            clean_prob_this_model = F.softmax(clean_logits_this_model)
            clean_prob = clean_prob + clean_prob_this_model
            adv_logits_this_model = model(adv_data_batch)
            adv_prob_this_model = F.softmax(adv_logits_this_model)
            adv_prob = adv_prob + adv_prob_this_model
        clean_prob = clean_prob / len(model_list)
        _, clean_prediction = clean_prob.max(dim = 1)
        adv_prob = adv_prob / len(model_list)
        _, adv_prediction = adv_prob.max(dim = 1)

        clean_loss = - torch.log(clean_prob).gather(dim = 1, index = clean_label_batch.view(-1, 1)).view(-1).mean()
        adv_loss = - torch.log(adv_prob).gather(dim = 1, index = adv_label_batch.view(-1, 1)).view(-1).mean()
        clean_acc = (clean_prediction == clean_label_batch).float().mean()
        adv_acc = (adv_prediction == adv_label_batch).float().mean()

        clean_acc_calculator.update(clean_acc.item(), clean_data_batch.size(0))
        clean_loss_calculator.update(clean_loss.item(), clean_data_batch.size(0))
        adv_acc_calculator.update(adv_acc.item(), adv_data_batch.size(0))
        adv_loss_calculator.update(adv_loss.item(), adv_data_batch.size(0))

    clean_acc_this_epoch = clean_acc_calculator.average
    clean_loss_this_epoch = clean_loss_calculator.average
    adv_acc_this_epoch = adv_acc_calculator.average
    adv_loss_this_epoch = adv_loss_calculator.average

    print('Clean loss / acc: %.4f / %.2f%%' % (clean_loss_this_epoch, clean_acc_this_epoch * 100.))
    print('Adversarial loss / acc: %.4f / %.2f%%' % (adv_loss_this_epoch, adv_acc_this_epoch * 100.))

    tosave['clean_acc'] = clean_acc_this_epoch
    tosave['clean_loss'] = clean_loss_this_epoch
    tosave['adv_acc'] = adv_acc_this_epoch
    tosave['adv_loss'] = adv_loss_this_epoch

    if out_file != None:
        json.dump(tosave, open(out_file, 'w'))

    return clean_acc_this_epoch, clean_loss_this_epoch, adv_acc_this_epoch, adv_loss_this_epoch

def curve_train(model, curve_type, train_loader, test_loader, train_batches, attacker, epoch_num, optimizer, lr_func,
    out_folder, model_name, device, criterion, tosave, **tricks):

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    acc_calculator = AverageCalculator()
    loss_calculator = AverageCalculator()

    for epoch_idx in range(epoch_num):

        acc_calculator.reset()
        loss_calculator.reset()

        model.train()
        for idx, (data_batch, label_batch) in enumerate(train_loader, 0):

            sys.stdout.write('Instance Idx: %d\r' % idx)

            # Update the learning rate
            if lr_func != None:
                epoch_batch_idx = epoch_idx + 1. / train_batches * idx
                lr_this_batch = lr_func(epoch_batch_idx)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_batch

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch

            # Generate coeffs
            t = np.random.uniform(0, 1)
            if curve_type.lower() in ['poly_chain',]:
                coeffs = poly_chain(t, pt_num = model.num_bends)
            elif curve_type.lower() in ['bezier_curve', 'bezier']:
                coeffs = bezier_curve(t, pt_num = model.num_bends)
            else:
                raise ValueError('Unrecognized curve type: %s' % curve_type)

            # Attack
            if attacker != None:
                data_batch, label_batch = attacker.attack_curve(model, optimizer, data_batch, label_batch, criterion, coeffs)

            logits = model(data_batch, coeffs)
            loss = criterion(logits, label_batch)
            acc = accuracy(logits.data, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_calculator.update(loss.item(), data_batch.size(0))
            acc_calculator.update(acc.item(), data_batch.size(0))

        loss_this_epoch = loss_calculator.average
        acc_this_epoch = acc_calculator.average
        print('Train loss / acc after epoch %d: %.4f / %.2f%%' % (epoch_idx, loss_this_epoch, acc_this_epoch * 100.))
        tosave['train_loss'][epoch_idx] = loss_this_epoch
        tosave['train_acc'][epoch_idx] = acc_this_epoch

        loss_calculator.reset()
        acc_calculator.reset()

        for idx, (data_batch, label_batch) in enumerate(test_loader, 0):

            sys.stdout.write('Instance Idx: %d\r' % idx)

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch

            t = np.random.uniform(0, 1)
            if curve_type.lower() in ['poly_chain',]:
                coeffs = poly_chain(t, pt_num = model.num_bends)
            elif curve_type.lower() in ['bezier_curve', 'bezier']:
                coeffs = bezier_curve(t, pt_num = model.num_bends)
            else:
                raise ValueError('Unrecognized curve type: %s' % curve_type)

            if attacker != None:
                data_batch, label_batch = attacker.attack_curve(model, optimizer, data_batch, label_batch, criterion, coeffs)

            logits = model(data_batch, coeffs)
            loss = criterion(logits, label_batch)
            acc = accuracy(logits.data, label_batch)

            loss_calculator.update(loss.item(), data_batch.size(0))
            acc_calculator.update(acc.item(), data_batch.size(0))

        loss_this_epoch = loss_calculator.average
        acc_this_epoch = acc_calculator.average
        print('Test loss / acc after epoch %d: %.4f / %.2f%%' % (epoch_idx, loss_this_epoch, acc_this_epoch * 100.))
        tosave['test_loss'][epoch_idx] = loss_this_epoch
        tosave['test_acc'][epoch_idx] = acc_this_epoch

        json.dump(tosave, open(os.path.join(out_folder, '%s.json' % model_name), 'w'))

    json.dump(tosave, open(os.path.join(out_folder, '%s.json' % model_name), 'w'))
    torch.save(model.state_dict(), os.path.join(out_folder, '%s.ckpt' % model_name))

    return model, tosave

