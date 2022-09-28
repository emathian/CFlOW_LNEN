import os, math
import numpy as np
import torch

MODEL_DIR  = f'/gpfsscratch/rech/ohv/ueu39kt/CFLOW/models/TumorNormal_train'

__all__ = ('save_weights_epoch','save_results', 'save_weights', 'load_weights', 'adjust_learning_rate', 'warmup_learning_rate')

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def save_results(c , det_roc_obs, seg_roc_obs, seg_pro_obs, model_name, class_name, run_date):
    result = '{:.2f},{:.2f},{:.2f} \t\tfor {:s}/{:s}/{:s} at epoch {:d}/{:d}/{:d} for {:s}\n'.format(
        det_roc_obs.max_score, seg_roc_obs.max_score, seg_pro_obs.max_score,
        det_roc_obs.name, seg_roc_obs.name, seg_pro_obs.name,
        det_roc_obs.max_epoch, seg_roc_obs.max_epoch, seg_pro_obs.max_epoch, class_name)
    if not os.path.exists(c.res_dir):
        os.makedirs(c.res_dir, exist_ok = True)
    if not os.path.exists(os.path.join(c.res_dir, c.class_name)):
        os.makedirs(os.path.join(c.res_dir, c.class_name), exist_ok = True)
    if c.dataset != 'TumorNormal':
        fp = open(os.path.join(c.res_dir, c.class_name, '{}_{}.txt'.format(model_name, run_date)), "w")
    else:
        fp = open(os.path.join(c.res_dir, '{}_{}.txt'.format(model_name, run_date)), "w")
    fp.write(result)
    fp.close()


def save_weights(c, encoder, decoders, model_name, run_date):
    if not os.path.exists(c.weights_dir):
        os.makedirs(c.weights_dir, exist_ok = True)
    if not os.path.exists(os.path.join(c.weights_dir, c.class_name)):
        os.makedirs(os.path.join(c.weights_dir, c.class_name), exist_ok = True)
    state = {'encoder_state_dict': encoder.state_dict(),
             'decoder_state_dict': [decoder.state_dict() for decoder in decoders]}
    filename = '{}_{}.pt'.format(model_name, run_date)
    path = os.path.join(c.weights_dir, c.class_name,  filename)
    if c.parallel:
        if c.idr_torch_rank == 0:
            torch.save(state, path)
    else:
        torch.save(state, path)
    print('Function: save_weights - Saving weights to {}'.format(filename))

def save_weights_epoch(c, encoder, decoders, model_name, epoch, sub_epoch):
    epoch = str(epoch)
    sub_epoch = str(sub_epoch)
    if not os.path.exists(c.weights_dir):
        os.makedirs(c.weights_dir, exist_ok = True)
    if not os.path.exists(os.path.join(c.weights_dir, c.class_name)):
        os.makedirs(os.path.join(c.weights_dir, c.class_name), exist_ok = True)
    if not os.path.exists(os.path.join(c.weights_dir, c.class_name, epoch)):
        os.makedirs(os.path.join(c.weights_dir, c.class_name, epoch), exist_ok = True)
    state = {'encoder_state_dict': encoder.state_dict(),
             'decoder_state_dict': [decoder.state_dict() for decoder in decoders]}
    filename = '{}_{}_{}.pt'.format(model_name, epoch, sub_epoch)
    path = os.path.join(c.weights_dir, c.class_name, epoch,  filename)
    print('Path : ', path)
    if c.parallel:
        if c.idr_torch_rank == 0:
            print('Path : ', path)
            torch.save(state, path)
    else:
        torch.save(state, path)
    print('Saving weights to {}'.format(filename))

def load_weights(encoder, decoders, filename):
    path = os.path.join(filename)
    state = torch.load(path, map_location='cuda:0')
    encoder.load_state_dict(state['encoder_state_dict'], strict=False)
    decoders = [decoder.load_state_dict(state, strict=False) for decoder, state in zip(decoders, state['decoder_state_dict'])]
    print('Loading weights from {}'.format(filename))


def adjust_learning_rate(c, optimizer, epoch):
    lr = c.lr
    if c.lr_cosine:
        eta_min = lr * (c.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / c.meta_epochs)) / 2
    else:
        steps = np.sum(epoch >= np.asarray(c.lr_decay_epochs))
        if steps > 0:
            lr = lr * (c.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(c, epoch, batch_id, total_batches, optimizer):
    if c.lr_warm and epoch < c.lr_warm_epochs:
        p = (batch_id + epoch * total_batches) / \
            (c.lr_warm_epochs * total_batches)
        lr = c.lr_warmup_from + p * (c.lr_warmup_to - c.lr_warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #
    for param_group in optimizer.param_groups:
        lrate = param_group['lr']
    return lrate
