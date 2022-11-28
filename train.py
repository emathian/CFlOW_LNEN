import os, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from skimage.measure import label, regionprops
from tqdm import tqdm
from visualize import *
from model import load_decoder_arch, load_encoder_arch, positionalencoding2d, activation
from utils import *
from custom_datasets import *
from custom_models import *
import pandas as pd
## parallel
import hostlist
import torch.distributed as dist
from ignite.contrib import metrics
from torch.nn.parallel import DistributedDataParallel as DDP


gamma = 0.0
theta = torch.nn.Sigmoid()
log_theta = torch.nn.LogSigmoid()


def train_meta_epoch(c, epoch, loader, encoder, decoders, optimizer, pool_layers, N):
    P = c.condition_vec
    L = c.pool_layers
    decoders = [decoder.train() for decoder in decoders]
    adjust_learning_rate(c, optimizer, epoch)
    I = len(loader)
    print('Len train loader = ', I)
    iterator = iter(loader)
    for sub_epoch in range(c.sub_epochs):
        print('Epoch: {:d} \t sub-epoch: {:.4f} '.format(epoch, sub_epoch))
        train_loss = 0.0
        train_count = 0
        
        for i in range(I):
            if i % 20 == 0:
                print('step  % : ', (i/I) * 100, ' i/I = ', i , '/' , I)               
            # warm-up learning rate
            lr = warmup_learning_rate(c, epoch, i+sub_epoch*I, I*c.sub_epochs, optimizer)
            # sample batch
            try:
                image, _, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                image, _, _, _ = next(iterator)
            # encoder prediction
            image = image.to(c.device)  # single scale
            with torch.no_grad():
                _ = encoder(image)
            # train decoder
            e_list = list()
            c_list = list()
            for l, layer in enumerate(pool_layers):
                if 'vit' in c.enc_arch:
                    e = activation[layer].transpose(1, 2)[...,1:]
                    e_hw = int(np.sqrt(e.size(2)))
                    e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
                else:
                    e = activation[layer].detach()  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S    
                #
                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                perm = torch.randperm(E).to(c.device)  # BHW
                decoder = decoders[l]
                #
                FIB = E//N  # number of fiber batches
                assert FIB > 0, 'MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!'
                for f in range(FIB):  # per-fiber processing
                    idx = torch.arange(f*N, (f+1)*N)
                    c_p = c_r[perm[idx]]  # NxP
                    e_p = e_r[perm[idx]]  # NxC
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p,])
                    else:
                        z, log_jac_det = decoder(e_p)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
                    train_loss += t2np(loss.sum())
                    train_count += len(loss)
            if c.parallel:
                # ~~~~~~~~~~~~~~~~~ SAVE PART ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                epoch_s = str(epoch)
                sub_epoch_s = str(sub_epoch)
                if not os.path.exists(c.weights_dir):
                    os.makedirs(c.weights_dir, exist_ok = True )
                if not os.path.exists(os.path.join(c.weights_dir, c.class_name)):
                    os.makedirs(os.path.join(c.weights_dir, c.class_name), exist_ok = True)
                if not os.path.exists(os.path.join(c.weights_dir, c.class_name, epoch_s)):
                    os.makedirs(os.path.join(c.weights_dir, c.class_name, epoch_s), exist_ok = True)

                for j, ddp_decoder in enumerate(decoders):
                    if i % 1000 == 0:
                                
                        mean_train_loss = train_loss / train_count
                        print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_train_loss, lr))
                        filename = '{}_mataepoch_{}_subepoch_{}_loader_{}_decoder_{}.pt'.format(c.model, epoch_s, sub_epoch_s, i,j)
                        path = os.path.join(c.weights_dir, c.class_name, epoch_s,  filename)
                        if c.idr_torch_rank == 0:
                            print('Path : ', path)
                            torch.save(ddp_decoder.state_dict(), path)
            
        else:
            save_weights_epoch(c, encoder, decoders, c.model, epoch, sub_epoch)
        mean_train_loss = train_loss / train_count
        if c.verbose:
            print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_train_loss, lr))
    #


def test_meta_epoch(c, epoch, loader, encoder, decoders, pool_layers, N):
    # test
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    #
    P = c.condition_vec
    decoders = [decoder.eval() for decoder in decoders]
    height = list()
    width = list()
    image_list = list()
    gt_label_list = list()
    gt_mask_list = list()
    test_dist = [list() for layer in pool_layers]
    test_loss = 0.0
    test_count = 0
    start = time.time()
    files_path_list = []
    loss_list = []
    I = len(loader)

    with torch.no_grad():
        for i, (image, label, mask, filespath) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            if i % 10 == 0:
                print('step  % : ', (i/I) * 100, ' i/I = ', i , '/' , I)
            files_path_list.append(filespath)
            # save
            if c.viz:
                image_list.extend(t2np(image))
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))
            # data
            image = image.to(c.device) # single scale
            _ = encoder(image)  # BxCxHxW
            # test decoder
            e_list = list()
            for l, layer in enumerate(pool_layers):
                if 'vit' in c.enc_arch:
                    e = activation[layer].transpose(1, 2)[...,1:]
                    e_hw = int(np.sqrt(e.size(2)))
                    e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
                else:
                    e = activation[layer]  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S
                #
                if i == 0:  # get stats
                    height.append(H)
                    width.append(W)
                #
                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                #
                m = F.interpolate(mask, size=(H, W), mode='nearest')
                m_r = m.reshape(B, 1, S).transpose(1, 2).reshape(E, 1)  # BHWx1
                #
                decoder = decoders[l]
                FIB = E//N + int(E%N > 0)  # number of fiber batches
                for f in range(FIB):
                    if f < (FIB-1):
                        idx = torch.arange(f*N, (f+1)*N)
                    else:
                        idx = torch.arange(f*N, E)
                    #
                    c_p = c_r[idx]  # NxP
                    e_p = e_r[idx]  # NxC
                    m_p = m_r[idx] > 0.5  # Nx1
                    #
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p,])
                    else:
                        z, log_jac_det = decoder(e_p)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    test_loss += t2np(loss.sum())
                    test_count += len(loss)
                    test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()
                    #
            
            if i % 1000 == 0 :
                print('Epoch: {:d} \t step: {:.4f} '.format(epoch, i))
    fps = len(loader.dataset) / (time.time() - start)
    mean_test_loss = test_loss / test_count
    if c.verbose:
        print('Epoch: {:d} \t test_loss: {:.4f} and {:.2f} fps'.format(epoch, mean_test_loss, fps))
    #
    return height, width, image_list, test_dist, gt_label_list, gt_mask_list, files_path_list


def test_meta_epoch_lnen(c, epoch, loader, encoder, decoders, pool_layers, N):
    # test
    print('\nCompute loss and scores on test set:')
    #
    P = c.condition_vec
    decoders = [decoder.eval() for decoder in decoders]
    height = list()
    width = list()
    test_loss = 0.0
    test_count = 0
    start = time.time()
    score_label_mean_l = []
    I = len(loader)
    os.makedirs(os.path.join(c.viz_dir, c.class_name), exist_ok= True)
    if not c.infer_train:
        res_tab_name = 'results_table.csv'
    else:
        res_tab_name = 'results_table_train.csv'
    print('os.path.join(c.viz_dir, c.class_name, res_tab_name)  ', os.path.join(c.viz_dir, c.class_name, res_tab_name))
    with open(os.path.join(c.viz_dir, c.class_name, res_tab_name), 'w') as table_file: 
        table_file.write("file_path,binary_lab,MaxScoreAnomalyMap,MeanScoreAnomalyMap\n")
    table_file.close()
    with torch.no_grad():
        for i, (image, label, mask, filespath) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            if i % 1000 == 0:
                print('\n test_meta_epoch_lnen - step  % : ', (i/I) * 100, ' i/I = ', i , '/' , I)
            files_path_list_c = filespath
            # save
            
            labels_c = t2np(label)
            # data
            
            image = image.to(c.device) # single scale
            _ = encoder(image)  # BxCxHxW
            # test decoder
            e_list = list()
            test_dist = [list() for layer in pool_layers]
            test_map = [list() for p in pool_layers]
            for l, layer in enumerate(pool_layers):
                if 'vit' in c.enc_arch:
                    e = activation[layer].transpose(1, 2)[...,1:]
                    e_hw = int(np.sqrt(e.size(2)))
                    e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
                else:
                    e = activation[layer]  # BxCxHxW
#                     if l == 2:
#                         avg_pool = torch.nn.AvgPool2d(3, stride=4)
#                         avg_pool_2 = torch.nn.AvgPool2d(3)
#                         pool_enc = avg_pool(e)
#                         pool_enc_2 = avg_pool_2(pool_enc)
#                         pool_enc_2 = pool_enc_2.squeeze().detach().cpu().numpy()
#                         print("pool_enc_2 ", pool_enc_2.shape)
#                         write_enc_v_map(c, pool_enc_2, files_path_list_c)           
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S
                #
                if i == 0:  # get stats
                    height.append(H)
                    width.append(W)
                #
                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                #
                m = F.interpolate(mask, size=(H, W), mode='nearest')
                m_r = m.reshape(B, 1, S).transpose(1, 2).reshape(E, 1)  # BHWx1
                #
                decoder = decoders[l]
                FIB = E//N + int(E%N > 0)  # number of fiber batches
                for f in range(FIB):
                    if f < (FIB-1):
                        idx = torch.arange(f*N, (f+1)*N)
                    else:
                        idx = torch.arange(f*N, E)
                    #
                    c_p = c_r[idx]  # NxP
                    e_p = e_r[idx]  # NxC
                    m_p = m_r[idx] > 0.5  # Nx1
                    #
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p,])
                    else:
                        z, log_jac_det = decoder(e_p)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    test_loss += t2np(loss.sum())
                    test_count += len(loss)
                    test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()

            test_map = [list() for p in pool_layers]
            for l, p in enumerate(pool_layers):
                test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1
                test_norm-= torch.max(test_norm) # normalize likelihoods to (-Inf:0] by subtracting a constant
                test_prob = torch.exp(test_norm) # convert to probs in range [0:1]
                test_mask = test_prob.reshape(-1, height[l], width[l])
                test_mask = test_prob.reshape(-1, height[l], width[l])

                # upsample
                test_map[l] = F.interpolate(test_mask.unsqueeze(1),
                    size=c.crp_size, mode='bilinear', align_corners=True).squeeze().numpy()
            # score aggregation
            score_map = np.zeros_like(test_map[0])
            for l, p in enumerate(pool_layers):
                score_map += test_map[l]
            score_mask = score_map
            # invert probs to anomaly scores
            # ORIGINAL
            # super_mask = score_mask.max() - score_mask
            super_mask = score_mask
            # WARNING sice we write the map on the fly we can't get a proper value for score_mask.max() so we are writing  score_mask
            # Our probs are then reversed
            
            ######## WRITE ANOM MAP
            #write_anom_map(c, super_mask, files_path_list_c)           
            
            score_label_max = np.max(super_mask, axis=(1, 2))
            score_label_mean = np.mean(super_mask, axis=(1, 2))
            ### write table 
            # files_path_list_c
            res_df = pd.DataFrame()
            res_df['FilesPath'] = files_path_list_c
            res_df['BinaryLabels'] = labels_c
            res_df['MaxScoreAnomalyMap'] = score_label_max.flatten().tolist()
            res_df['MeanScoreAnomalyMap'] = score_label_mean.flatten().tolist()
            with open(os.path.join(c.viz_dir, c.class_name, res_tab_name), 'a') as table_file: 
                for row in range(res_df.shape[0]):
                    file_path_ = res_df[ 'FilesPath'][row]
                    binary_lab_ = res_df[ 'BinaryLabels'][row]
                    MaxScoreAnomalyMap = res_df[ 'MaxScoreAnomalyMap'][row]
                    MeanScoreAnomalyMap = res_df[ 'MeanScoreAnomalyMap'][row]
                    table_file.write(f"{file_path_},{binary_lab_},{MaxScoreAnomalyMap},{MeanScoreAnomalyMap}\n")
                table_file.close()
            if i % 1000 == 0 :
                print('Epoch: {:d} \t step: {:.4f} '.format(epoch, i))
        


def test_meta_fps(c, epoch, loader, encoder, decoders, pool_layers, N):
    # test
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    #
    P = c.condition_vec
    decoders = [decoder.eval() for decoder in decoders]
    height = list()
    width = list()
    image_list = list()
    gt_label_list = list()
    gt_mask_list = list()
    test_dist = [list() for layer in pool_layers]
    test_loss = 0.0
    test_count = 0
    A = len(loader.dataset)
    with torch.no_grad():
        # warm-up
        for i, (image, _, _, _) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            # data
            image = image.to(c.device) # single scale
            _ = encoder(image)  # BxCxHxW
        # measure encoder only
        torch.cuda.synchronize()
        start = time.time()
        for i, (image, _, _, _) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            # data
            image = image.to(c.device) # single scale
            _ = encoder(image)  # BxCxHxW
        # measure encoder + decoder
        torch.cuda.synchronize()
        time_enc = time.time() - start
        start = time.time()
        for i, (image, _, _, _) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            # data
            image = image.to(c.device) # single scale
            _ = encoder(image)  # BxCxHxW
            # test decoder
            e_list = list()
            for l, layer in enumerate(pool_layers):
                if 'vit' in c.enc_arch:
                    e = activation[layer].transpose(1, 2)[...,1:]
                    e_hw = int(np.sqrt(e.size(2)))
                    e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
                else:
                    e = activation[layer]  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S
                #
                if i == 0:  # get stats
                    height.append(H)
                    width.append(W)
                #
                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                #
                decoder = decoders[l]
                FIB = E//N + int(E%N > 0)  # number of fiber batches
                for f in range(FIB):
                    if f < (FIB-1):
                        idx = torch.arange(f*N, (f+1)*N)
                    else:
                        idx = torch.arange(f*N, E)
                    #
                    c_p = c_r[idx]  # NxP
                    e_p = e_r[idx]  # NxC
                    #
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p,])
                    else:
                        z, log_jac_det = decoder(e_p)
    #
    torch.cuda.synchronize()
    time_all = time.time() - start
    fps_enc = A / time_enc
    fps_all = A / time_all
    print('Encoder/All {:.2f}/{:.2f} fps'.format(fps_enc, fps_all))
    #
    return height, width, image_list, test_dist, gt_label_list, gt_mask_list


def train(c):
    if c.parallel :
        idr_torch_rank = int(os.environ['SLURM_PROCID'])
        # New config
        c.idr_torch_rank = idr_torch_rank
        
        local_rank = int(os.environ['SLURM_LOCALID'])
        print('local_rank  ', local_rank)
        idr_torch_size = int(os.environ['SLURM_NTASKS'])
        cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
        torch.backends.cudnn.enabled = False
        
        # get node list from slurm
        hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
        gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
        # define MASTER_ADD & MASTER_PORT
        os.environ['MASTER_ADDR'] = hostnames[0]
        os.environ['MASTER_PORT'] = str(12456 + int(min(gpu_ids))); #Avoid port conflits in the node #str(12345 + gpu_ids)

        dist.init_process_group(backend='nccl', 
                            init_method='env://', 
                            world_size=idr_torch_size, 
                            rank=idr_torch_rank)
        torch.cuda.set_device(local_rank)
    # According to the tutorial 
    gpu = torch.device("cuda")
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    L = c.pool_layers # number of pooled layers
    print('Number of pool layers =', L)
    encoder, pool_layers, pool_dims = load_encoder_arch(c, L)
    encoder = encoder.to(gpu).eval()
    if  c.parallel:
        ddp_encoder = DDP(encoder, device_ids=[local_rank]) # , output_device=local_rank
    # NF decoder
    decoders = [load_decoder_arch(c, pool_dim) for pool_dim in pool_dims]
    decoders = [decoder.to(gpu) for decoder in decoders]
    if c.parallel:
        ddp_decoders = []
        for decoder in decoders:
             ddp_decoders.append(DDP(decoder, device_ids=[local_rank])) # , output_device=local_rank
    params = list(decoders[0].parameters())

    for l in range(1, L):
        if c.parallel:
            params += list(ddp_decoders[l].parameters())
        else:
            params += list(decoders[l].parameters())
    # optimizer
    optimizer = torch.optim.Adam(params, lr=c.lr)
    # data
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}
    # task data
    if c.dataset == 'mvtec':
        train_dataset = MVTecDataset(c, is_train=True)
        test_dataset  = MVTecDataset(c, is_train=False)
    elif c.dataset == 'stc':
        train_dataset = StcDataset(c, is_train=True)
        test_dataset  = StcDataset(c, is_train=False)
    elif c.dataset == 'TumorNormal':
        if c.action_type == 'norm-train':
            train_dataset = TumorNormalDataset(c, is_train=True)
            test_dataset  = TumorNormalDataset(c, is_train=False)
        else:
            test_dataset  = TumorNormalDataset(c, is_train=False)
    elif c.dataset == 'TCAC':
        if c.action_type == 'norm-train':
            train_dataset = TCACDataset(c, is_train=True)
            test_dataset  = TCACDataset(c, is_train=False)
        else:
            test_dataset  = TCACDataset(c, is_train=False)
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))
    #
    if c.parallel and c.action_type == 'norm-train':
        batch_size_per_gpu =  c.batch_size // idr_torch_size
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=idr_torch_size, rank=idr_torch_rank) 
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,  batch_size=batch_size_per_gpu,  shuffle=False,   num_workers=0,                         pin_memory=True, sampler=train_sampler)
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=True, drop_last=False, **kwargs)

    else:
        if c.action_type == 'norm-train':
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, drop_last=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=True, drop_last=False, **kwargs)
            print('train/test loader length', len(train_loader.dataset), len(test_loader.dataset))
            print('train/test loader batches', len(train_loader), len(test_loader))
        else:
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=True, drop_last=False, **kwargs)
    N = 256  # hyperparameter that increases batch size for the decoder model by N
  
    # stats
    det_roc_obs = Score_Observer('DET_AUROC')
    seg_roc_obs = Score_Observer('SEG_AUROC')
    seg_pro_obs = Score_Observer('SEG_AUPRO')
    if c.action_type == 'norm-test':
        c.meta_epochs = 1
    for epoch in range(c.meta_epochs):  
        if c.action_type == 'norm-test' and c.checkpoint:
            if c.parallel:
                print("Load weights Parallel")
                for i, ddp_decoder in enumerate(ddp_decoders):
                    c_checkpoint = c.checkpoint[:-3]+f'_{i}.pt'
                    print(c_checkpoint)
                    ddp_decoder.load_state_dict(torch.load(c_checkpoint))
            else:
                load_weights(encoder, decoders, c.checkpoint)
            
#                 load_weights(encoder, decoders, c.checkpoint)
        elif c.action_type == 'norm-train':
            if c.parallel:
                train_meta_epoch(c, epoch, train_loader, ddp_encoder, ddp_decoders, optimizer, pool_layers, N)
            else:
                train_meta_epoch(c, epoch, train_loader, encoder, decoders, optimizer, pool_layers, N)
        else:
            raise NotImplementedError('{} is not supported action type!'.format(c.action_type))
        
        #height, width, test_image_list, test_dist, gt_label_list, gt_mask_list = test_meta_fps(
        #    c, epoch, test_loader, encoder, decoders, pool_layers, N)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #if not c.parallel:    
        
        if c.dataset != 'TumorNormal' and c.dataset != 'TCAC' :
            print('Len test_loader  ', len(test_loader))
            if c.parallel:
                print('Eval model parallel')
                height, width, test_image_list, test_dist, gt_label_list, gt_mask_list, files_path_list = test_meta_epoch(
            c, epoch, test_loader, ddp_encoder, ddp_decoders, pool_layers, N)
            else:
                height, width, test_image_list, test_dist, gt_label_list, gt_mask_list, files_path_list = test_meta_epoch(
            c, epoch, test_loader, encoder, decoders, pool_layers, N)
        else:
            if c.action_type == 'norm-train':
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                #LINE FOR VALIDATION
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                height, width, test_image_list, test_dist, gt_label_list, gt_mask_list, files_path_list = test_meta_epoch(
                c, epoch, test_loader, encoder, decoders, pool_layers, N)
            else:
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # LINE FOR EVALUATION
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if c.parallel:
                    print('EVAL IN C.PARALLEL test_meta_epoch_lnen')
                    height, width, test_image_list, test_dist, gt_label_list, gt_mask_list, files_path_list = test_meta_epoch_lnen(
                c, epoch, test_loader, ddp_encoder, ddp_decoders, pool_layers, N) 
                else:
                    print('TEst not paraller LNEN')
                    height, width, test_image_list, test_dist, gt_label_list, gt_mask_list, files_path_list = test_meta_epoch_lnen(
                c, epoch, test_loader, encoder, decoders, pool_layers, N) # test_meta_epoch_lnen
    
        if  c.dataset != 'TumorNormal' and c.dataset != 'TCAC'  :
            files_path_list =  [item for sublist in files_path_list for item in sublist]
            res_df = pd.DataFrame()
            res_df['FilesPath'] = files_path_list
            res_df['BinaryLabels'] = gt_label_list
            # PxEHW
            test_map = [list() for p in pool_layers]
            for l, p in enumerate(pool_layers):
                test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1
                test_norm-= torch.max(test_norm) # normalize likelihoods to (-Inf:0] by subtracting a constant
                test_prob = torch.exp(test_norm) # convert to probs in range [0:1]
                test_mask = test_prob.reshape(-1, height[l], width[l])
                test_mask = test_prob.reshape(-1, height[l], width[l])

                # upsample
                test_map[l] = F.interpolate(test_mask.unsqueeze(1),
                    size=c.crp_size, mode='bilinear', align_corners=True).squeeze().numpy()
            # score aggregation
            score_map = np.zeros_like(test_map[0])
            for l, p in enumerate(pool_layers):
                score_map += test_map[l]
            score_mask = score_map
            # invert probs to anomaly scores
            super_mask = score_mask.max() - score_mask
            # calculate detection AUROC
            score_label = np.max(super_mask, axis=(1, 2))
            score_label_mean = np.mean(super_mask, axis=(1, 2))
            res_df['MaxScoreAnomalyMap'] = score_label
            res_df['MeanScoreAnomalyMap'] = score_label_mean
            # Save result table 
            if c.action_type == 'norm-test':
                export_results_df(c, res_df)
                if  c.dataset != 'TumorNormal'  : # to comment
                    # Export anomaly map
                    write_anom_map(c, super_mask, files_path_list)

            gt_label = np.asarray(gt_label_list, dtype=np.bool)
            if not c.infer_train:
                det_roc_auc = roc_auc_score(gt_label, score_label)
                _ = det_roc_obs.update(100.0*det_roc_auc, epoch)
                print('DET_AUROC ', det_roc_auc)
                # calculate segmentation AUROC
                if c.dataset != 'TumorNormal'  :
                    gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=np.bool), axis=1)
                    seg_roc_auc = roc_auc_score(gt_mask.flatten(), super_mask.flatten())
                    print(' SEG_AUROC  ', seg_roc_auc)
                    save_best_seg_weights = seg_roc_obs.update(100.0*seg_roc_auc, epoch)
                    if save_best_seg_weights and c.action_type != 'norm-test':
                        if c.parallel:
                            print('Save best Weight parallel')
                            save_weights(c, ddp_encoder, ddp_decoders, c.model, run_date) 
                        else:
                            save_weights(c, encoder, decoders, c.model, run_date)  # avoid unnecessary saves
                # calculate segmentation AUPRO
                # from https://github.com/YoungGod/DFR:
                # No mask for TunorNormal -> Results not meaningful
                if c.pro and  c.dataset != 'TumorNormal' :  # and (epoch % 4 == 0):  # AUPRO is expensive to compute
                    max_step = 1000
                    expect_fpr = 0.3  # default 30%
                    max_th = super_mask.max()
                    min_th = super_mask.min()
                    delta = (max_th - min_th) / max_step
                    ious_mean = []
                    ious_std = []
                    pros_mean = []
                    pros_std = []
                    threds = []
                    fprs = []
                    binary_score_maps = np.zeros_like(super_mask, dtype=np.bool)
                    for step in range(max_step):
                        thred = max_th - step * delta
                        # segmentation
                        binary_score_maps[super_mask <= thred] = 0
                        binary_score_maps[super_mask >  thred] = 1
                        pro = []  # per region overlap
                        iou = []  # per image iou
                        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
                        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map 
                        for i in range(len(binary_score_maps)):    # for i th image
                            # pro (per region level)
                            label_map = label(gt_mask[i], connectivity=2)
                            props = regionprops(label_map)
                            for prop in props:
                                x_min, y_min, x_max, y_max = prop.bbox    # find the bounding box of an anomaly region 
                                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                                # cropped_mask = gt_mask[i][x_min:x_max, y_min:y_max]   # bug!
                                cropped_mask = prop.filled_image    # corrected!
                                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                                pro.append(intersection / prop.area)
                            # iou (per image level)
                            intersection = np.logical_and(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
                            union = np.logical_or(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
                            if gt_mask[i].any() > 0:    # when the gt have no anomaly pixels, skip it
                                iou.append(intersection / union)
                        # against steps and average metrics on the testing data
                        ious_mean.append(np.array(iou).mean())
                        #print("per image mean iou:", np.array(iou).mean())
                        ious_std.append(np.array(iou).std())
                        pros_mean.append(np.array(pro).mean())
                        pros_std.append(np.array(pro).std())
                        # fpr for pro-auc
                        gt_masks_neg = ~gt_mask
                        fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / gt_masks_neg.sum()
                        fprs.append(fpr)
                        threds.append(thred)
                    # as array
                    threds = np.array(threds)
                    pros_mean = np.array(pros_mean)
                    pros_std = np.array(pros_std)
                    fprs = np.array(fprs)
                    ious_mean = np.array(ious_mean)
                    ious_std = np.array(ious_std)
                    # best per image iou
                    best_miou = ious_mean.max()
                    #print(f"Best IOU: {best_miou:.4f}")
                    # default 30% fpr vs pro, pro_auc
                    idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
                    fprs_selected = fprs[idx]
                    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
                    pros_mean_selected = pros_mean[idx]    
                    seg_pro_auc = auc(fprs_selected, pros_mean_selected)
                    _ = seg_pro_obs.update(100.0*seg_pro_auc, epoch)
            
            if  c.dataset != 'TumorNormal':
                save_results(c, det_roc_obs, seg_roc_obs, seg_pro_obs, c.model, c.class_name, run_date)
            # export visualuzations
            if c.viz  and  c.dataset != 'TumorNormal' :
                precision, recall, thresholds = precision_recall_curve(gt_label, score_label)
                a = 2 * precision * recall
                b = precision + recall
                f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                det_threshold = thresholds[np.argmax(f1)]
                print('Optimal DET Threshold: {:.2f}'.format(det_threshold))
                precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), super_mask.flatten())
                a = 2 * precision * recall
                b = precision + recall
                f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                seg_threshold = thresholds[np.argmax(f1)]
                print('Optimal SEG Threshold: {:.2f}'.format(seg_threshold))
                export_groundtruth(c, test_image_list, gt_mask)
                export_scores(c, test_image_list, super_mask, seg_threshold)
                export_test_images(c, test_image_list, gt_mask, super_mask, seg_threshold)
                export_hist(c, gt_mask, super_mask, seg_threshold)
