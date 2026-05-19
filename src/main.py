import numpy as np
import os
import pickle
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import utils
from data.dataloader_detection import load_dataset_detection
from data.dataloader_chb import load_dataset_chb
from data.dataloader_prediction import load_dataset_prediction
from args import get_args
from collections import OrderedDict
from json import dumps
from model.gru_gcn import GRU_GCN_classification
from model.DCRNN import DCRNNModel_classification, DCRNNModel_nextTimePred
from model.EvoBrain import EvoBrain_classification
from model.EGCN import EvolveGCN_Model_classification
from model.BIOT import BIOTClassifier
from model.graphs4mer import GraphS4mer
from model.lstm import LSTMModel
from model.cnnlstm import CNN_LSTM
from tensorboardX import SummaryWriter
from tqdm import tqdm
from dotted_dict import DottedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import pandas as pd
import sklearn
import time
import torch
torch.autograd.set_detect_anomaly(True)

def main(args):

    # Get device
    args.cuda = torch.cuda.is_available()
    device = args.device if args.cuda else "cpu"

    # Set random seed
    utils.seed_torch(seed=args.rand_seed)

    # Get save directories
    args.save_dir = utils.get_save_dir(
        args.save_dir, args.dataset, args.task, args.max_seq_len, args.model_name, args.graph_type, args.rand_seed)
    # Save args
    args_file = os.path.join(args.save_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # Set up logger
    log = utils.get_logger(args.save_dir, 'train')
    tbx = SummaryWriter(args.save_dir)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))

    if args.model_name in ("BIOT", "labram", "eegpt"):
        # Raw-signal foundation models — BIOT does its own STFT,
        # LaBraM/EEGPT consume raw EEG (200 Hz / interpolated to 256 Hz).
        args.use_fft = False

    # Build dataset
    log.info('Building dataset...')
    if args.dataset == 'CHBMIT':
        print("Loading CHBMIT dataset...")
        dataloaders, datasets, scaler = load_dataset_chb(
            task = args.task,
            input_dir=args.input_dir,
            raw_data_dir=args.raw_data_dir,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            time_step_size=args.time_step_size,
            max_seq_len=args.max_seq_len,
            standardize=False,
            num_workers=args.num_workers,
            augmentation=args.data_augment,
            adj_mat_dir='./data/electrode_graph/adj_mx_3d.pkl',
            graph_type=args.graph_type,
            top_k=args.top_k,
            filter_type=args.filter_type,
            use_fft=args.use_fft,
            sampling_ratio=args.sampling_ratio,
            seed=123,
            preproc_dir=args.preproc_dir,
            return_raw=(args.model_name == "light_mamba_band_plv"),
            dense_labels=args.dense_labels)
    else: #TUSZ
        print("Loading TUSZ dataset...")
        if args.task == 'detection':
            dataloaders, datasets, scaler = load_dataset_detection(
                input_dir=args.input_dir,
                raw_data_dir=args.raw_data_dir,
                train_batch_size=args.train_batch_size,
                test_batch_size=args.test_batch_size,
                time_step_size=args.time_step_size,
                max_seq_len=args.max_seq_len,
                standardize=True,
                num_workers=args.num_workers,
                augmentation=args.data_augment,
                adj_mat_dir='./data/electrode_graph/adj_mx_3d.pkl',
                graph_type=args.graph_type,
                top_k=args.top_k,
                filter_type=args.filter_type,
                use_fft=args.use_fft,
                sampling_ratio=args.sampling_ratio,
                seed=123,
                preproc_dir=args.preproc_dir,
                dense_labels=args.dense_labels)


        elif args.task == 'prediction':
            dataloaders, datasets, scaler = load_dataset_prediction(
                input_dir=args.input_dir,
                raw_data_dir=args.raw_data_dir,
                train_batch_size=args.train_batch_size,
                test_batch_size=args.test_batch_size,
                time_step_size=args.time_step_size,
                max_seq_len=args.max_seq_len,
                standardize=True,
                num_workers=args.num_workers,
                augmentation=args.data_augment,
                adj_mat_dir='./data/electrode_graph/adj_mx_3d.pkl',
                graph_type=args.graph_type,
                top_k=args.top_k,
                filter_type=args.filter_type,
                use_fft=args.use_fft,
                sampling_ratio=args.sampling_ratio,
                seed=123,
                preproc_dir=args.preproc_dir)
        else:
            raise NotImplementedError

    # Build model
    log.info('Building model...')
    if args.model_name == "evobrain":
        if args.agg != "max":
            log.info("Using EvoBrain with aggregation method: {}".format(args.agg))
        model = EvoBrain_classification(args=args, num_classes=args.num_classes, device=device)
    elif args.model_name in ("light_st_hyper", "light_st_hyper_linear",
                              "light_st_hyper_dwsep", "light_st_hyper_pair"):
        # dwsep uses Conv1d which hits a cublasLt symbol issue on this
        # torch/CUDA combo when dispatched through cuDNN. Disable for dwsep.
        if args.model_name == "light_st_hyper_dwsep":
            torch.backends.cudnn.enabled = False
        from model.light_dyn_hyper import LightSTHyper_classification
        # _pair ablation: pairwise spatial self-attention replaces the
        # spatio-temporal hyperedge block; backbone stays Mamba.
        backbone = {
            "light_st_hyper": "mamba",      # BiMamba (main)
            "light_st_hyper_linear": "linear",
            "light_st_hyper_dwsep": "dwsep",
            "light_st_hyper_pair": "mamba",
        }[args.model_name]
        hyper_block = "pairwise" if args.model_name == "light_st_hyper_pair" else "static"
        model = LightSTHyper_classification(
            args=args, num_classes=args.num_classes, device=device,
            backbone_type=backbone, hyper_block_type=hyper_block)
    elif args.model_name == "light_dense_hyper":
        # Per-second variant: (B, T, 1) logits, (B, T) labels.
        if not args.dense_labels:
            raise ValueError("--model_name light_dense_hyper requires --dense_labels")
        from model.light_dense_hyper import DenseLightHyper_classification
        model = DenseLightHyper_classification(
            args=args, num_classes=args.num_classes, device=device,
            backbone_type="mamba")
    elif args.model_name == "evobrain_dense":
        # Paper EvoBrain backbone + per-t readout (dense baseline).
        if not args.dense_labels:
            raise ValueError("--model_name evobrain_dense requires --dense_labels")
        from model.EvoBrain_dense import EvoBrainDense_classification
        model = EvoBrainDense_classification(
            args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "dcrnn_dense":
        # Paper DCRNN backbone + per-t FC + max-pool (dense baseline).
        if not args.dense_labels:
            raise ValueError("--model_name dcrnn_dense requires --dense_labels")
        from model.DCRNN_dense import DCRNNDense_classification
        model = DCRNNDense_classification(
            args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "gru_gcn_dense":
        # Paper GRU-GCN backbone + per-t readout (dense baseline).
        if not args.dense_labels:
            raise ValueError("--model_name gru_gcn_dense requires --dense_labels")
        from model.gru_gcn_dense import GRU_GCN_Dense_classification
        model = GRU_GCN_Dense_classification(
            args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "lstm_dense":
        # Paper LSTM backbone + per-t FC (dense baseline).
        if not args.dense_labels:
            raise ValueError("--model_name lstm_dense requires --dense_labels")
        from model.lstm_dense import LSTMDense_classification
        model = LSTMDense_classification(args, args.num_classes, device)
    elif args.model_name == "cnnlstm_dense":
        # Paper CNN-LSTM backbone + per-t FC (dense baseline).
        if not args.dense_labels:
            raise ValueError("--model_name cnnlstm_dense requires --dense_labels")
        from model.cnnlstm_dense import CNN_LSTM_Dense_classification
        model = CNN_LSTM_Dense_classification(args.num_classes, args.dataset)
    elif args.model_name == "biot_dense":
        # Paper BIOT (LinearAttention Transformer) + per-t readout (dense baseline).
        if not args.dense_labels:
            raise ValueError("--model_name biot_dense requires --dense_labels")
        args.use_fft = False  # BIOT does its own STFT on raw signal
        from model.BIOT_dense import BIOTDenseClassifier_classification
        model = BIOTDenseClassifier_classification(
            n_classes=args.num_classes, n_channels=args.num_nodes,
            n_fft=args.input_dim, hop_length=int(args.input_dim / 2),
            T_out=args.max_seq_len)
    elif args.model_name == "dcrnn":
        model = DCRNNModel_classification(
            args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "evolvegcn":
        model = EvolveGCN_Model_classification(args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "graphs4mer":
        model = GraphS4mer(args=args, num_classes=args.num_classes)
    elif args.model_name == "gru_gcn":
        model = GRU_GCN_classification(args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "BIOT":
        args.use_fft = False
        model = BIOTClassifier(n_classes=args.num_classes, n_channels=args.num_nodes, n_fft=args.input_dim, hop_length=int(args.input_dim / 2))
    elif args.model_name == "lstm":
        model = LSTMModel(args, args.num_classes, device)
    elif args.model_name == "cnnlstm":
        model = CNN_LSTM(args.num_classes, args.dataset)
    elif args.model_name == "labram":
        from model.labram import LaBraM_classification
        model = LaBraM_classification(args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "eegpt":
        from model.eegpt import EEGPT_classification
        model = EEGPT_classification(args=args, num_classes=args.num_classes, device=device)
    else:
        raise NotImplementedError

    if not args.test:
        if not args.fine_tune:
            if args.load_model_path is not None:
                model = utils.load_model_checkpoint(
                    args.load_model_path, model)
        else:  # fine-tune from pretrained model
            if args.load_model_path is not None:
                args_pretrained = copy.deepcopy(args)
                setattr(
                    args_pretrained,
                    'num_rnn_layers',
                    args.pretrained_num_rnn_layers)
                pretrained_model = DCRNNModel_nextTimePred(
                    args=args_pretrained, device=device)  # placeholder
                pretrained_model = utils.load_model_checkpoint(
                    args.load_model_path, pretrained_model)

                model = utils.build_finetune_model(
                    model_new=model,
                    model_pretrained=pretrained_model,
                    num_rnn_layers=args.num_rnn_layers)
            else:
                raise ValueError(
                    'For fine-tuning, provide pretrained model in load_model_path!')

        num_params = utils.count_parameters(model)
        log.info('Total number of trainable parameters: {}'.format(num_params))

        model = model.to(device)

        # Train
        train(model, dataloaders, args, device, args.save_dir, log, tbx)

        # Load best model after training finished. If no in-training eval was
        # triggered (eval_every > num_epochs), there is no best ckpt and we
        # fall back to the in-memory final-epoch model.
        best_path = os.path.join(args.save_dir, 'best.pth.tar')
        if os.path.exists(best_path):
            model = utils.load_model_checkpoint(best_path, model)
        else:
            log.info('No best.pth.tar — using final-epoch in-memory model.')
        model = model.to(device)

    else:
        if args.load_model_path is not None:
            model = utils.load_model_checkpoint(
                args.load_model_path, model)

    # Evaluate on dev and test set
    log.info('Training DONE. Evaluating model...')
    model = model.to(device)
    dev_results = evaluate(model,
                           dataloaders['dev'],
                           args,
                           args.save_dir,
                           device,
                           log,
                           is_test=True,
                           nll_meter=None,
                           eval_set='dev')

    dev_results_str = ', '.join(
        ('{}: {:.3f}' if isinstance(v, float) else '{}: {}').format(k, v)
        for k, v in dev_results.items())
    log.info('DEV set prediction results: {}'.format(dev_results_str))

    test_results = evaluate(model,
                            dataloaders['test'],
                            args,
                            args.save_dir,
                            device,
                            log,
                            is_test=True,
                            nll_meter=None,
                            eval_set='test',
                            best_thresh=dev_results['best_thresh'])

    # Log to console
    test_results_str = ', '.join(
        ('{}: {:.3f}' if isinstance(v, float) else '{}: {}').format(k, v)
        for k, v in test_results.items())
    log.info('TEST set prediction results: {}'.format(test_results_str))


def train(model, dataloaders, args, device, save_dir, log, tbx):
    """
    Perform training and evaluate on val set
    """

    # Define loss function
    if (args.task == 'detection') or (args.task == 'prediction'):
        if args.pos_weight is not None:
            pw = torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw).to(device)
            log.info(f"Using BCEWithLogitsLoss(pos_weight={args.pos_weight})")
        else:
            loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    # Data loaders
    train_loader = dataloaders['train']
    dev_loader = dataloaders['dev']

    # Get saver
    saver = utils.CheckpointSaver(save_dir,
                                  metric_name=args.metric_name,
                                  maximize_metric=args.maximize_metric,
                                  log=log)

    # To train mode
    model.train()

    # Get optimizer and scheduler
    optimizer = optim.Adam(params=model.parameters(),
                           lr=args.lr_init, weight_decay=args.l2_wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # average meter for validation loss
    nll_meter = utils.AverageMeter()

    # Train
    log.info('Training...')
    epoch = 0
    step = 0
    prev_val_loss = 1e10
    # Track best dev metric (e.g. AUROC) for early-stopping. Decoupled from
    # prev_val_loss because dev loss can keep shrinking via confidence
    # sharpening long after dev AUROC has plateaued — this masks overfit and
    # prevents patience from triggering. We reset patience only when the
    # ckpt-selection metric actually improves.
    if args.maximize_metric:
        best_dev_metric = -1e10
    else:
        best_dev_metric = 1e10
    patience_count = 0
    early_stop = False
    memory_usage_list = []
    time_list = []
    while (epoch != args.num_epochs) and (not early_stop):
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        total_samples = len(train_loader.dataset)
        with torch.enable_grad(), \
                tqdm(total=total_samples) as progress_bar:
            for batch in train_loader:
                # Dataloader may return 6-tuple or 7-tuple (with raw_signal at end
                # when return_raw=True for the band-PLV model).
                if len(batch) == 7:
                    x, y, seq_lengths, supports, adj, file_name, raw_signal = batch
                    raw_signal = raw_signal.to(device)
                else:
                    x, y, seq_lengths, supports, adj, file_name = batch
                    raw_signal = None
                batch_size = x.shape[0]

                # input seqs
                x = x.to(device)
                # Dense: keep (B, T) per-second labels; binary: flatten to (B,).
                if args.dense_labels:
                    y = y.to(device).float()
                else:
                    y = y.view(-1).to(device)
                seq_lengths = seq_lengths.view(-1).to(device)
                supports = supports.to(device)
                adj = adj.to(device)

                # Zero out optimizer first
                optimizer.zero_grad()

                # Forward
                # (batch_size, num_classes)
                start_time = time.time()
                initial_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0

                if args.model_name in ("evobrain", "light_st_hyper",
                                       "light_st_hyper_linear", "light_st_hyper_dwsep",
                                       "light_st_hyper_pair",
                                       "light_dense_hyper", "evobrain_dense",
                                       "gru_gcn_dense",
                                       "evolvegcn", "gru_gcn"):
                    logits, _ = model(x, seq_lengths, adj)
                elif args.model_name in ("dcrnn", "dcrnn_dense"):
                    logits, _ = model(x, seq_lengths, supports)
                elif args.model_name in ("BIOT", "biot_dense", "labram", "eegpt"):
                    logits, _ = model(x)
                elif args.model_name in ("lstm", "cnnlstm", "graphs4mer",
                                          "lstm_dense", "cnnlstm_dense"):
                    logits, _ = model(x, seq_lengths)
                else:
                    print("model_name: ", args.model_name)
                    raise NotImplementedError

                # Dense (B, T, 1) seq2seq path: BCE over (B, T)
                if args.dense_labels and logits.dim() == 3:
                    if logits.shape[-1] == 1:
                        logits = logits.squeeze(-1)         # (B, T)
                    loss = loss_fn(logits, y.float())       # y: (B, T)
                    # Optional temporal smoothness regularizer.
                    if getattr(args, 'smooth_weight', 0.0) > 0:
                        diff = logits[:, 1:] - logits[:, :-1]   # (B, T-1)
                        smooth_loss = (diff ** 2).mean()
                        loss = loss + args.smooth_weight * smooth_loss
                elif args.dense_labels and logits.dim() == 2:
                    # Binary clip-level model with --dense_labels (broadcast
                    # baseline path): collapse per-t labels to clip label for
                    # the BCE loss. Frame metrics still computed at eval time
                    # via --eval_broadcast.
                    if logits.shape[-1] == 1:
                        logits = logits.view(-1)
                    y_clip = (y.amax(dim=-1) > 0.5).float()
                    loss = loss_fn(logits, y_clip)
                else:
                    if logits.shape[-1] == 1:
                        logits = logits.view(-1)
                    loss = loss_fn(logits, y)

                # Hypergraph aux regularizer (bce|entropy) on light_st_hyper_*.
                if getattr(args, "aux_type", "none") != "none" and hasattr(model, "compute_aux_loss"):
                    aux2 = model.compute_aux_loss(y)
                    if aux2.requires_grad:
                        loss = loss + args.aux_weight * aux2
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()

                end_time = time.time()
                max_memory = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0

                memory_usage_list.append(max_memory - initial_memory)
                time_list.append(end_time - start_time)

                step += batch_size

                # Log info
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         loss=loss_val,
                                         lr=optimizer.param_groups[0]['lr'])

                tbx.add_scalar('train/Loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)
                if (args.stop == True) and (len(time_list) > 1000):
                    break

            if epoch % args.eval_every == 0:
                # Evaluate and save checkpoint
                log.info('Evaluating at epoch {}...'.format(epoch))
                eval_results = evaluate(model,
                                        dev_loader,
                                        args,
                                        save_dir,
                                        device,
                                        log,
                                        is_test=False,
                                        nll_meter=nll_meter)
                best_path = saver.save(epoch,
                                       model,
                                       optimizer,
                                       eval_results[args.metric_name])

                # Mid-training test snapshot (logged but not used for saving/early-stop).
                if not getattr(args, 'skip_midtest', False):
                    mid_test_results = evaluate(model,
                                                dataloaders['test'],
                                                args,
                                                save_dir,
                                                device,
                                                log,
                                                is_test=True,
                                                nll_meter=None)
                else:
                    mid_test_results = None

                # Accumulate patience on the same metric used for ckpt
                # selection (args.metric_name). dev_loss can decrease forever
                # via confidence sharpening even after AUROC plateaus.
                cur_metric = eval_results[args.metric_name]
                if args.maximize_metric:
                    improved = cur_metric > best_dev_metric
                else:
                    improved = cur_metric < best_dev_metric
                if improved:
                    patience_count = 0
                    best_dev_metric = cur_metric
                else:
                    patience_count += 1
                prev_val_loss = eval_results['loss']

                # Early stop
                if patience_count == args.patience:
                    early_stop = True

                # Back to train mode
                model.train()

                # Log to console
                results_str = ', '.join(
                    ('{}: {:.3f}' if isinstance(v, float) else '{}: {}').format(k, v)
                    for k, v in eval_results.items())
                log.info('Dev {}'.format(results_str))
                if mid_test_results is not None:
                    test_str = ', '.join(
                        ('{}: {:.3f}' if isinstance(v, float) else '{}: {}').format(k, v)
                        for k, v in mid_test_results.items())
                    log.info('MidTest {}'.format(test_str))

                # Log to TensorBoard
                log.info('Visualizing in TensorBoard...')
                for k, v in eval_results.items():
                    if isinstance(v, (int, float)):
                        tbx.add_scalar('eval/{}'.format(k), v, step)
                if mid_test_results is not None:
                    for k, v in mid_test_results.items():
                        if isinstance(v, (int, float)):
                            tbx.add_scalar('midtest/{}'.format(k), v, step)

        # Step lr scheduler
        scheduler.step()

    max_memory_usage = np.max(memory_usage_list) / (1024 ** 2)  
    avg_time_per_batch = np.mean(time_list)

    log.info(f"Average Training Time per Batch: {avg_time_per_batch:.4f} seconds")


def evaluate(
        model,
        dataloader,
        args,
        save_dir,
        device,
        log,
        is_test=False,
        nll_meter=None,
        eval_set='dev',
        best_thresh=0.5):
    # To evaluate mode
    model.eval()

    # Define loss function
    if (args.task == 'detection') or (args.task == 'prediction'):
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    y_pred_all = []
    y_true_all = []
    y_prob_all = []
    file_name_all = []
    hidden_all = []
    time_list = []
    with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
        for batch in dataloader:
            if len(batch) == 7:
                x, y, seq_lengths, supports, adj, file_name, raw_signal = batch
                raw_signal = raw_signal.to(device)
            else:
                x, y, seq_lengths, supports, adj, file_name = batch
                raw_signal = None
            batch_size = x.shape[0]

            # Input seqs
            x = x.to(device)
            # Dense: keep (B, T) label shape; binary: flatten to (B,).
            if args.dense_labels:
                y = y.to(device).float()
            else:
                y = y.view(-1).to(device)
            seq_lengths = seq_lengths.view(-1).to(device)
            supports = supports.to(device)
            adj = adj.to(device)

            start_time = time.time()
            # Forward
            # (batch_size, num_classes)
            if args.model_name in ("evobrain", "light_st_hyper",
                                   "light_st_hyper_linear", "light_st_hyper_dwsep",
                                   "light_st_hyper_pair",
                                   "light_dense_hyper", "evobrain_dense",
                                   "gru_gcn_dense",
                                   "evolvegcn", "gru_gcn"):
                logits, hidden = model(x, seq_lengths, adj)
            elif args.model_name in ("dcrnn", "dcrnn_dense"):
                logits, hidden = model(x, seq_lengths, supports)
            elif args.model_name in ("BIOT", "biot_dense", "labram", "eegpt"):
                logits, hidden = model(x)
            elif args.model_name in ("lstm", "cnnlstm", "graphs4mer",
                                      "lstm_dense", "cnnlstm_dense"):
                logits, hidden = model(x, seq_lengths)
            else:
                raise NotImplementedError

            # Unified (B, T) accumulators (T=1 for clip-only models).
            # For binary models with --eval_broadcast and (B, T) labels, tile
            # the (B,) prob to (B, T) so frame metrics can be computed too.
            if args.dense_labels and logits.dim() == 3:
                if logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)                  # (B, T)
                prob_bt = torch.sigmoid(logits).cpu().numpy()    # (B, T)
                true_bt = y.cpu().numpy().astype(int)            # (B, T)
                loss_logits, loss_target = logits, y
                prob_2d = True
            elif args.num_classes == 1 and getattr(args, 'eval_broadcast', False) \
                    and y.dim() == 2:
                logits_b = logits.view(-1)                       # (B,)
                prob_b = torch.sigmoid(logits_b).cpu().numpy()   # (B,)
                T_eval = y.shape[1]
                prob_bt = np.tile(prob_b[:, None], (1, T_eval))  # (B, T)
                true_bt = y.cpu().numpy().astype(int)            # (B, T)
                loss_logits = logits_b
                loss_target = (y.amax(dim=-1) > 0.5).float()
                prob_2d = True
            elif args.num_classes == 1:
                logits_b = logits.view(-1)
                prob_b = torch.sigmoid(logits_b).cpu().numpy()
                true_b = y.view(-1).cpu().numpy().astype(int)
                prob_bt = prob_b[:, None]                        # (B, 1)
                true_bt = true_b[:, None]
                loss_logits = logits_b
                loss_target = y.view(-1).float()
                prob_2d = True
            else:
                # multi-class softmax (legacy)
                y_prob = F.softmax(logits, dim=1).cpu().numpy()
                y_pred = np.argmax(y_prob, axis=1).reshape(-1)
                y_true = y.cpu().numpy().astype(int)
                loss_logits, loss_target = logits, y
                prob_bt = None
                prob_2d = False

            time_list.append(time.time() - start_time)

            # Update loss
            loss = loss_fn(loss_logits, loss_target)
            if nll_meter is not None:
                nll_meter.update(loss.item(), batch_size)

            if prob_2d:
                y_prob_all.append(prob_bt)
                y_true_all.append(true_bt)
            else:
                y_pred_all.append(y_pred)
                y_true_all.append(y_true)
                y_prob_all.append(y_prob)
            file_name_all.extend(file_name)
            hidden_all.append(hidden.cpu().reshape(hidden.shape[0], -1))

            # Log info
            progress_bar.update(batch_size)
            if (args.stop == True) and (len(time_list) > 1000):
                break

    hidden_all = np.concatenate(hidden_all, axis=0)

    # ---- Build unified frame-level and clip-level views from (B, T) accumulators ----
    multi_class_path = (args.num_classes != 1)
    if not multi_class_path:
        prob_2d = np.concatenate(y_prob_all, axis=0)                       # (N, T_or_1)
        true_2d = np.concatenate(y_true_all, axis=0)                       # (N, T_or_1)
        y_prob_frame = prob_2d.reshape(-1)
        y_true_frame = true_2d.reshape(-1)
        y_prob_clip = prob_2d.max(axis=1)
        y_true_clip = (true_2d.max(axis=1) > 0).astype(int)
        # Default reporting uses frame view when T>1 (dense/broadcast), clip otherwise.
        if prob_2d.shape[1] > 1:
            y_prob_all = y_prob_frame
            y_true_all = y_true_frame
            y_pred_all = (y_prob_all > best_thresh).astype(int)
        else:
            y_prob_all = y_prob_clip
            y_true_all = y_true_clip
            y_pred_all = (y_prob_all > best_thresh).astype(int)
    else:
        y_pred_all = np.concatenate(y_pred_all, axis=0)
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_prob_all = np.concatenate(y_prob_all, axis=0)

    if is_test:
        results_file = os.path.join(save_dir, f'{eval_set}_results.npz')
        save_kwargs = dict(
            y_true=y_true_all, y_pred=y_pred_all, y_prob=y_prob_all,
            file_names=file_name_all)
        if not multi_class_path:
            save_kwargs.update(
                y_true_frame=y_true_frame, y_prob_frame=y_prob_frame,
                y_true_clip=y_true_clip, y_prob_clip=y_prob_clip)
        np.savez(results_file, **save_kwargs)
        print(f"Evaluation results saved to {results_file}")

    if eval_set == 'test' and getattr(args, 'save_hidden', False):
        output_file = os.path.join(save_dir, "hidden.csv")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df = pd.DataFrame(hidden_all)
        df.to_csv(output_file, mode='w', header=False, index=False)

        output_file = os.path.join(save_dir, "true_labels.csv")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df = pd.DataFrame(np.expand_dims(y_true_all, axis=0))
        df.to_csv(output_file, mode='w', header=False, index=False)

    avg_time_per_batch = np.mean(time_list)
    log.info(f"Average Test Time per Batch: {avg_time_per_batch:.4f} seconds")

    avg_for_eval = "binary" if ((args.task == "detection") or
                                (args.task == "prediction")) else "weighted"
    eval_loss = nll_meter.avg if (nll_meter is not None) else loss.item()
    results_list = [('loss', eval_loss)]

    if multi_class_path:
        if args.task in ("detection", "prediction") and eval_set == 'dev' and is_test:
            best_thresh = (float(args.fix_threshold)
                           if args.fix_threshold is not None
                           else utils.thresh_max_f1(y_true=y_true_all, y_prob=y_prob_all))
            y_pred_all = (y_prob_all > best_thresh).astype(int)
        scores_dict, _, _ = utils.eval_dict(
            y_pred=y_pred_all, y=y_true_all, y_prob=y_prob_all,
            file_names=file_name_all, average=avg_for_eval)
        for k in ('acc', 'F1', 'recall', 'precision'):
            results_list.append((k, scores_dict[k]))
        results_list.append(('best_thresh', best_thresh))
        if 'auroc' in scores_dict:
            results_list.append(('auroc', scores_dict['auroc']))
        return OrderedDict(results_list)

    # ---- Binary path: frame-level AND clip-level metrics ----
    if args.task in ("detection", "prediction") and eval_set == 'dev' and is_test:
        if args.fix_threshold is not None:
            thresh_frame = thresh_clip = float(args.fix_threshold)
        else:
            thresh_frame = utils.thresh_max_f1(y_true=y_true_frame, y_prob=y_prob_frame)
            thresh_clip = utils.thresh_max_f1(y_true=y_true_clip, y_prob=y_prob_clip)
    else:
        thresh_frame = thresh_clip = best_thresh

    y_pred_frame = (y_prob_frame > thresh_frame).astype(int)
    y_pred_clip = (y_prob_clip > thresh_clip).astype(int)

    frame_scores, _, _ = utils.eval_dict(
        y_pred=y_pred_frame, y=y_true_frame, y_prob=y_prob_frame,
        file_names=None, average=avg_for_eval)
    clip_scores, _, _ = utils.eval_dict(
        y_pred=y_pred_clip, y=y_true_clip, y_prob=y_prob_clip,
        file_names=None, average=avg_for_eval)

    if is_test:
        roc_file = os.path.join(save_dir, f'{eval_set}_roc_data.npz')
        fpr_f, tpr_f, _ = sklearn.metrics.roc_curve(y_true_frame, y_prob_frame)
        fpr_c, tpr_c, _ = sklearn.metrics.roc_curve(y_true_clip, y_prob_clip)
        np.savez(roc_file, fpr_frame=fpr_f, tpr_frame=tpr_f,
                 fpr_clip=fpr_c, tpr_clip=tpr_c,
                 thresh_frame=thresh_frame, thresh_clip=thresh_clip)
        print(f"ROC curve data saved to {roc_file}")

    # Primary view: frame if dense, clip otherwise.
    if args.dense_labels:
        primary = ('frame', frame_scores, thresh_frame)
    else:
        primary = ('clip', clip_scores, thresh_clip)

    p_name, p_scores, p_thresh = primary
    for k in ('acc', 'F1', 'recall', 'precision'):
        results_list.append((k, p_scores[k]))
    results_list.append(('best_thresh', p_thresh))
    if 'auroc' in p_scores:
        results_list.append(('auroc', p_scores['auroc']))
    # Explicit frame_/clip_ keys for downstream tables.
    for prefix, scores, t in [('frame', frame_scores, thresh_frame),
                              ('clip', clip_scores, thresh_clip)]:
        for k in ('acc', 'F1', 'recall', 'precision'):
            results_list.append((f'{prefix}_{k}', scores[k]))
        results_list.append((f'{prefix}_best_thresh', t))
        if 'auroc' in scores:
            results_list.append((f'{prefix}_auroc', scores['auroc']))
    results_list.append(('primary_view', p_name))

    return OrderedDict(results_list)

def check_tensor(data, description):
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"{description} is not a tensor! Found type: {type(data)}")


if __name__ == '__main__':
    main(get_args())
