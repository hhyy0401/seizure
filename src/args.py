import argparse


def get_args():
    parser = argparse.ArgumentParser('Train.')

    # General args
    parser.add_argument('--dataset',
                        type=str,
                        default="TUSZ",
                        choices=("TUSZ", "CHBMIT"),
                        help='Dataset to use.')
    parser.add_argument('--device',
                        type=str,
                        default="cuda",
                        help='Device to run the model.')
    parser.add_argument('--save_dir',
                        type=str,
                        default="/nas/share/r_kotoge/code/tts/result",
                        help='Directory to save the outputs and checkpoints.')
    parser.add_argument(
        '--load_model_path',
        type=str,
        default=None,
        help='Model checkpoint to start training/testing from.')
    parser.add_argument('--test',
                        default=False,
                        action='store_true',
                        help='Whether perform training.')
    parser.add_argument('--rand_seed',
                        type=int,
                        default=123,
                        help='Random seed.')
    parser.add_argument(
        '--task',
        type=str,
        default='detection',
        choices=(
            'detection',
            'classification',
            'prediction',
            'SS pre-training'),
        help="Seizure detection, Seizure prediction, seizure type classification, \
                            or SS pre-training.")
    parser.add_argument('--fine_tune',
                        default=False,
                        action='store_true',
                        help='Whether to fine-tune pre-trained model.')

    # Input args
    parser.add_argument(
        '--graph_type',
        choices=(
            'individual',
            'combined',
            'dynamic',
            'none'),
        default='dynamic',
        help='Graph mode. "none" skips dataloader-side graph construction '
             '(use when the model builds its graph internally — e.g. C+, E+).')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default='12',
                        help='Maximum sequence length in seconds.')
    parser.add_argument(
        '--output_seq_len',
        type=int,
        default=12,
        help='Output seq length for SS pre-training, in seconds.')
    parser.add_argument('--time_step_size',
                        type=int,
                        default=1,
                        help='Time step size in seconds.')
    parser.add_argument('--input_dir',
                        type=str,
                        default="/nas/share/r_kotoge/code/tts/resampled",
                        help='Dir to resampled EEG signals (.h5 files).')
    parser.add_argument('--raw_data_dir',
                        type=str,
                        default="/nas/share/r_kotoge/code/tts/TUSZ",
                        help='Dir to TUH data with raw EEG signals.')
    parser.add_argument('--preproc_dir',
                        type=str,
                        default=None,
                        help='Dir to preprocessed (Fourier transformed) data.')
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help='Top-k neighbors of each node to keep, for graph sparsity.')

    # Model args
    # NOTE on `ada_mshyper` and `st_hyper`:
    #   These are baselines ported verbatim from their official repositories
    #   (Ada-MSHyper NeurIPS 2024 / ST-Hyper CIKM 2025 same authors). Both
    #   originals do *time series forecasting*. We keep the algorithm 100%
    #   intact (normalization, AHL, CSCM, intra/inter-scale hypergraph
    #   processing, 3-path summation, Linear_Tran) and **only replace the
    #   final forecasting projection with a classification FF head**. The
    #   internal `pred_len` becomes an embedding dim rather than a forecast
    #   horizon. See model/ada_mshyper_eeg.py and model/st_hyper_eeg.py
    #   docstrings + BASELINES.md for the exact delta.
    parser.add_argument("--model_name", type=str, default="light_st_hyper", choices=(
        # Main model + ablation backbones (all share LightSTHyper architecture):
        "light_st_hyper",          # BiMamba backbone (main)
        "light_st_hyper_linear",   # linear backbone (no temporal modeling — ablation)
        "light_st_hyper_dwsep",    # depthwise-separable 1D conv backbone (ablation)
        "light_st_hyper_pair",     # pairwise spatial self-attn (no hyperedge — ablation)
        # Dense (per-second) variants for point-wise detection:
        "light_dense_hyper",       # per-t HyperedgeBlock, (B,T,1) seq2seq logits (ours dense)
        "evobrain_dense",          # paper EvoBrain backbone + per-t readout (baseline dense)
        "dcrnn_dense",             # paper DCRNN backbone + per-t FC + max-pool
        "gru_gcn_dense",           # paper GRU-GCN backbone + per-t readout (GCN at last-t)
        "lstm_dense",              # paper LSTM backbone + per-t FC
        "cnnlstm_dense",           # paper CNN-LSTM backbone + per-t FC
        "biot_dense",              # paper BIOT (Transformer) + per-t classifier
        # Paper Table 1 baselines:
        "evobrain", "dcrnn", "evolvegcn", "gru_gcn", "graphs4mer", "BIOT", "lstm", "cnnlstm",
        # EEG foundation models (pretrained encoders, fine-tuned end-to-end):
        "labram", "eegpt",
    ))
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to a pretrained checkpoint to initialise the model '
                             '(labram, eegpt). When unset, models fall back to env vars '
                             'LABRAM_CKPT / EEGPT_CKPT, then to the default scratch paths.')
    parser.add_argument('--n_hyperedges', type=int, default=8,
                        help='Number of hyperedges for light_dyn_hyper/light_static_hyper.')
    parser.add_argument('--use_node_emb', action='store_true', default=False,
                        help='Add a learnable per-node embedding after the temporal backbone (light_st_hyper_* only).')
    parser.add_argument('--aux_type', type=str, default='none',
                        choices=('none', 'bce', 'entropy'),
                        help='Auxiliary regularizer on the last hypergraph layer '
                             '(light_st_hyper_* only). '
                             '"bce": per-edge BCE deep supervision; '
                             '"entropy": per-(t,n) membership entropy.')
    parser.add_argument('--aux_weight', type=float, default=0.3,
                        help='Weight applied to the auxiliary loss when --aux_type != none.')
    # Temporal smoothness regularizer for dense (point-wise) detection.
    # Encourages adjacent-timestep logits to be similar — reflects the prior
    # that seizure events are continuous in time, not flickering. Only active
    # when --dense_labels is set and --smooth_weight > 0.
    parser.add_argument('--smooth_weight', type=float, default=0.0,
                        help='Weight for L2 smoothness loss on adjacent per-t logits. '
                             'loss = BCE + smooth_weight * mean((logits[:, t+1] - logits[:, t])^2). '
                             '0 disables.')
    # Point-wise (per-second) detection flags
    parser.add_argument('--dense_labels', action='store_true', default=False,
                        help='Per-second labels: dataloader returns y of shape (T,), '
                             'model emits (B, T, 1), trainer does BCE over (B, T). '
                             'Required for --model_name light_dense_hyper / evobrain_dense.')
    parser.add_argument('--eval_broadcast', action='store_true', default=False,
                        help='At eval time, broadcast a clip-level (B,1) prediction to (B,T) '
                             'so frame-level metrics can be computed. Use with binary baselines '
                             'plus --dense_labels to get frame metrics without retraining.')
    parser.add_argument('--save_hidden', action='store_true', default=False,
                        help='Save final-test hidden states (hidden.csv, can be ~6GB on TUSZ). '
                             'Off by default to keep disk usage small during sweeps.')
    parser.add_argument('--n_hyper_layers', type=int, default=2,
                        help='Number of hypergraph layers.')
    parser.add_argument('--n_pma_seeds', type=int, default=1,
                        help='Number of PMA seed queries (Set Transformer readout).')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                        help='Bi-directional Mamba in the per-channel backbone (default).')
    parser.add_argument('--no_bidirectional', dest='bidirectional', action='store_false',
                        help='Force uni-directional Mamba (paper-strict ablation).')
    parser.add_argument('--fs', type=float, default=200.0,
                        help='Sampling rate of resampled signal (200 Hz default).')
    parser.add_argument('--num_nodes',
                        type=int,
                        default=19,
                        help='Number of nodes in graph.')
    parser.add_argument('--num_rnn_layers',
                        type=int,
                        default=2,
                        help='Number of RNN layers in encoder and/or decoder.')
    parser.add_argument(
        '--pretrained_num_rnn_layers',
        type=int,
        default=3,
        help='Number of RNN layers in encoder and decoder for SS pre-training.')
    parser.add_argument('--rnn_units',
                        type=int,
                        default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dcgru_activation',
                        type=str,
                        choices=('relu', 'tanh'),
                        default='tanh',
                        help='Nonlinear activation used in DCGRU cells.')
    parser.add_argument('--input_dim',
                        type=int,
                        default=100,
                        help='Input seq feature dim.')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=1,
        help='Number of classes for seizure detection/classification.')
    parser.add_argument('--output_dim',
                        type=int,
                        default=100,
                        help='Output seq feature dim.')
    parser.add_argument('--max_diffusion_step',
                        type=int,
                        default=2,
                        help='Maximum diffusion step.')
    parser.add_argument('--cl_decay_steps',
                        type=int,
                        default=3000,
                        help='Scheduled sampling decay steps.')
    parser.add_argument(
        '--use_curriculum_learning',
        default=False,
        action='store_true',
        help='Whether to use curriculum training for seq-seq model.')
    parser.add_argument(
        '--use_fft',
        default=True,
        action='store_true',
        help='Whether the input data is Fourier transformed EEG signal or raw EEG.')
    
    parser.add_argument(
        '--stop',
        default=False,
        action='store_true',
        help='Whether to stop running the model.')


    # Training/test args
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=128,
                        help='Training batch size.')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=256,
                        help='Dev/test batch size.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.0,
                        help='Dropout rate for dropout layer before final FC.')
    parser.add_argument('--eval_every',
                        type=int,
                        default=1,
                        help='Evaluate on dev set every x epoch.')
    parser.add_argument(
        '--metric_name',
        type=str,
        default='auroc',
        choices=(
            'F1',
            'acc',
            'loss',
            'auroc'),
        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--lr_init',
                        type=float,
                        default=3e-4,
                        help='Initial learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=5e-4,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs for training.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--metric_avg',
                        type=str,
                        default='weighted',
                        help='weighted, micro or macro.')
    parser.add_argument('--data_augment',
                        default=True,
                        action='store_true',
                        help='Whether perform data augmentation.')
    parser.add_argument('--skip_midtest',
                        default=False,
                        action='store_true',
                        help='Skip mid-training test-set evaluation (logged but unused for selection).')
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Number of epochs of patience before early stopping.')
    parser.add_argument(
        '--pos_weight',
        type=float,
        default=None,
        help='BCEWithLogitsLoss pos_weight (None disables). Use ~ neg/pos ratio '
             'to compensate for class imbalance and calibrate logits.')
    parser.add_argument(
        '--sampling_ratio',
        type=float,
        default=1.0,
        help='Train-set non-seizure / seizure ratio. 1.0 = 1:1 (default). '
             'Set higher (e.g. 50) to expose model to natural prevalence; '
             'pair with --pos_weight for proper calibration.')
    parser.add_argument(
        '--fix_threshold',
        type=float,
        default=None,
        help='If set (e.g. 0.5), use this fixed decision threshold instead of '
             'dev-set max-F1 tuning. Required for fair cross-model comparison '
             'when logits are calibrated via --pos_weight.')
    parser.add_argument(
        '--agg',
        type=str,
        default="max",
        help='Aggregation method for node features for graph classification.')

    args = parser.parse_args()

    # which metric to maximize
    if args.metric_name == 'loss':
        # Best checkpoint is the one that minimizes loss
        args.maximize_metric = False
    elif args.metric_name in ('F1', 'acc', 'auroc'):
        # Best checkpoint is the one that maximizes F1 or acc
        args.maximize_metric = True
    else:
        raise ValueError(
            'Unrecognized metric name: "{}"'.format(
                args.metric_name))

    # must provide load_model_path if testing only
    if (args.load_model_path is None) and (args.test):
        raise ValueError(
            'For evaluation only, please provide trained model checkpoint in argument load_model_path.')

    # filter type for dcrnn
    if args.graph_type == "individual":
        args.filter_type = "dual_random_walk"
    if args.graph_type == "dynamic":
        args.filter_type = "dual_random_walk"
    if args.graph_type == "combined":
        args.filter_type = "laplacian"
    if args.graph_type == "none":
        args.filter_type = "dual_random_walk"  # unused but expected downstream

    return args