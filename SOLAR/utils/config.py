from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(add_help=False)

#     parser.add_argument('--data', default='UKDALE', type=str)
    parser.add_argument('--model_name', default='CRNN', type=str)
    parser.add_argument('--experiment_name', default='NILM', help='Name of the experiment')
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--fuso', default='Europe/Rome', type=str)
    parser.add_argument('--ckpt_path', default='./ckpts', type=str)
    parser.add_argument('--log_path', default='./logs', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--house_indicies_train', default=[1], type=int, nargs='+')
    parser.add_argument('--house_indicies_test', default=[2], type=int, nargs='+')
    parser.add_argument('--train_size', type=float, default=0.8, help="train split size")
    parser.add_argument('--window_stride', type=int, default=1)
    parser.add_argument('--window_len', type=int, default=100)
    parser.add_argument('--resample_period', type=int, default=5, help="seconds to resample C2G data's")
    parser.add_argument('--n_epochs', default=150, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--early_stopping', default=10, type=int)

    # parser.add_argument('--dropout', default=0.1, type=float)
    # parser.add_argument('--kernel', default=5, type=int)
    # parser.add_argument('--num_conv_layers', default=3, type=int)
    # parser.add_argument('--gru_units', default=64, type=int)    
#     parser.add_argument('--hidden_size', default=256, type=int)
#     parser.add_argument('--n_heads', type=int, default=2)
#     parser.add_argument('--n_layers', type=int, default=2)
#     parser.add_argument('--d_ff', type=int, default=128)

#     parser.add_argument('--appliance_names', default=['kettle', 'microwave', 'fridge', 'washing machine', 'dish washer'], nargs='+', type=str)

#     parser.add_argument('--seq_len', type=int, default=480)
#     parser.add_argument('--pred_len', type=int, default=480)

#     parser.add_argument('--num_workers', default=4, type=int)
#     parser.add_argument('--seed', default=5, type=int, help='Seed for the reproducibility of the experiment')
#     parser.add_argument('--gpu_ids', default=[0], nargs='+', type=int, help='GPU ids')

    args = parser.parse_args()
    #args = update_preprocessing_parameters(args)
    return args


# # IF DATASET IS UKDALE 
# def update_preprocessing_parameters(args):
#     args.house_indicies_train = [1,5]
#     args.house_indicies_test = [2]
#     args.out_size = len(args.appliance_names)
#     return args

# # IF DATASET IS REFIT 
# def update_preprocessing_parameters(args):
#     args.house_indicies_train = [3,4,5,7,8,10,12,13,15,16,17,18,19]
#     args.house_indicies_test = [2,9]
#     #args.out_size = len(args.appliance_names)
#     return args

