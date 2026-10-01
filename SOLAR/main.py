import os
import torch
import torch.nn as nn
import logging
from utils.config import get_args
from utils.random_seed import random_seed
from utils.summary import Summary
from data.C2G_parser import C2G_parser
from data.C2G_dataloader import C2GDataloader
from experiment import train, test
import numpy as np
import json
import inspect
import onnxruntime as ort
import sys

if __name__ == '__main__':
    ##### Args #####
    args = get_args()

    random_seed(args.seed)
    print(f'Fixing random seed: {args.seed}')

    ##### Experiment name, checkpoint, logging #####
    exp_name = f'{args.experiment_name}'
    file_name = f'{exp_name}_winSize_{args.window_len}_winStride{args.window_stride}_period_{args.resample_period}_seed_{args.seed}'
    print(f'File name: {file_name}')
    traintest_name = (f'train_{args.house_indicies_train}_test{args.house_indicies_test}')
    path_ckpts = os.path.join(args.experiment_name, file_name, args.model_name, traintest_name,  args.ckpt_path)
    path_logs = os.path.join(args.experiment_name, file_name, args.model_name, traintest_name,  args.log_path)
    os.makedirs(path_ckpts, exist_ok=True)
    print(f'ckpts in: {path_ckpts}')
    os.makedirs(path_logs, exist_ok=True)
    print(f'logs in: {path_logs}')
    path_ckpts = os.path.join(path_ckpts, 'model.pth')

    logging.basicConfig(
        handlers=[
            # Handler per la scrittura su File
            logging.FileHandler(os.path.join(path_logs, 'log.txt'), mode='w'),
            # Handler per la stampa a Terminale (stdout)
            logging.StreamHandler(sys.stdout)
        ],
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )

    #### Data loading #####
    logging.debug(f"Caricando i dati da {args.data_path}")

    ds_parser = C2G_parser(args)
    dataloader = C2GDataloader(args, ds_parser)
    train_loader, val_loader, test_loader, x_min, x_max, y_min, y_max = dataloader.get_dataloaders()
    logging.debug(f"Train loader length: {len(train_loader)}, Val loader length: {len(val_loader)}, Test loader length: {len(test_loader)}")
    

    ######### Model costruction ##############
    logging.debug(f"Costruendo il modello {args.model_name}")
    
    if args.model_name == 'CRNN':
        from networks import CRNNModel
        model = CRNNModel()
    elif args.model_name == 'CNN':
        from networks import CNNModel
        model = CNNModel()
    elif args.model_name == 'L_CNN':
        from networks import LightCNNModel
        model = LightCNNModel()
    else:
        logging.error(f"{args.model_name} non è tra i modelli creabili in networks.py")
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.debug(f'pytorch_total_params: {pytorch_total_params}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.debug(f"Sto usando il device {device} per addestrare il modello {path_ckpts}")

    print("CUDA disponibile:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Versione CUDA di PyTorch:", torch.version.cuda)
        print("Nome della GPU:", torch.cuda.get_device_name(0))
    model = model.to(device)

    ##### Optimizer #####
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    ##### Training #####
    y_min = torch.tensor(y_min).to(device)
    y_max = torch.tensor(y_max).to(device)

    train(args, model, train_loader, val_loader, y_min, y_max, optimizer, path_ckpts, device)

    ##### Testing #####
    
    del val_loader

    if args.model_name == 'CRNN':
        from networks import CRNNModel
        model = CRNNModel()
    elif args.model_name == 'CNN':
        from networks import CNNModel
        model = CNNModel()
    elif args.model_name == 'L_CNN':
        from networks import LightCNNModel
        model = LightCNNModel()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'pytorch_total_params: {pytorch_total_params}')
    model = model.to(device)
    model.load_state_dict(torch.load(path_ckpts))
    print(f'Loading best model from: {path_ckpts}')

    res, pred_all, true_all = test(args, model, test_loader, y_min, y_max, device)

    ##### Saving results #####
    res.to_csv(os.path.join(path_logs, 'res.csv'), index=True)
    print(res)
    np.save(os.path.join(path_logs,'pred_all.npy'), pred_all)
    np.save(os.path.join(path_logs,'true_all.npy'), true_all)
    with open(os.path.join(path_logs,'args.json'), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    ##### Saving onnx model ############
    model = model.to('cpu')
    model.eval()
    dummy = torch.randn(1, args.window_len , 1)   # batch fisso a 1, shape statica

    path_base, ext = os.path.splitext(path_ckpts)
    print (path_base)
    path_onnx = path_base + '.onnx'

    kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False

    torch.onnx.export(
        model, dummy, path_onnx,
        input_names=['input'], output_names=['output'],
        opset_version=13,
        do_constant_folding=True,
        **kwargs,
    )
    # ---- parametri di normalizzazione ----
    with open(os.path.join(path_logs, 'norm_params.json'), 'w') as f:
        json.dump({
            "window_size": args.window_len,
            "x_min": float(x_min), "x_max": float(x_max),
            "y_min": float(y_min), "y_max": float(y_max),
        }, f, indent=2)

    # ---- dati di calibrazione per la quantizzazione in esp-ppq ----
    x_train = torch.cat([x for x, _ in train_loader], dim=0)
    idx = np.random.choice(len(x_train), min(2048, len(x_train)), replace=False)
    np.save(os.path.join(path_logs, 'calib_x.npy'), x_train[idx].to(torch.float32))
    del train_loader
    del test_loader

    

    sess = ort.InferenceSession(path_onnx, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        y_torch = model(dummy).numpy()
    y_onnx = sess.run(None, {"input": dummy.numpy()})[0]
    logging.debug(f"Max difference between PyTorch and ONNX outputs: {np.abs(y_torch - y_onnx).max()}")   # deve essere ~1e-6