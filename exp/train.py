import os
import torch
from utils.dataset import PriceDataset
from utils.tools import EarlyStopping, adjust_learning_rate
from torch import optim, nn
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import DataLoader
def vali(model, args, vali_loader, criterion):
    total_loss = []
    model.eval()
    device = torch.device('cuda')
    with torch.no_grad():
        for i, (batch_x, batch_y, _, _) in enumerate(vali_loader):
            batch_x = batch_x.unsqueeze(2).to(device)
            batch_y = batch_y.unsqueeze(2).to(device)
            outputs = model(batch_x)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            loss = criterion(pred, true)
            total_loss.append(loss)
    total_loss = np.average(total_loss)
    model.train()
    return total_loss

def train(args, model, data):
    df = data['Price'].values.astype(np.float32)
    df_date = data['Datetime'].values
    train_dataset = PriceDataset(args, df, df_date, mode='train')
    vali_dataset = PriceDataset(args, df, df_date, mode='val')
    test_dataset = PriceDataset(args, df, df_date, mode='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
    vali_loader = DataLoader(vali_dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)

    path = args.checkpoints_path + '/' + args.model_type 
    if not os.path.exists(path):
        os.makedirs(path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience)
    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.SmoothL1Loss()

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        model.train()
        for i, (batch_x, batch_y, _, _) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().unsqueeze(2).to(device)
            batch_y = batch_y.float().unsqueeze(2).to(device).to(device)

            outputs = model(batch_x).to(device)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            loss.backward()
            model_optim.step()

        train_loss = np.average(train_loss)
        vali_loss = vali(model, args, vali_loader, criterion)
        test_loss = vali(model, args, test_loader, criterion)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))

    return model