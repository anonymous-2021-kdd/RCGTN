import argparse
import time
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from model import RCGTN
import numpy as np
import pandas as pd


import utils
import os
from torch.nn import init

"""
dataset          traffic      electricity     pems03    pems04    pems07   pems08
granularity      1hour         1hour           1hour     1hour    1hour    1hour
samples          17554         26304           358       307      883      170
nodes            862           321             26208     16992    28224    17856
"""

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id',type=int,default=0,help='')
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument('--dataset',type=str,default='pems08',help='dataset')
parser.add_argument('--seq_len',type=int,default=12,help='')
parser.add_argument('--out_len',type=int,default=12,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--out_dim',type=int,default=1,help='outputs dimension')
parser.add_argument('--num_nodes',type=int,default=170,help='number of nodes')
parser.add_argument('--range_size',type=int,default=10,help='range size for group_range attention')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--n_layers',type=int,default=4,help='layers of encoder and decoder')
parser.add_argument('--d_model',type=int,default=16,help='dimension of model')
parser.add_argument('--end_channels',type=int,default=64,help='end_channels')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=10,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--save',type=str,default='./model_save/pems08/',help='save path')

args = parser.parse_args()


#set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)



def eval_epoch(model, dataloader, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    scaler = dataloader['scaler']
    compute_loss = utils.masked_mae

    print("validation...", flush=True)

    outputs = []
    real_y = torch.Tensor(dataloader['y_val']).to(device)
    real = real_y.transpose(1,3)[:,0,:,:] # (num_samples, input_dim, num_nodes, input_length) --> (num_samples, num_nodes, input_length)

    for iter, (x, y,trend) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        test_trend = torch.Tensor(trend).to(device)
        test_trend = test_trend.transpose(1,2)
        with torch.no_grad():
            preds = model(testx,test_trend) #output = [batch_size,1,num_nodes,12]
        outputs.append(preds.squeeze())
    yhat = torch.cat(outputs, dim=0)
    pred = yhat[:real_y.size(0), ...]
    pred = scaler.inverse_transform(pred)
    # pred = yhat

    loss = compute_loss(pred, real, 0.0)
    mae = loss.item()
    mape = utils.masked_mape(pred,real,0.0).item()
    rmse = utils.masked_rmse(pred,real,0.0).item()


    return mae, mape, rmse


def train_epoch(model, optimizer, dataloader, args, device):
    ''' Epoch operation in training phase'''

    model.train()
    clip = 5
    scaler = dataloader['scaler']
    compute_loss  = utils.compute_loss

    train_mae = []
    train_mape = []
    train_rmse = []

    dataloader['train_loader'].shuffle()
    for iter, (x, y, trend) in enumerate(dataloader['train_loader'].get_iterator()):

        # prepare data
        trainx = torch.Tensor(x).to(device)
        trainx = trainx.transpose(1, 3)
        trainy = torch.Tensor(y).to(device)
        trainy = trainy.transpose(1, 3)
        trainy = trainy[:, 0, :, :]
        train_trend = torch.Tensor(trend).to(device)
        train_trend = train_trend.transpose(1,2)

        #foward
        optimizer.zero_grad()
        output = model(trainx, train_trend) # history and future trend
        output = output.squeeze()
        # (num_samples,  num_nodes, out_len)

        predict = scaler.inverse_transform(output)


        #real = torch.unsqueeze(trainy, dim=1) #(num_samples, 1, num_nodes, input_length)
        # print('out_size',output.size())
        # print('real_size',real.size())
        real = trainy
        loss = compute_loss(predict, real, 0.0)
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        mae = loss.item()
        mape = utils.masked_mape(predict,real,0.0).item()
        rmse = utils.masked_rmse(predict,real,0.0).item()

        train_mae.append(mae)
        train_mape.append(mape)
        train_rmse.append(rmse)

        if iter % args.print_every == 0:
            log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            print(log.format(iter, train_mae[-1], train_mape[-1], train_rmse[-1]), flush=True)


    return train_mae, train_mape, train_rmse



def train(model, optimizer,dataloader, args, device):

    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    ''' Start training '''

    print("Start training...", flush=True)


    history_loss = []
    train_time = []
    for epoch_i in range(1, args.epochs + 1):
        print('[ Epoch', epoch_i, ']')

        start_time = time.time()

        train_mae, train_mape, train_rmse  = train_epoch(model, optimizer, dataloader, args, device)
        s1 = time.time()
        valid_mae, valid_mape, valid_rmse = eval_epoch(model,  dataloader, device)

        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)


        end_time = time.time()

        log = 'Epoch: {:03d}/{}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(epoch_i,args.epochs, mtrain_mae, mtrain_mape, mtrain_rmse, (end_time - start_time)),
              flush=True)
        log = 'Epoch: {:03d}/{}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Inference Time: {:.4f}/epoch'
        print(log.format(epoch_i,args.epochs, valid_mae, valid_mape, valid_rmse, (end_time - s1)),
              flush=True)

        save_path = args.save + "batch_size{}/".format(args.batch_size)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(model.state_dict(), save_path +"epoch_"+str(epoch_i)+"_"+str(round(valid_mae,2))+".pth")

        history_loss.append(valid_mae)
        train_time.append(end_time-start_time)
        curr_best_epoch = np.argmin(history_loss) +1
        if((epoch_i - curr_best_epoch)>20):        # early stopping
            break
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    if(epoch_i < args.epochs):
        print('early stop!')
    print("Training finished")
    best_epoch = np.argmin(history_loss) + 1
    print("The valid loss on best epoch{} is {}".format(best_epoch,str(round(history_loss[best_epoch-1], 4))))

    return history_loss


def test(model, dataloader, args, device, history_loss):

    ''' testing '''
    model.eval()

    scaler = dataloader['scaler']

    print("testing...", flush=True)

    best_epoch = np.argmin(history_loss)+1
    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    save_path = args.save + "batch_size{}/".format(args.batch_size)
    model.load_state_dict(torch.load(save_path +"epoch_"+str(best_epoch)+"_"+str(round(history_loss[best_epoch-1],2))+".pth"))

    outputs = []
    real_y = torch.Tensor(dataloader['y_test']).to(device)
    real_y = real_y.transpose(1,3)[:,0,:,:] # (num_samples, input_dim, num_nodes, input_length) --> (num_samples, num_nodes, input_length)

    for iter, (x, y,trend) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        test_trend = torch.Tensor(trend).to(device)
        test_trend = test_trend.transpose(1,2)
        with torch.no_grad():
            preds = model(testx,test_trend) #output = [batch_size,1,num_nodes,12]
        outputs.append(preds.squeeze())
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:real_y.size(0), ...]
    print(yhat.size())
    print(real_y.size())

    for i in range(real_y.size(2)):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = real_y[:, :, i]
        metrics = utils.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}%, Test RMSE: {:.4f}'
        print(log.format(i+1 , metrics[0], metrics[1]*100, metrics[2]))

    torch.save(model.state_dict(),
               save_path + "exp" + str(args.expid) + "_best_" + str(round(history_loss[best_epoch-1], 2)) + ".pth")

    print(' ')
    if(args.out_len>10):
        for i in [2,5,11]:
            pred = scaler.inverse_transform(yhat[:, :, i])
            real = real_y[:, :, i]
            metrics = utils.metric(pred, real)
            log = 'Evaluate for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}%, Test RMSE: {:.4f}'
            print(log.format(i+1 , metrics[0], metrics[1]*100, metrics[2]))

    pred = scaler.inverse_transform(yhat)
    metrics = utils.metric(pred, real_y)
    log = 'Average:, Test MAE: {:.4f}, Test MAPE: {:.4f}%, Test RMSE: {:.4f}'
    print(log.format(metrics[0], metrics[1]*100, metrics[2]))

    return pred


if __name__ == "__main__":
    t1 = time.time()

    args.data = './data/{}/'.format(args.dataset)
    args.save = './garage/{}/'.format(args.dataset)
    device = torch.device('cuda:{}'.format(args.gpu_id))



    dataloader = utils.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)


    model = RCGTN(
        in_dim=args.in_dim, out_dim=args.out_dim, seq_len=args.seq_len, out_len=args.out_len, end_channels = args.end_channels,d_inner=args.d_model,
        en_layers=args.n_layers,  de_layers=args.n_layers, d_model=args.d_model,
        num_nodes=args.num_nodes, range_size=args.range_size, dropout=args.dropout,if_ffn=True
    )
    if args.multi_gpu:
        model = nn.DataParallel(model,device_ids=[args.gpu_id,args.gpu_id+1])
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr= args.learning_rate, weight_decay=args.weight_decay)

    history_loss = train(model, optimizer, dataloader, args, device)
    test(model, dataloader, args, device, history_loss)


    t2 = time.time()
    print("Total time spent: {:.4f} minutes".format((t2-t1)/60))














