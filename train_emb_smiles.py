import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm.auto import tqdm
import torch_ema
import argparse
import matplotlib.pyplot as plt
import os
import torchvision
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
from torch.utils.data import Dataset
from bfns.bfn_continuous import BFNContinuousData
from bfns.bfn_discretised import BFNDiscretisedData
from networks.unet_emb_smiles import UNet
from utils import default_transform
import argparse
from enum import Enum, auto
from torch.utils.tensorboard import SummaryWriter


with open('bfn_exercise.pkl', 'rb') as file:
    data = pickle.load(file)
# data = data[:100000]


ls_emb_smiles = []
ls_logp = []

for i in data:
    ls_emb_smiles.append(i['emb_smiles'].reshape(1, -1))
    ls_logp.append(i['logp'])
    
emb_smiles_all = np.concatenate(ls_emb_smiles, axis=0)  
logp_all = np.array(ls_logp)


scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(emb_smiles_all)
emb_smiles_all_transform = scaler.transform(emb_smiles_all)


device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

class BFNType(Enum):
    Continuous = auto()
    Discretized = auto()

class TimeType(Enum):
    ContinuousTimeLoss = auto()
    DiscreteTimeLoss = auto()
    
class emb_smiles_dataset(Dataset):
    def __init__(self):
        self.emb_smiles_all_transform = emb_smiles_all_transform
        self.logp_all = logp_all
        
        # self.img_labels = pd.read_csv(annotations_file)
        # self.img_dir = img_dir
        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.emb_smiles_all_transform)

    def __getitem__(self, idx):
        emb_smiles = self.emb_smiles_all_transform[idx, :].reshape((1, -1))
        logp = self.logp_all[idx]
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return emb_smiles, logp
    

def train(args: argparse.ArgumentParser, bfnType: BFNType, timeType: TimeType):
    transform = default_transform(args.height, args.width)
    dataset = emb_smiles_dataset()
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch, shuffle=True, num_workers=8)
    
    if bfnType == BFNType.Continuous:
        unet = UNet(1, 1).to(device)
    elif bfnType == BFNType.Discretized:
        unet = UNet(3, 6).to(device)
    else:
        raise ValueError("The BFNType must be either Continuous or Discretized.")
    
    if args.load_model_path is not None:
        unet.load_state_dict(torch.load(args.load_model_path))
        
    if bfnType == BFNType.Continuous:
        bfn = BFNContinuousData(unet, sigma=args.sigma).to(device)
    elif bfnType == BFNType.Discretized:
        bfn = BFNDiscretisedData(unet, K=args.K, sigma=args.sigma).to(device)
    else:
        raise ValueError("The BFNType must be either Continuous or Discretized.")
    
    optimizer = AdamW(unet.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    ema = torch_ema.ExponentialMovingAverage(unet.parameters(), decay=0.9999)
    ema.to(device)
    writer = SummaryWriter()
    
    num = 1
    # for epoch in tqdm(range(1, args.epoch+1), desc='Training', unit='epoch'):
    for epoch in range(1, args.epoch+1):
        losses = []
        for X, _ in tqdm(train_loader, desc='Epoch {}'.format(epoch), unit='batch'):

            optimizer.zero_grad()
            if timeType == TimeType.ContinuousTimeLoss:
                loss = bfn.process_infinity(X.to(device))
            elif timeType == TimeType.DiscreteTimeLoss:
                loss = bfn.process_discrete(X.to(device), max_step=args.max_step)
            else:
                raise ValueError("The TimeType must be either ContinuousTimeLoss or DiscreteTimeLoss.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1)
            
            optimizer.step()
            ema.update()
            losses.append(loss.item())
            writer.add_scalar('Loss/train', loss.item(), num)
            num += 1
        scheduler.step()
        
        ave = 0
        for loss in losses:
            ave += loss
        ave = ave / len(losses)
        tqdm.write('Epoch {}: Loss: {:.8f}'.format(epoch, ave))
        writer.add_scalar('Loss/epoch_train', ave, epoch)
        torch.save(unet.state_dict(), f'models_all_data/model.pth')
        if epoch % args.save_every_n_epoch == 0:
            torch.save(unet.state_dict(), f'models_all_data/model_{epoch}.pth')
            tqdm.write('Epoch {}: saved: {}'.format(epoch, f'models_all_data/model_{epoch}.pth'))

    torch.save(unet.state_dict(), args.save_model_path)
    tqdm.write('Epoch {}: saved: {}'.format(epoch, args.save_model_path))

def setup_train_common_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--load_model_path", type=str, default="models_all_data_archive/model_10.pth")
    parser.add_argument("--save_model_path", type=str, default="models_all_data/model.pth")
    parser.add_argument("--save_every_n_epoch", type=int, default=1)
    parser.add_argument("--data_path", type=str, default="./animeface")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    # Discrete Time Loss Option
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--max_step", type=int, default=1000)
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = setup_train_common_parser(parser)

    args, unknown = parser.parse_known_args()
    train(args, bfnType=BFNType.Continuous, timeType=TimeType.ContinuousTimeLoss)