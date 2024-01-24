import os, warnings, random, gc
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from math import isnan
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

from dataloaders import make_semi_loader
from models.resnet_proposed import resnet50

from utils.saver import Saver
from utils.tqdm_config import get_tqdm_config


class SimRegMatchTrainer(object):
    def __init__(self, args):
        self.args = args
        
        self.saver = Saver(self.args)
        self.saver.save_experiment_config(self.args)
        self.experiment_dir = self.saver.experiment_dir
        print(self.experiment_dir)
        
        self.result_dir = os.path.join(self.experiment_dir, 'csv')
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.experiment_dir)
        self.labeled_loader, self.unlabeled_loader, self.valid_loader, self.test_loader = \
            make_semi_loader(self.args, num_workers=0)
        
        self.model = resnet50(dropout=self.args.dropout).to(self.args.cuda)
        
        if self.args.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.args.lr)
        elif self.args.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                              lr=self.args.lr,
                                              momentum=self.args.momentum,
                                              weight_decay=self.args.weight_decay)

        if self.args.loss=='mse':
            self.criterion = nn.MSELoss().to(self.args.cuda)
            self.criterion_unlabel = nn.MSELoss(reduction='none').to(self.args.cuda)
        elif self.args.loss=='l1':
            self.criterion = nn.L1Loss().to(self.args.cuda)
            self.criterion_unlabel = nn.L1Loss(reduction='none').to(self.args.cuda)
        
        self.args.best_valid_loss = np.inf
        self.args.best_valid_epoch = 0
        self.cnt_train, self.cnt_valid = 0, 0

    def train(self, epoch):
        self.model.train()
        losses_t, losses_l, losses_u = 0.0, 0.0, 0.0
        iter_unlabeled = iter(self.unlabeled_loader)
        
        total_steps = len(self.labeled_loader.dataset) // self.args.batch_size
        with tqdm(**get_tqdm_config(total=total_steps,
                  leave=True, color='green')) as pbar:
            for idx, samples_labeled in enumerate(self.labeled_loader):
                # Load data
                inputs_l = samples_labeled['input'].to(self.args.cuda)
                labels_l = samples_labeled['label'].to(self.args.cuda)

                try:
                    samples_unlabeled = next(iter_unlabeled)
                except:
                    iter_unlabeled = iter(self.unlabeled_loader)
                    samples_unlabeled = next(iter_unlabeled)

                inputs_u = samples_unlabeled['weak'].to(self.args.cuda)
                inputs_s = samples_unlabeled['strong'].to(self.args.cuda)
                
                # Predict labeled examples 
                preds_x, vecs_x = self.model(inputs_l)

                # Predict strong-augmented examples 
                preds_w, vecs_w = [], []
                for _ in range(self.args.iter_u):
                    tmp_preds, tmp_vecs = self.model(inputs_u)
                    preds_w.append(tmp_preds.unsqueeze(-1))
                    vecs_w.append(tmp_vecs.unsqueeze(-1))
                
                preds_w = torch.cat(preds_w, dim=-1)
                vecs_w = torch.cat(vecs_w, dim=-1)
                
                # Predict strong-augmented examples
                preds_s, _ = self.model(inputs_l)
                
                # loss calculation for labeled examples
                loss_x = self.criterion(preds_x, labels_l)
                
                # loss calculation for unlabeled examples
                v_mean = torch.mean(preds_w, axis=2) # pseudo labeling based on our model
                vecs_w = torch.mean(vecs_w, axis=2)

                # similarity distribution
                vecs_w, vecs_x = F.normalize(vecs_w, dim=1), F.normalize(vecs_x, dim=1) 
                similaritys = vecs_w @ vecs_x.T
                similaritys = torch.softmax(similaritys/self.args.t, dim=1)
                
                # similarity-based pseudo-label
                v_similarity = similaritys @ labels_l
                
                # pseudo-label calibration
                v_mean = self.args.beta*v_mean + (1-self.args.beta)*v_similarity # beta*modelPL + (1-beta)*simPL
                
                # uncertainty estimation
                v_uncertainty = torch.pow(torch.std(preds_w, axis=2), 2)
                v_uncertainty = torch.sum(v_uncertainty, axis=1)
                
                # pseudo-label filtering
                mask = (v_uncertainty < self.args.threshold)
                loss_u = self.criterion_unlabel(v_mean.detach(), preds_s).sum(axis=1)
                loss_u = (loss_u * mask).sum()/(int(mask.sum()))
                
                self.args.threshold = np.percentile(v_uncertainty.detach().cpu().numpy(), q=self.args.percentile)           
                self.writer.add_scalar(
                    'Threshold',
                    self.args.threshold,
                    global_step=self.cnt_train
                )
                           
                if isnan(loss_u.item()):
                    loss_u = torch.tensor(0).to(self.args.cuda)
                
                loss = loss_x + self.args.lambda_u*loss_u

                del(inputs_s, inputs_u, preds_w, preds_s)
                gc.collect()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                losses_t += loss.item()
                losses_l += loss_x.item()
                losses_u += loss_u.item()
                
                self.writer.add_scalars(
                    'Train steps',
                    {'Total Loss': losses_t/(idx+1),
                    'Labeled Loss': losses_l/(idx+1),
                    'Unlabeled Loss': losses_u/(idx+1)},
                    global_step=self.cnt_train
                )
                self.cnt_train += 1
                
                description = "%-7s(%3d/%3d) Total: %.4f| Labeled: %.4f| Unlabeled: %.4f"%(
                    'SEMI', idx, total_steps, losses_t/(idx+1), losses_l/(idx+1), losses_u/(idx+1)
                )
                pbar.set_description(description)
                pbar.update(1)  

    def validation(self, epoch):
        self.model.eval()
        losses_t = 0.0
        
        total_steps = len(self.valid_loader.dataset) // self.args.batch_size + 1
        with tqdm(**get_tqdm_config(total=total_steps,
                                    leave=True, color='blue')) as pbar:
            for idx, samples in enumerate(self.valid_loader):
                inputs_l = samples['input'].to(self.args.cuda)
                labels_l = samples['label'].to(self.args.cuda)
                
                preds, _ = self.model(inputs_l)
                loss = self.criterion(preds, labels_l)
                                
                losses_t += loss.item()
                
                if idx == 0:
                    labels_total = labels_l.detach().cpu()
                    preds_total = preds.detach().cpu()
                else:
                    labels_total = torch.cat((labels_total, labels_l.detach().cpu()), dim=0)
                    preds_total = torch.cat((preds_total,
                                             preds.detach().cpu()), dim=0)

                r2, mae, rmse = self.regression_metrics(labels_total, preds_total)
                self.writer.add_scalars(
                    'Validation steps',
                    {'Loss': losses_t/(idx+1),
                     'MAE': mae,
                     'RMSE': rmse,
                     'R2': r2},
                    global_step=self.cnt_valid
                )
                self.cnt_valid += 1

                desc = "%-7s(%5d/%5d) Loss: %.4f| R^2: %.4f| MAE: %.4f| RMSE: %.4f "%("Valid", idx, total_steps, losses_t/(idx+1), r2, mae, rmse)
                pbar.set_description(desc)
                pbar.update(1)

            desc = "%-7s(%5d/%5d) Loss: %.4f| R^2: %.4f| MAE: %.4f| RMSE: %.4f "%("Valid", epoch, self.args.epochs, losses_t/(idx+1), r2, mae, rmse)
            pbar.set_description(desc)

            losses_t /= (idx+1)
            if self.args.best_valid_loss > losses_t:
                self.args.best_valid_loss = losses_t
                self.args.best_valid_epoch = epoch
                
                self.args.valid_r2 = str(r2)
                self.args.valid_mae = str(mae)
                self.args.valid_rmse = str(rmse)
                
                labels_total, preds_total = labels_total.numpy(), preds_total.numpy()

                labels_total, preds_total = pd.DataFrame(labels_total), pd.DataFrame(preds_total)
                df = pd.concat([labels_total, preds_total], axis=1)
                df.columns = ['Real', 'Pred']
                
                df.to_csv(os.path.join(self.result_dir, f'valid_{str(epoch)}.csv'), index=False)
                torch.save(self.model.state_dict(),
                    os.path.join(self.experiment_dir, 'best_model.pth'))
                
                self.saver.save_experiment_config(self.args)

    def inference(self, epoch):
        weight = torch.load(os.path.join(os.path.join(self.experiment_dir, 'best_model.pth')), map_location=self.args.cuda)
        self.model.load_state_dict(weight)
        
        self.model.eval()
        losses_t = 0.0
        
        total_steps = len(self.test_loader.dataset) // self.args.batch_size +1
        with tqdm(**get_tqdm_config(total=total_steps,
                                    leave=True, color='red')) as pbar:
            for idx, samples in enumerate(self.test_loader):
                inputs_l = samples['input'].to(self.args.cuda)
                labels_l = samples['label'].to(self.args.cuda)
                
                preds, _ = self.model(inputs_l)
                loss = self.criterion(preds, labels_l)
                                
                losses_t += loss.item()
                
                if idx == 0:
                    labels_total = labels_l.detach().cpu()
                    preds_total = preds.detach().cpu()
                else:
                    labels_total = torch.cat((labels_total, labels_l.detach().cpu()), dim=0)
                    preds_total = torch.cat((preds_total,
                                             preds.detach().cpu()), dim=0)

                r2, mae, rmse = self.regression_metrics(labels_total, preds_total)
                desc = "%-7s(%5d/%5d) Loss: %.4f| R^2: %.4f| MAE: %.4f| RMSE: %.4f "%("Test", idx, total_steps, losses_t/(idx+1), r2, mae, rmse)
                pbar.set_description(desc)
                pbar.update(1)

            desc = "%-7s(%5d/%5d) Loss: %.4f| R^2: %.4f| MAE: %.4f| RMSE: %.4f "%("Test", epoch, self.args.epochs, losses_t/(idx+1), r2, mae, rmse)
            pbar.set_description(desc)

            r2, mae, rmse = self.regression_metrics(labels_total, preds_total)
            losses_t /= (idx+1)
            
            self.args.best_test_loss = losses_t
            
            self.args.test_r2 = str(r2)
            self.args.test_mae = str(mae)
            self.args.test_rmse = str(rmse)
            
            labels_total, preds_total = labels_total.numpy(), preds_total.numpy()

            labels_total, preds_total = pd.DataFrame(labels_total), pd.DataFrame(preds_total)
            df = pd.concat([labels_total, preds_total], axis=1)
            df.columns = ['Real', 'Pred']
            
            df.to_csv(os.path.join(self.result_dir, f'test_{str(epoch)}.csv'), index=False)
            
            self.saver.save_experiment_config(self.args)

    @staticmethod
    def regression_metrics(reals, preds):
        reals, preds = reals.numpy(), preds.numpy()
        
        r2, mae = r2_score(reals, preds), mean_absolute_error(reals, preds)
        rmse = np.sqrt(mean_squared_error(reals, preds))
        
        return r2, mae, rmse
