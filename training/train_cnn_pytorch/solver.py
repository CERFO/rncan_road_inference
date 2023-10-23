import os
import numpy as np
import time
import datetime
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net,R2AttU_Net_CERFO,init_weights
from losses import DiceFocalLoss, WarmUpCosineDecayScheduler
import csv
import torch_optimizer
from pytorch_model_summary import summary


class Solver(object):
    def __init__(self, config, train_loader, valid_loader):

        # Log file
        self.log_path = config.log_path + config.exp_name + '.log'
        read_mode = 'r+' if os.path.exists(self.log_path) else 'w+'
        with open(self.log_path, read_mode) as log_file:
            if log_file.read() == '':
                log_file.write('epoch,DiceCoef,accuracy,loss,val_DiceCoef,val_accuracy,val_loss\n')
        
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        # self.criterion = torch.nn.BCELoss()
        self.criterion = DiceFocalLoss()

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        
        # Learning rate scheduler
        warmup_learning_rate = 0.0
        warmup_epoch = 6
        hold_base_rate_epoch = 14
        sample_count = len(self.train_loader) * self.batch_size
        total_steps = np.floor(self.num_epochs * sample_count / self.batch_size)
        warmup_steps = np.floor(warmup_epoch * sample_count / self.batch_size)
        hold_base_rate_steps = np.floor(hold_base_rate_epoch * sample_count / self.batch_size)
        self.lr_scheduler = WarmUpCosineDecayScheduler(
            learning_rate_base=self.lr,
            total_steps=total_steps,
            warmup_learning_rate=warmup_learning_rate,
            warmup_steps=warmup_steps,
            hold_base_rate_steps=hold_base_rate_steps,
            )

        # Path
        self.model_path = config.model_path + config.exp_name + '.hdf5'
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

    def build_model(self):
        """Build Unet"""
        if self.model_type =='U_Net':
            self.unet = U_Net(img_ch=self.img_ch,output_ch=self.output_ch)
        elif self.model_type =='R2U_Net':
            self.unet = R2U_Net(img_ch=self.img_ch,output_ch=self.output_ch,t=self.t)
        elif self.model_type =='AttU_Net':
            self.unet = AttU_Net(img_ch=self.img_ch,output_ch=self.output_ch)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=self.img_ch,output_ch=self.output_ch,t=self.t)
        elif self.model_type == 'R2AttU_Net_CERFO':
            self.unet = R2AttU_Net_CERFO(img_ch=self.img_ch,output_ch=self.output_ch,t=self.t)
        # summary(self.unet, torch.zeros((1, 4, 512, 512)).to(self.device), max_depth=None, print_summary=True)

        # weights initialization
        # init_weights(self.unet, init_type='kaiming')

        # Optimizer
        # self.optimizer = optim.Adam(list(self.unet.parameters()), self.lr, [self.beta1, self.beta2])
        self.optimizer = torch_optimizer.Ranger(list(self.unet.parameters()))
        
        # To GPU(s)
        self.unet = torch.nn.DataParallel(self.unet)
        self.unet.to(self.device)

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()


    def train(self):
        """Train encoder, generator and discriminator."""

        #====================================== Training ===========================================#
        #===========================================================================================#
        
        # U-Net Train
        if os.path.isfile(self.model_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(self.model_path))
            print('%s is Successfully Loaded from %s'%(self.model_type,self.model_path))
        else:
            # Train for Encoder
            lr = self.lr
            best_unet_score = 0.
            
            for epoch in range(self.num_epochs):

                self.unet.train(True)
                
                train_epoch_loss = 0
                acc = 0.    # Accuracy
                DC = 0.     # Dice Coefficient
                length = 0  # Number of iteration
                epoch_start_time = time.time()

                for i, (images, GT) in enumerate(self.train_loader):
                    # GT : Ground Truth
                    images = images.to(self.device)
                    GT = GT.to(self.device)

                    # SR : Segmentation Result
                    SR = self.unet(images)
                    SR = F.sigmoid(SR)

                    # Loss
                    loss = self.criterion(SR,GT)
                    train_epoch_loss += loss.item()

                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                    self.optimizer.step()

                    # Step metrics
                    acc += float(get_accuracy(SR,GT))
                    DC += float(dice_coef(SR,GT))
                    length += 1
                    
                    # Decay learning rate
                    lr = self.lr_scheduler.get_batch_lr()
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

                # Compute epoch metrics
                train_epoch_loss /= length
                train_acc = acc/length
                train_DC = DC/length
                
                
                #===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()

                val_epoch_loss = 0
                acc = 0.    # Accuracy
                DC = 0.     # Dice Coefficient
                length=0
                for i, (images, GT) in enumerate(self.valid_loader):
                    # GT : Ground Truth
                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    
                    # SR : Segmentation Result
                    SR = self.unet(images)
                    SR = F.sigmoid(SR)

                    # Loss
                    loss = self.criterion(SR,GT)
                    val_epoch_loss += loss.item()

                    # Step metrics
                    acc += float(get_accuracy(SR,GT))
                    DC += float(dice_coef(SR,GT))
                    length += 1

                # Compute epoch metrics
                val_epoch_loss /= length
                val_acc = acc/length
                val_DC = DC/length


                #===================================== Log values ====================================#
                # Print the log info
                epoch_time = round(time.time() - epoch_start_time)
                print('Epoch %d/%d - %ds - loss: %.4f - accuracy: %.4f - DiceCoef: %.4f - val_loss: %.4f - val_accuracy: %.4f - val_DiceCoef: %.4f' % (
                      epoch+1, self.num_epochs, epoch_time, \
                      train_epoch_loss, train_acc, train_DC, \
                      val_epoch_loss, val_acc, val_DC))
                
                # Write values to log file
                with open(self.log_path, 'a+') as log_file:
                    log_file.write(str(epoch) + ',' + str(train_DC) + ',' + str(train_acc) + ',' + str(train_epoch_loss) + ',' + str(val_DC) + ',' + str(val_acc) + ',' + str(val_epoch_loss) + '\n')
                
                # Save Best U-Net model
                # unet_score = val_DC
                # if unet_score > best_unet_score:
                    # best_unet_score = unet_score
                    # best_epoch = epoch
                    # best_unet = self.unet.state_dict()
                model_to_save = self.unet.state_dict()
                torch.save(model_to_save,self.model_path)
                
                
                
                
            

            
