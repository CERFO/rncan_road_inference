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
from losses import SoftmaxOutput, SupervisedLoss, WarmUpCosineDecayScheduler, NormalizeOutput, SupervisedLossCompact
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
                log_file.write('epoch,train_epoch_loss,train_keypoint_prob_loss,train_direction_prob_loss'
                               ',train_direction_vector_loss,val_epoch_loss,val_keypoint_prob_loss'
                               ',val_direction_prob_loss,val_direction_vector_loss\n')
        
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        # self.criterion = SupervisedLoss()
        self.criterion = SupervisedLossCompact()

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        
        # Learning rate scheduler
        warmup_learning_rate = 0.0
        warmup_epoch = 4
        hold_base_rate_epoch = 6
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

        # Optimizer
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
            print('%s already exists here! :%s'%(self.model_type,self.model_path))
        else:
            # Train for Encoder
            lr = self.lr
            best_unet_score = 0.
            
            for epoch in range(self.num_epochs):

                self.unet.train(True)
                
                train_epoch_loss = 0
                keypoint_prob_loss = 0
                direction_prob_loss = 0
                direction_vector_loss = 0
                length = 0  # Number of iteration
                epoch_start_time = time.time()

                for i, (images, GT) in enumerate(self.train_loader):
                    # GT : Ground Truth
                    images = images.to(self.device)
                    GT = GT.to(self.device)

                    # SR : Segmentation Result
                    SR = self.unet(images)
                    # SR = SoftmaxOutput(SR)
                    SR = NormalizeOutput(SR)

                    # Loss
                    batch_keypoint_prob_loss, batch_direction_prob_loss, batch_direction_vector_loss = self.criterion(SR,GT)
                    loss = batch_keypoint_prob_loss + batch_direction_prob_loss + batch_direction_vector_loss
                    train_epoch_loss += loss.item()

                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 2.0)
                    self.optimizer.step()

                    # Step metrics
                    keypoint_prob_loss += batch_keypoint_prob_loss.item()
                    direction_prob_loss += batch_direction_prob_loss.item()
                    direction_vector_loss += batch_direction_vector_loss.item()
                    length += 1
                    
                    # Decay learning rate
                    lr = self.lr_scheduler.get_batch_lr()
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

                # Compute epoch metrics
                train_epoch_loss /= length
                train_keypoint_prob_loss = keypoint_prob_loss/length
                train_direction_prob_loss = direction_prob_loss/length
                train_direction_vector_loss = direction_vector_loss/length
                
                
                #===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()

                val_epoch_loss = 0
                keypoint_prob_loss = 0
                direction_prob_loss = 0
                direction_vector_loss = 0
                length = 0
                
                for i, (images, GT) in enumerate(self.valid_loader):
                    # GT : Ground Truth
                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    
                    # SR : Segmentation Result
                    SR = self.unet(images)
                    # SR = SoftmaxOutput(SR)
                    SR = NormalizeOutput(SR)

                    # Loss
                    batch_keypoint_prob_loss, batch_direction_prob_loss, batch_direction_vector_loss = self.criterion(SR,GT)
                    loss = batch_keypoint_prob_loss + batch_direction_prob_loss + batch_direction_vector_loss
                    val_epoch_loss += loss.item()

                    # Step metrics
                    keypoint_prob_loss += batch_keypoint_prob_loss.item()
                    direction_prob_loss += batch_direction_prob_loss.item()
                    direction_vector_loss += batch_direction_vector_loss.item()
                    length += 1

                # Compute epoch metrics
                val_epoch_loss /= length
                val_keypoint_prob_loss = keypoint_prob_loss/length
                val_direction_prob_loss = direction_prob_loss/length
                val_direction_vector_loss = direction_vector_loss/length


                #===================================== Log values ====================================#
                # Print the log info
                epoch_time = round(time.time() - epoch_start_time)
                print('Epoch %d/%d - %ds - loss: %.4f - key_prob_loss: %.4f - dir_prob_loss: %.4f - dir_vec_loss: %.4f - val_loss: %.4f - val_key_prob_loss: %.4f - val_dir_prob_loss: %.4f - val_dir_vec_loss: %.4f' % (
                      epoch+1, self.num_epochs, epoch_time, \
                      train_epoch_loss, train_keypoint_prob_loss, train_direction_prob_loss, train_direction_vector_loss, \
                      val_epoch_loss, val_keypoint_prob_loss, val_direction_prob_loss, val_direction_vector_loss))
                
                # Write values to log file
                with open(self.log_path, 'a+') as log_file:
                    log_file.write(str(epoch) + ',' + str(train_epoch_loss) + ',' + str(train_keypoint_prob_loss) + ',' + str(train_direction_prob_loss) + ',' + str(train_direction_vector_loss) + ',' + str(val_epoch_loss) + ',' + str(val_keypoint_prob_loss) + ',' + str(val_direction_prob_loss) + ',' + str(val_direction_vector_loss) + '\n')
                
                # Save Best U-Net model
                model_to_save = self.unet.state_dict()
                torch.save(model_to_save,self.model_path)
