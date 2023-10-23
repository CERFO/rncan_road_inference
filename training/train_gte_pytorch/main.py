import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ' '
import torch
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random

def main(config):
    try:
        cudnn.benchmark = True
        if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net','R2AttU_Net_CERFO']:
            err_msg = ('Error, model_type should be selected in U_Net, R2U_Net, AttU_Net, R2AttU_Net, R2AttU_Net_CERFO'
                       f'Your input for model_type was {config.model_type}')
            raise ValueError(err_msg)

        print(config)

        # Create directories if not exist
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)
        if not os.path.exists(config.log_path):
            os.makedirs(config.log_path)

        train_loader = get_loader(
            image_path=config.train_path,
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            mode='train'
        )
        valid_loader = get_loader(
            image_path=config.valid_path,
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            mode='valid'
        )

        solver = Solver(config, train_loader, valid_loader)

        # Train and sample the images
        if config.mode.lower() == 'train':
            solver.train()
        else:
            raise NotImplementedError('Test is not implemented')

    except Exception as ex:
        print(f"An error occured based on the input parameters: {ex}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=4)
    parser.add_argument('--output_ch', type=int, default=19)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--beta1', type=float, default=0.5, help="Momentum1 in Adam")
    parser.add_argument('--beta2', type=float, default=0.999, help="Momentum2 in Adam")

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='R2AttU_Net_CERFO'
                        , help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/R2AttU_Net_CERFO')
    parser.add_argument('--log_path', type=str, required=True, default=None)
    parser.add_argument('--exp_name', type=str, required=True, default=None)
    parser.add_argument('--model_path', type=str, required=True, default=None)
    parser.add_argument('--train_path', type=str, required=True, default=None)
    parser.add_argument('--valid_path', type=str, required=True, default=None)
    parser.add_argument('--cuda_idx', type=int, default=0)

    config = parser.parse_args()
    main(config)
