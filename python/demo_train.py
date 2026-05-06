import os
import csv
import argparse
import numpy as np
import time

import torch
from tools.data_loader import get_loader
from tools.misc import mkexperiment,save_torch_result
from evaluate_result import calculate_nmse
import sys
from tqdm import tqdm
from tools.evaluation import *
import traceback
from basicsr.archs.mambairv2_arch import MambaIRv2
from tools.save_config import save_training_config



def main(config):
    #-----GPU-----#
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_NUM
    torch.backends.cudnn.benchmark = True

    #-----random seed-----#
    np.random.seed(1)
    torch.manual_seed(1)
    time2 = 0

    # -----experiment-----#
    experiment_path = mkexperiment(config, cover=True)
    save_inter_result = os.path.join(experiment_path, 'inter_result')
    model_path = os.path.join(config.model_path,config.name)

    #-----dataloader-----#
    data_dir = config.data_dir
    train_batch = get_loader(data_dir, config, crop_key=config.CROP_KEY,num_workers=1, shuffle=True, mode='train')
    val_batch = get_loader(data_dir, config, crop_key=False,num_workers=1, shuffle=True, mode='test')
    brain_batch = get_loader(data_dir, config, crop_key=False,num_workers=1, shuffle=False, mode='brain')

    # -----model-----#
    net = MambaIRv2(img_size=config.INPUT_H,in_chans=config.INPUT_C,out_chans=config.OUTPUT_C)

    scaler = torch.cuda.amp.GradScaler()

    # -----lossfunc-----#
    criterion = torch.nn.MSELoss()   

    config_file = save_training_config(net,config,experiment_path,criterion)
    print(f"Training configuration saved to :{config_file}")

    if torch.cuda.is_available():
        net.cuda()
        criterion.cuda()

    # -----optim-----#
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr,betas=(config.beta1, config.beta2))

    #-----Setup device-----#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.mode =='train':
        total_iters = 0

        train_loss = 0; train_length = 0; val_loss = 0; val_nmse = 0
        val_ssim = 0; val_psnr = 0; val_pccs = 0; val_length = 0;val_mae=0
        pbar = tqdm(range(1,config.num_epochs+1), desc='', mininterval=1, file=sys.stdout)

        for epoch in pbar:

            # for epoch in range(1, config.num_epochs + 1):
            # ********************************************train*****************************************************#
            for i,(images, GT) in enumerate(train_batch):
                images, GT = images.type(torch.FloatTensor), GT.type(torch.FloatTensor)
                images, GT = images.to(device), GT.to(device)

                optimizer.zero_grad()  # clear grad

                with torch.cuda.amp.autocast():
                    SR = net(images)  # forward
                    loss = criterion(SR, GT)

                train_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer=optimizer)
                scaler.update()

                train_length += images.size(0)
                total_iters += 1

                # learing rate decay
                if (total_iters % config.lr_updata) == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.8

                if total_iters%config.step == 0:
                    lr = optimizer.param_groups[0]['lr']
                    # # *************************************validation**********************************************#
                    with torch.no_grad():
                        net.eval()
                        for i,(images_val, GT_val) in enumerate(val_batch):
                            images_val, GT_val = images_val.type(torch.FloatTensor), GT_val.type(torch.FloatTensor)
                            images_val, GT_val = images_val.to(device), GT_val.to(device)

                            SR_val = net(images_val)
                            loss_val = criterion(SR_val, GT_val)

                            mae_ = get_mae(SR_val[:,0:1,:,:], GT_val[:,0:1,:,:])
                            nmse_ = calculate_nmse(SR_val[:,0:1,:,:].cpu().detach().numpy(), GT_val[:,0:1,:,:].cpu().detach().numpy())
                            ssim_ = get_ssim(SR_val[:,0:1,:,:], GT_val[:,0:1,:,:])
                            psnr_ = get_psnr(SR_val[:, 0:1, :, :], GT_val[:, 0:1, :, :])

                            val_mae += mae_
                            val_nmse += nmse_
                            val_ssim += ssim_
                            val_psnr += psnr_
                            # val_pccs += pccs_
                            val_loss += loss_val.item()
                            val_length += images_val.size(0)

                        # Print the log info
                        time1 = time.perf_counter()
                        temp_time = time1-time2
                        sum_times = val_length//config.BATCH_SIZE if val_length%config.BATCH_SIZE==0 else \
                                    val_length//config.BATCH_SIZE+1
                        pbar_dict = {'Total_iters':total_iters, 'Train Loss':train_loss/train_length, 'Val Loss':val_loss/val_length,
                                     'Val Nmse':val_nmse/sum_times, 'Val SSIM':val_ssim/sum_times, 'Val PSNR':val_psnr/sum_times,
                                     'Val MAE':val_mae/sum_times, 'Val PCCs':val_pccs/sum_times, 'lr':lr, 'time':temp_time}
                        pbar.set_postfix(pbar_dict)


                        print()
                        time2 = time.perf_counter()

                        train_loss = 0; train_length = 0; val_loss = 0
                        val_nmse = 0; val_ssim = 0; val_psnr = 0; val_pccs = 0; val_length = 0; val_mae =0

                        ## ********************************************test***********************************************#
                        for i, (brain_images, _) in enumerate(brain_batch):
                            brain_images = brain_images.type(torch.FloatTensor)
                            brain_images = brain_images.to(device)
                            SR_brain = net(brain_images)

                            # save interresult in fold
                            save_dir = os.path.join(save_inter_result, 'inter_t2star_' + str(total_iters) + '_brain')
                            save_torch_result(SR_brain[:, 0:1, :, :], save_dir,
                                              format='png', cmap='jet', norm=False, crange=[0, 0.2])
                            del brain_images, SR_brain
                        net.train()

            # -----save_model-----#
            if (epoch) % config.model_save_step == 0 and epoch > config.model_save_start:
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                torch.save(net.state_dict(), model_path + '/' + config.name + '_epoch_' +str(epoch) + '.pth')

        f.close()
        # writer_train.close()
        # writer_val.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment name
    parser.add_argument('--name', type=str, default='experiment')
    parser.add_argument('--experiment_path', type=str, default='')
    parser.add_argument('--data_dir', type=str,default='')  ## data path
    parser.add_argument('--GPU_NUM', type=str, default='')

    # model hyper-parameters
    parser.add_argument('--INPUT_H', type=int, default=)
    parser.add_argument('--INPUT_W', type=int, default=)
    parser.add_argument('--INPUT_C', type=int, default=)
    parser.add_argument('--OUTPUT_C', type=int, default=1)
    parser.add_argument('--LABEL_start', type=int, default=3)
    parser.add_argument('--LABEL_end', type=int, default=3)
    parser.add_argument('--DATA_C', type=int, default=3)

    parser.add_argument('--CROP_KEY', type=bool, default=True)
    parser.add_argument('--CROP_SIZE', type=int, default=64)

    # training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--BATCH_SIZE', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_updata', type=int, default=80000)  # epoch num for lr updata
    parser.add_argument('--beta1', type=float, default=0.9)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

    parser.add_argument('--step', type=int, default=500)
    parser.add_argument('--model_save_start', type=int, default=1)
    parser.add_argument('--model_save_step', type=int, default=50)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='.')
    parser.add_argument('--result_path', type=str, default='')

    config = parser.parse_args()

    config.name = ''




