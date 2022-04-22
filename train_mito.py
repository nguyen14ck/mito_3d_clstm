# %%
# region IMPORT ###########################################

# %%
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torchinfo import summary

# import warnings 
# warnings.filterwarnings("ignore")

# %%
from src.data.data_demo_r import get_data
from src.models.segmentation.unet3d_monai_clstm_dep3 import Unet3D_CLSTM
from src.loss.segmentation.loss_mito import BCELoss
from src.loss.segmentation.loss_torch_2 import Semantic_loss_functions
from src.utils.utils_meta import set_seed


# endregion IMPORT


# %%
# SEEDING & DEVICE
seed = 1
set_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





# %%
# region TRAIN ############################################

# %%
# TENSORBOARD WRITER
model_name = 'mito_train_cloud'
writer = SummaryWriter('runs/' + model_name)


# %%
# TRAIN FUCTION
def train(train_loader, val_loader, model, lfn_1, lfn_2, optimizer, scheduler, nepochs):
    val_interval = 1
    best_metric = float('inf')
    best_metric_epoch = -1

    train_len = len(train_loader)
    val_len = len(val_loader)

    k1 = 0
    k2 = 0

    for epoch in range(nepochs):
        
        print("\n" + "==" * 40)
        print(f"Train epoch {epoch + 1}/{nepochs}")
        model.train()
        train_epoch_nums = torch.zeros(4)

        if epoch < 3:
            k1 = 1
            k2 = 1
        else:
            k1 = 1
            k2 = 1
            

        for batch_idx, inputs in enumerate(train_loader):

            inputs, labels = inputs["image"].to(device), inputs["labels"].to(device)

            outputs = model(inputs)
            loss_1 = lfn_1(outputs, labels)
            loss_2, dice_score = lfn_2.log_cosh_dice_loss_2(outputs, labels)
            if dice_score > 1:
                loss_2, dice_score = lfn_2.log_cosh_dice_loss_2(outputs, labels)
            loss = k1*loss_1 + k2*loss_2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_nums[0] += loss.item()
            train_epoch_nums[1] += loss_1.item()
            train_epoch_nums[2] += loss_2.item()
            train_epoch_nums[3] += dice_score.item()

            print(f"Epoch {epoch + 1}/{nepochs}, Batch {batch_idx+1}/{train_len}, Train: loss={loss.item():.4f}, "
                                                f"loss_bce={loss_1.item():.4f}, loss_lcd={loss_2.item():.4f}, "
                                                f"dice_score={dice_score.item():.4f}")
            writer.add_scalar("Train/train_loss", loss.item(), train_len * epoch + batch_idx+1)
            writer.add_scalar("Train/train_loss_bce", loss_1.item(), train_len * epoch + batch_idx+1)
            writer.add_scalar("Train/train_loss_lcd", loss_2.item(), train_len * epoch + batch_idx+1)
            writer.add_scalar("Train/train_dice", dice_score.item(), train_len * epoch + batch_idx+1)


        train_epoch_nums /= train_len
        print(f"Epoch {epoch + 1} TRAIN AVG LOSS: {train_epoch_nums[0]:.4f}, "
                                f"LOSS_BCE: {train_epoch_nums[1]:.4f}, LOSS_LCD: {train_epoch_nums[2]:.4f}, "
                                f"DICE: {train_epoch_nums[3]:.4f}")
        writer.add_scalar("Train/Train_AVG_Loss", train_epoch_nums[0], epoch + 1)
        writer.add_scalar("Train/Train_AVG_Loss_BCE", train_epoch_nums[1], epoch + 1)
        writer.add_scalar("Train/Train_AVG_Loss_LCD", train_epoch_nums[2], epoch + 1)
        writer.add_scalar("Train/Train_AVG_Dice", train_epoch_nums[3], epoch + 1)


        # --------------------------------------------------------
        # VALIDATION

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():

                print('-'*15 + ' VALIDATION ' + '-'*15)
                val_epoch_nums = torch.zeros(4)

                for batch_idx, inputs in enumerate(val_loader):

                    inputs, labels = inputs["image"].to(device), inputs["labels"].to(device)

                    outputs = model(inputs)
                    loss_1 = lfn_1(outputs, labels)
                    loss_2, dice_score = lfn_2.log_cosh_dice_loss_2(outputs, labels)
                    loss = k1*loss_1 + k2*loss_2

                    val_epoch_nums[0] += loss.item()
                    val_epoch_nums[1] += loss_1.item()
                    val_epoch_nums[2] += loss_2.item()
                    val_epoch_nums[3] += dice_score.item()

                    print(f"Epoch {epoch + 1}/{nepochs}, Batch {batch_idx+1}/{val_len}, Validation: loss={loss.item():.4f}, "
                                                        f"loss_bce={loss_1.item():.4f}, loss_lcd={loss_2.item():.4f}, "
                                                        f"dice_score={dice_score.item():.4f}")
                    writer.add_scalar("Validation/validation_loss", loss.item(), val_len * epoch + batch_idx+1)
                    writer.add_scalar("Validation/validation_loss_bce", loss_1.item(), val_len * epoch + batch_idx+1)
                    writer.add_scalar("Validation/validation_loss_lcd", loss_2.item(), val_len * epoch + batch_idx+1)
                    writer.add_scalar("Validation/validation_dice", dice_score.item(), val_len * epoch + batch_idx+1)

                val_epoch_nums /= val_len
                
                print(f"Epoch {epoch + 1} TRAIN AVG LOSS: {train_epoch_nums[0]:.4f}, "
                                f"LOSS_BCE: {train_epoch_nums[1]:.4f}, TRAIN AVG LOSS_LCD: {train_epoch_nums[2]:.4f}, "
                                f"DICE: {train_epoch_nums[3]:.4f}")
                print(f"Epoch {epoch + 1} VALIDATION AVG LOSS: {val_epoch_nums[0]:.4f}, "
                                f"LOSS_BCE: {val_epoch_nums[1]:.4f}, LOSS_LCD: {val_epoch_nums[2]:.4f}, "
                                f"DICE: {val_epoch_nums[3]:.4f}")
                writer.add_scalar("Validation/Validation_AVG_Loss", val_epoch_nums[0], epoch + 1)
                writer.add_scalar("Validation/Validation_AVG_Loss_BCE", val_epoch_nums[1], epoch + 1)
                writer.add_scalar("Validation/Validation_AVG_Loss_LCD", val_epoch_nums[2], epoch + 1)
                writer.add_scalar("Validation/Validation_AVG_Dice", val_epoch_nums[3], epoch + 1)


                model_file = ''
                if val_epoch_nums[0]<=0.15:
                    if val_epoch_nums[0]<best_metric:
                        best_metric = val_epoch_nums[0]
                        best_metric_epoch = epoch + 1
                    # model_file = 'checkpoints/' + model_name + '_best.pth'
                    model_file = 'checkpoints/' + model_name + '_epoch_{}_metric_{:.4f}_.pth'.format(best_metric_epoch, best_metric)
              
                if model_file != '':
                    torch.save({
                                'model_state_dict': model,
                                'optimizer_state_dict': optimizer,
                                }, 
                                model_file)



                print(
                    "CURREN EPOCH: {} CURRENT LOSS: {:.4f} BEST LOSS: {:.4f} at epoch {}".format(
                        epoch + 1, val_epoch_nums[0], best_metric, best_metric_epoch)
                    )
                writer.add_scalar("Validation/val_loss_epoch", val_epoch_nums[0], epoch + 1)
               
        scheduler.step()

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


        
# endregion TRAIN




def main():

    # ARGUMENTS
    # =============================
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store', type=str, default=None, nargs=1)
    parser.add_argument('--data', action='store', type=str, default='data/demo', nargs=1)

    args = parser.parse_args()


    # DATA
    # =============================
    patch_size_train = (128,128,16)
    patch_size_val = (128,128,16)
    patch_size_test = (128,128,16)
    patch_data = get_data(data_root=args.data, batch_size=(1,1,1,1), num_patches=(20,10,0,0), 
                        patch_size=(patch_size_train, patch_size_val, patch_size_test))

    train_dataset = patch_data['train_dataset']
    val_dataset = patch_data['val_dataset']
    eval_dataset = patch_data['eval_dataset']  
    test_dataset = patch_data['test_dataset']
    train_loader = patch_data['train_loader'] 
    val_loader = patch_data['val_loader']
    eval_loader = patch_data['eval_loader']
    test_loader = patch_data['test_loader'] 
    
    
    # MODEL
    # =============================
    if args.model is None:
        model = Unet3D_CLSTM(encoder_name='seresnext', encoder_depth=3, encoder_weights=None, 
                        decoder_channels=(64, 32, 16), in_channels=1, classes=2, num_clstm_layers=1, device=device,
                        activation=None, decoder_attention_type=None)
    else:
        checkpoint = torch.load('checkpoints/mito_rat.pth', map_location=device)
        model = checkpoint['model_state_dict']
        optimizer = checkpoint['optimizer_state_dict']
        del checkpoint
    model.to(device)
    print('')
    
    
    # CHECK MODEL
    inputs = torch.randn(1,1,128,128,16).to(device)
    # outputs = model(inputs)
    summary(model, input_data=inputs, depth=4, col_names=['output_size', 'num_params'])
    
    
    # LOSS, OPTIMIZER & SCHEDULER
    # =============================
    lfn_1 = BCELoss()
    lfn_2 = Semantic_loss_functions(activation='softmax', bce=True)

    learning_rate = 0.00001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8, weight_decay=0.0001)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    
    
    
    
    # TRAIN
    # =============================
    nepochs = 150
    train(train_loader, val_loader, model, lfn_1, lfn_2, optimizer, scheduler, nepochs)


if __name__ == '__main__':
    main()