# %%
# region IMPORT ###########################################
import argparse
import numpy as np
import torch
import torch.nn.functional as Fn
try:
    from torchinfo import summary
except: 
    from torchsummary import summary

# import warnings 
# warnings.filterwarnings("ignore")

import hdf5storage
from src.data.data_demo_r import get_data
from src.inference.aggregator import GridAggregator
from src.inference.grid_sampler import GridSampler
from connectomics.utils.processing import bc_watershed # after installing
from src.metrics.segmentation.dice_score import dice_score
from src.metrics.detection3d.detection3d_eval import get_mAP_3d

from src.utils.utils_meta import set_seed


# endregion IMPORT



# %%
# SEEDING & DEVICE
seed = 1
set_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')







# %%
# region TEST ############################################

def test(eval_dataset, eval_loader, patch_size_test, model):
    patch_overlap = (4, 4, 4)
    test_batch = 2
    label_channels = 2


    dscores = []
    test_outputs = torch.tensor([], dtype=torch.float32, device=torch.device('cpu'))
    test_outputs_thres = torch.tensor([], dtype=torch.float32, device=torch.device('cpu'))
    num_tests = len(eval_loader)
    model.eval()
    print('')


    # %%
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(eval_loader):
            if batch_idx in list(range(16)):
                batch_image = batch_data['image']
                batch_target = batch_data['label']
                batch_file_name = batch_data['file_name']
                    
                
                for i in range(batch_image.shape[0]):

                    input_tensor = batch_image[i][0]

                    grid_sampler = GridSampler(
                    input_tensor,
                    patch_size_test,
                    patch_overlap,
                    )
                    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=test_batch)
                    aggregator = GridAggregator(grid_sampler)

                    with torch.no_grad():
                        for patches_batch in patch_loader:
                            inputs = patches_batch['image'].to(device)
                            locations = patches_batch['location']
                            logits = model(inputs)
                            labels = Fn.sigmoid(logits)
                            aggregator.add_batch(labels, locations)

                    
                    foreground = aggregator.get_output_tensor()

                    output_thres = torch.zeros_like(foreground)
                    output_thres[foreground>0.5] = 1

                    tem_name = 'temp/jpg/' + batch_file_name[i] + '.jpg'

                test_outputs = torch.cat((test_outputs, foreground.cpu().unsqueeze(dim=0)), dim=0)
                test_outputs_thres = torch.cat((test_outputs_thres,  output_thres.cpu().unsqueeze(dim=0)), dim=0)

                print(f'Batch {batch_idx+1}/{num_tests}')


    # %%
    del batch_data
    del foreground

    # %%
    print('====================Semenatic Segmentation====================')
    for i in range(len(eval_loader)):
        dice_index = dice_score(test_outputs_thres[i,0], torch.tensor(eval_dataset.data[i]['label'][0]))
        print(f'Rat eval volume {i}, Dice score = {dice_index}')
    print('\n\n') 



    # %%
    print('====================Instance Segmentation====================')
    for i in range(len(eval_loader)):
        print(f'---------Volume {i}---------')
        pred_instances = bc_watershed(test_outputs[i].numpy()*255, thres1=0.5, thres2=0.8, thres3=0.1, thres_small=128)
        get_mAP_3d(gt_seg=eval_dataset.data[i]['instances'][0], pred_seg=pred_instances, predict_heatmap=test_outputs[i][0])

    # %%
        pred_instances_dict = {'pred_instances': pred_instances.astype(np.uint16)}
        hdf5storage.savemat('./outputs/' + f'pred_instances_rat_{i}.mat', pred_instances_dict, format='7.3') 


    # %%
        semantic_dict = {'pred_semantic': test_outputs_thres[i].numpy().astype(np.uint8)}
        hdf5storage.savemat('./outputs/' + f'pred_semantic_rat_{i}.mat', semantic_dict, format='7.3')
        print('\n\n') 


# endregion TEST







# %%
# region MAIN #############################################
###########################################################


def main():

    # ARGUMENTS
    # =============================
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store', type=str, default='checkpoints/mito_rat.pth', nargs=1)
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
    checkpoint = torch.load(args.model, map_location=device)
    model = checkpoint['model_state_dict']
    # optimizer = checkpoint['optimizer_state_dict']
    del checkpoint
    model.to(device)
    print('')
    
    
    # CHECK MODEL
    inputs = torch.randn(1,1,128,128,16).to(device)
    # outputs = model(inputs)
    summary(model, input_data=inputs, depth=4, col_names=['output_size', 'num_params'])
    
    
    
    # TEST
    # =============================
    test(eval_dataset, eval_loader, patch_size_test, model)


if __name__ == '__main__':
    main()

# endregion MAIN



