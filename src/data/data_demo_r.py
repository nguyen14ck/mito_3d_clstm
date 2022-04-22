
# %%
# region IMPORT ###########################################
###########################################################

import os
import sys
local_path = '../../' # Jupyter
# local_path = './' # Terminal
sys.path.append(local_path)
import glob
import random
random.seed(1)
import numpy as np


import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')
import hdf5storage

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as Ft
from monai.transforms import RandSpatialCropSamplesd


import monai
from monai.data import Dataset
from monai.transforms import \
    apply_transform, Compose, Randomizable, Transform, \
    ToTensord, AddChanneld,  \
    Resized, Resize, CropForegroundd, RandCropByPosNegLabeld, RandSpatialCropd, \
    RandRotate90d, Orientationd, RandAffined, Spacingd, RandRotated, RandZoomd, Rand3DElasticd, RandAxisFlipd, \
    RandGaussianNoised, RandGaussianSmoothd, RandAdjustContrastd, ScaleIntensityd, ScaleIntensityRanged, RandShiftIntensityd

from connectomics.data.utils.data_segmentation import seg_widen_border, seg_to_instance_bd,  seg_to_weights

# endregion IMPORT






# %%
# region DATASET CLASS ####################################
###########################################################
class PatchDataset(Dataset):
    
    def __init__(self, data_root, data_list, set_type=0, n_patches=5, patch_size=(256,256), image_shape=None, transform=None, transform_patch=None):

        
        # basic data config
        self.data_root = data_root
        self.data_list = data_list
        self.set_type = set_type
        self.num_classes = 1
        

        # image patch sampler
        self.num_per_image = n_patches
        self.patch_size = np.array(patch_size)
        if self.num_per_image > 0:
            self.sampler = RandSpatialCropSamplesd(keys=['image', 'label'], roi_size=self.patch_size, num_samples=self.num_per_image,
                                        random_center=True, random_size=False)
        else:
            self.sampler = None

        # set image_resize attribute
        if image_shape is not None:
            if isinstance(image_shape, int):
                self.image_shape = (image_shape, image_shape)
            
            elif isinstance(image_shape, tuple) or isinstance(image_shape, list):
                assert len(image_shape) == 1 or len(image_shape) == 2, 'Invalid image_shape tuple size'
                if len(image_shape) == 1:
                    self.image_shape = (image_shape[0], image_shape[0])
                else:
                    self.image_shape = image_shape
            else:
                raise NotImplementedError 
                
        else:
            self.image_shape = image_shape
            
        # set transform attribute
        self.transform = transform
        self.transform_patch = transform_patch

        # 
        # topt = '4-2-1'
        # topt[0] == '4': # instance boundary mask
        #  _, bd_sz,do_bg = [int(x) for x in topt.split('-')]
        self.LABEL_EROSION = 1
        self.boundary_size = 2 # bd_sz
        self.do_background = 1 # do_bg

        # store im info
        self.pre_im_id = None
        self.pre_im_file = None
        self.data = [None for i in range(len(self.data_list))]
        # self.data = []

    
    def read_full_image(self, image_id):
        # Read full image data and store into the self.data dictionary
        image_file = os.path.normpath(self.data_list[image_id])
        # file_parts = image_file.split('\\')
        file_parts = os.path.split(image_file)

        # image = sio.loadmat(image_file)
        # image = image['outImg']
        
        if self.set_type != 2: # load label if having annotation (not test data)
            # image = sio.loadmat(image_file)
            image = hdf5storage.loadmat(image_file)
            image = image['image']
            
            
            label_file = image_file.replace('images', 'labels').replace('image_', 'seg_')
            # label = sio.loadmat(label_file)
            label = hdf5storage.loadmat(label_file)
            label = label['label']

            # shell_file = label_file[:-4] + '_boundary.mat'
            shell_file = label_file
            shell = hdf5storage.loadmat(shell_file)
            # shell = shell['out_shell']
            shell = shell['label']


            location_file = label_file.replace('.mat', '.txt')
            locations = []
            locations = np.zeros((1, 3))
            
        else:
            image = hdf5storage.loadmat(image_file)
            image = image['image']

            label = np.zeros_like(image)
            locations = np.zeros((1, 3))

        im_min = np.min(image)
        im_max = np.max(image)
        image = (image - im_min) / (im_max - im_min)
        image = image.astype(np.float32)
        label = np.array(label).astype(np.float32)
        shell = np.array(shell).astype(np.float32)


        if self.image_shape is not None:    
            image = Ft.resize(image, self.image_shape)
            label = Ft.resize(label, self.image_shape)
            shell = Ft.resize(shell, self.image_shape)

        if self.transform is not None:
            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)
            shell = np.expand_dims(shell, axis=0)
            image = self.transform(image).squeeze()
            label = self.transform(label).squeeze()
            shell = self.transform(shell).squeeze()
            

        if self.num_per_image > 0 or self.transform is None:
            image = np.array(image)
            label = np.array(label).astype(np.float32)
            shell = np.array(shell).astype(np.float32)


        # add channel dim
        if self.num_per_image > 0:
            while(len(image.shape) - len(self.patch_size) < 1): 
                image = np.expand_dims(image, axis=0)
                label = np.expand_dims(label, axis=0)
                shell = np.expand_dims(shell, axis=0)

            # boundary
            # while(len(labels.shape) - len(self.patch_size) < 1): 
            #     labels = np.expand_dims(labels, axis=0)
        else:
            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)
            shell = np.expand_dims(shell, axis=0)

        
        instances = label.copy()
        label[label>0] = 1
        shell[shell>0] = 1 ### CHANGE ###
        # label = np.array(label).astype(np.float32)


        self.pre_im_id = image_id
        self.pre_im_file = file_parts[-1]

        # self.data = {}
        # self.data['image'] = image
        # self.data['label'] = label
        # self.data['file_name'] = file_parts[-1]
        # self.data['id'] = image_id

        # self.data.append({})
        self.data[image_id] = {}
        self.data[image_id]['image'] = image
        self.data[image_id]['label'] = label
        # self.data[image_id]['labels'] = labels
        self.data[image_id]['shell'] = shell

        self.data[image_id]['instances'] = instances
        self.data[image_id]['file_name'] = file_parts[-1]
        self.data[image_id]['id'] = image_id
        self.data[image_id]['locations'] = locations


    def get_patch(self, idx, image_id):
        patches = self.sampler(self.data[image_id])
        if len(patches) != self.num_per_image:
            raise RuntimeWarning(
                f"`patch_func` must return a sequence of length: samples_per_image={self.num_per_image}."
            )
        patch_id = (idx - image_id * self.num_per_image) * (-1 if idx < 0 else 1)
        patch = patches[patch_id]

        # Transform dictionary
        if self.transform_patch is not None: 
            patch = self.transform_patch(patch)

        patch['file_name'] = self.pre_im_file
        patch['id'] = patch_id


        label = patch['label'][0]
        label_2 = seg_widen_border(label, self.LABEL_EROSION) # eroison
        label_3 = seg_to_instance_bd(label, self.boundary_size, self.do_background).astype(np.float32) # boundary
 
        weight_region = seg_to_weights([label_2], wopts=['1']) #  0 (no weight), 1 (weight_binary_ratio), 2 (weight_unet3d)
        weight_boundary = seg_to_weights([label_3], wopts=['1'])
        # labels = np.stack([label, label_3])
        labels = np.stack([label_2, label_3])
        weights = np.stack([weight_region[0][0], weight_boundary[0][0]])
        patch['labels'] = labels
        patch['weights'] = weights
        
        return patch

    
    def get_patch_2(self, idx, image_id):

        patch = self.crop_patch(image_id)

        # Transform dictionary
        if self.transform_patch is not None: 
            patch = self.transform_patch(patch)


        patch['file_name'] = self.pre_im_file
        patch['id'] = idx

        label = patch['label'][0]
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        label_2 = seg_widen_border(label, self.LABEL_EROSION) # eroison
        label_3 = seg_to_instance_bd(label_2, self.boundary_size, self.do_background).astype(np.float32) # boundary, for seeding, boundary should be built from the label_2 (eroison)
        
        weight_region = seg_to_weights([label_2], wopts=['1']) #  0 (no weight), 1 (weight_binary_ratio), 2 (weight_unet3d)
        # weight_boundary = seg_to_weights([label_3], wopts=['1'])
        weight_boundary = seg_to_weights([patch['shell'][0]], wopts=['1'])

        # labels = np.stack([label, label_3])
        # labels = np.stack([label_2, label_3])
        labels = np.stack([label_2, patch['shell'][0]])
        weights = np.stack([weight_region[0][0], weight_boundary[0][0]])

        patch['labels'] = labels
        patch['weights'] = weights

        return patch


    def get_patch_center(self, image_dims, patch_size, center_list):
        # center = center_list[np.random.choice(len(center_list), size=1, p=center_list[:,3])][0]
        center = center_list[np.random.choice(len(center_list), size=1)][0]

        shift_range = np.round(0.25 * patch_size).astype(np.int)

        center = np.array(center).astype(np.int)
        x = center[0]
        y = center[1]
        z = center[2]
            
        # Add random shift to coordinates for augmentation:
        x = x + np.random.choice(range(-shift_range[0], shift_range[0]+1))
        y = y + np.random.choice(range(-shift_range[1], shift_range[1]+1))
        z = z + np.random.choice(range(-shift_range[2], shift_range[2]+1))
        
        # Move inside if the center is too close to border:
        if (x<patch_size[0]/2) : x = np.ceil(patch_size[0]/2)
        if (y<patch_size[1]/2) : y = np.ceil(patch_size[1]/2)
        if (z<patch_size[2]/2) : z = np.ceil(patch_size[2]/2)
        if (x>image_dims[0]-patch_size[0]/2): x = image_dims[0] - np.ceil(patch_size[0]/2)
        if (y>image_dims[1]-patch_size[1]/2): y = image_dims[1] - np.ceil(patch_size[1]/2)
        if (z>image_dims[2]-patch_size[2]/2): z = image_dims[2] - np.ceil(patch_size[2]/2)

        
        return np.array([x, y, z]).astype(np.int)


    def crop_patch(self, image_id):
        center = self.get_patch_center(self.data[image_id]['image'].shape[1:], self.patch_size, self.data[image_id]['locations'])


        half = self.patch_size // 2

        # calculate subtomogram corners
        x = center[0] - half[0], center[0] + half[0]
        y = center[1] - half[1], center[1] + half[1]
        z = center[2] - half[2], center[2] + half[2]


        patch = {}
        # load reconstruction and ground truths
        patch['image'] = self.data[image_id]['image'][:, x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        patch['label'] = self.data[image_id]['label'][:, x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        patch['shell'] = self.data[image_id]['shell'][:, x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        patch['instances'] = self.data[image_id]['instances'][:, x[0]:x[1], y[0]:y[1], z[0]:z[1]]


        return patch
      
                                  
    def __len__(self):
        """
        return length of the dataset
        """
        if self.num_per_image > 0:
            return self.num_per_image * len(self.data_list)
        else:
            return len(self.data_list)
        
    
    def __getitem__(self, idx):
        """
        For given index, return images with resize and preprocessing.
        """
        if self.num_per_image > 0:
            # READ FULL IMAGE AND LABEL
            image_id = int(idx/self.num_per_image)

            # if image_id != self.pre_im_id:
            #     self.read_full_image(image_id)
                
            if self.data[image_id] is None:
                self.read_full_image(image_id)

            # if len(self.data) < image_id+1:
            #     self.read_full_image(image_id)

            # CROP PATCHES FROM DATA DICTIONARY
            if self.set_type==0 or self.set_type==1: # train or validation
                patch = self.get_patch_2(idx, image_id) # crop by random center
            else:
                patch = self.get_patch(idx, image_id) # random crop

            return patch

        else:
            self.read_full_image(idx)
            return self.data[idx]

                
# endregion DATASET CLASS


# %%
# %%
# region DATASET FUNCTION #################################
###########################################################

def get_data(data_root='data', ids=None, batch_size=(2,2,2,2), num_workers=0, num_patches=(2200,1700,0,0), patch_size=((144, 144, 144), (96, 96, 96), (160, 160, 160)), cfg=None):



    # %%
    ###### DATASETS ######
    #########################################################################################


    # %%
    # ------ DATA PATH ------
    # --------------------------------------------------------------------------------------

    data_train_path = data_root + '/rat/images'
    data_val_path = data_root + '/rat/images'
    data_test_path = data_root + '/rat/images'


    train_list = glob.glob(data_train_path + '/*.mat')
    val_list = glob.glob(data_val_path + '/*.mat')
    test_list = glob.glob(data_test_path + '/*.mat')

    if ids is not None:
        train_list = map(train_list.__getitem__, ids[0])
        val_list = map(val_list.__getitem__, ids[1])
        test_list = map(test_list.__getitem__, ids[2])
        train_list = list(train_list)
        val_list = list(val_list)
        test_list = list(test_list)
        
    im_resize = 1024

    patch_size_train = patch_size[0]
    patch_size_val = patch_size[1]
    patch_size_test = patch_size[2]


    




    # %%
    ###### TRANSFORMATION ######
    #########################################################################################

    

    # %%
    def image_preprocess_transforms():
        
        preprocess = transforms.Compose([
            # transforms.Resize(512),
            transforms.Resize(im_resize),
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor()
            ])
        
        return preprocess



    # %%
    def get_mean_std(data_root, batch_size=8, num_workers=0):
        
        pre_transforms = transforms.Compose([
            # image_preprocess_transforms(),
            # transforms.Grayscale(),
            transforms.ToTensor(), # transpose from H*W*C to C*H*W
        ])

         
        
        class MyDataset(Dataset):
            """custom dataset for .mat images"""

            def __init__(self, list_of_urls, transforms):
                self.list_of_urls = list_of_urls
                self.transform = transforms

            def __len__(self):
                return len(self.list_of_urls)

            def __getitem__(self, index):
                image_url = self.list_of_urls[index]
                image = hdf5storage.loadmat(image_url)
                image = pre_transforms(image['outImg'])
                # print(image)
                return image
        
        if data_root is not None:
            dataset = datasets.ImageFolder(root=data_root, transform=pre_transforms)
        else:
            dataset = MyDataset(test_list, pre_transforms)

        
        
        loader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            # sampler=SubsetRandomSampler(sample_ids),
                                            shuffle=False)

        mean = 0.
        std = 0.

        
    
        for images in loader:
            # print(images)
            # print(type(images[0]))
            batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)

        mean /= len(loader.dataset)
        std /= len(loader.dataset)
        
        print('mean: {}, std: {}'.format(mean, std))
        
        return mean, std




    # %%
    def image_common_transforms(mean=(0.5671, 0.4666, 0.3664), std=(0.2469, 0.2544, 0.2584)):
        preprocess = image_preprocess_transforms()
        
        common_transforms = transforms.Compose([
            # preprocess,
            transforms.ToTensor(), # transpose from H*W*C to C*H*W
            transforms.Normalize(mean, std)
        ])
        
        return common_transforms


    # %%
    def data_augmentation_preprocess(mean=(0.5671, 0.4666, 0.3664), std=(0.2469, 0.2544, 0.2584)):
    
        preprocess = image_preprocess_transforms()

        augmentation_transforms = transforms.Compose([
            # preprocess,
            # transforms.RandomResizedCrop(224),
            transforms.RandomResizedCrop(round(0.875*im_resize)),
            transforms.RandomAffine(degrees=30),
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(), # transpose from H*W*C to C*H*W
            transforms.Normalize(mean, std)
                                                    ])
        return augmentation_transforms


    # %%
    def patch_transform():
        tf = transforms.ToTensor()
        return tf


    def monai_transforms():
        # Define transforms for image
        train_transforms = Compose(
            [    
                RandRotate90d(keys=['image', 'label', 'shell', 'instances'], prob=0.7, spatial_axes=(0,1)), # count 0, ignore the first index of batch
                RandZoomd(keys=["image", "label", 'shell', 'instances'], prob=0.25, min_zoom=0.8, max_zoom=1.25, padding_mode='constant', keep_size=True),
                RandAdjustContrastd(keys=['image'], gamma=1.2, prob=0.2),
                RandGaussianNoised(keys=['image'], prob=0.2, mean=0.3, std=0.1),
                RandGaussianSmoothd(keys=['image'], prob=0.2),
            ],
            
        )
        val_transforms = train_transforms

        eval_transforms = Compose(
            [
                # AddChanneld(keys=["image", "label"]),
                # Resized(keys=["image", "label"], spatial_size=(512, 512)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = eval_transforms

        return train_transforms, val_transforms, eval_transforms, test_transforms




    # %%
    ###### DATA LOADER ######
    #########################################################################################

    # %%
    # ------ NORMALIZE ------
    # --------------------------------------------------------------------------------------
        
    # mean, std = get_mean_std(data_root=None, batch_size=batch_size, num_workers=num_workers)
    mean = 0.5008
    std = 0.2637

  

    # %%
    # ------ TRANSFORM ------
    # --------------------------------------------------------------------------------------

    # Image transforms
    common_transforms = image_common_transforms(mean, std)
    data_augmentation = False
    if data_augmentation:    
        train_transforms = data_augmentation_preprocess(mean, std)
    else:
        train_transforms = common_transforms

    # Patch transforms
    train_transforms, val_transforms, eval_transforms, test_transforms = monai_transforms()
    check_transforms = Compose(
                    Resize(spatial_size=(512,512,100))
    )



        



    # %%
    # ------ RELOAD DATASETS (TRANSFORMS) ------
    # --------------------------------------------------------------------------------------


    train_dataset =  PatchDataset(data_root, train_list, set_type=0, n_patches=num_patches[0], patch_size=patch_size_train, 
                                    image_shape=None, transform=None, transform_patch=train_transforms)
    print('Length of train dataset: {}'.format(len(train_dataset)))

    val_dataset =  PatchDataset(data_root, val_list, set_type=1, n_patches=num_patches[1], patch_size=patch_size_val, 
                                    image_shape=None, transform=None, transform_patch=None)
    print('Length of valid dataset: {}'.format(len(val_dataset)))

    eval_dataset =  PatchDataset(data_root, val_list, set_type=1, n_patches=num_patches[2], patch_size=patch_size_val, 
                                    image_shape=None, transform=None, transform_patch=None)
    print('Length of evaluation dataset: {}'.format(len(eval_dataset)))

    test_dataset =  PatchDataset(data_root, test_list, set_type=2, n_patches=num_patches[3], patch_size=patch_size_test, 
                                    image_shape=None, transform=None, transform_patch=None)
    print('Length of test dataset: {}'.format(len(test_dataset)))



    # %%
    # ------ LOADERS ------
    # --------------------------------------------------------------------------------------
    # num_workers = 0

    train_loader = DataLoader(train_dataset, 
                                batch_size=batch_size[0], 
                                shuffle=True,
                                # sampler=train_sampler, 
                                num_workers=num_workers)


    val_loader = DataLoader(val_dataset, 
                                batch_size=batch_size[1], 
                                shuffle=False, 
                                num_workers=num_workers)

    eval_loader = DataLoader(eval_dataset, 
                                batch_size=batch_size[2], 
                                shuffle=False, 
                                num_workers=num_workers)


    test_loader = DataLoader(test_dataset, 
                                batch_size=batch_size[3], 
                                shuffle=False, 
                                num_workers=num_workers)
  


    # %%

    pdata = {}
    pdata['train_dataset'] = train_dataset
    pdata['val_dataset'] = val_dataset
    pdata['eval_dataset'] = eval_dataset
    pdata['test_dataset'] = test_dataset

    pdata['train_loader'] = train_loader
    pdata['val_loader'] = val_loader
    pdata['eval_loader'] = eval_loader
    pdata['test_loader'] = test_loader

    
    return pdata