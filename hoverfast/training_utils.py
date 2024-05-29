import torch
import torch.nn as nn
import tables
import time
import torch.nn.functional as F
import math
from albumentations import *
import numpy as np
import skimage.morphology as ndi
from skimage.measure import regionprops
from .hoverfast import HoverFast
from .augment import *
from torch.utils.data import DataLoader
from .training_utils import *
from tqdm import tqdm
from torchmetrics.classification import BinaryConfusionMatrix
import os
import datetime
from tensorboardX import SummaryWriter


class Dataset(object):
    def __init__(self, fname, device ,transforms=None, edge_weight= False):
        """
        Initialize the Dataset object.

        Parameters:
        fname (str): Filename of the HDF5 database.
        device (torch.device): Device to perform computation on (GPU or CPU).
        transforms (callable, optional): Transformations to be applied to the images and labels. Default is None.
        edge_weight (bool, optional): Whether to compute edge weights. Default is False.
        """

        self.fname=fname
        self.edge_weight = edge_weight
        self.device = device
        self.transforms=transforms

        with tables.open_file(self.fname) as db:
            self.numpixels=db.root.numpixels[:]
            self.nitems=db.root.img.shape[0]
        
    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Parameters:
        index (int): Index of the item to retrieve.

        Returns:
        tuple: Tuple containing the image, mask, maps, edge weights, and boundary weights.
        """
        with tables.open_file(self.fname,'r') as db:
            img = db.root.img[index]
            label = db.root.label[index]
        
        if self.transforms:
            transforms=self.transforms()
            augmented = transforms(image=img, mask=label)
            img = augmented['image']
            label = augmented['mask']
        
        mask = label!=0
        if(self.edge_weight):
            eweight = ndi.binary_dilation(mask==1, ndi.square(5)) & ~mask
        else: #otherwise the edge weight is all ones and thus has no affect
            eweight = np.ones(mask.shape,dtype=mask.dtype)

        maps,bweight = make_maps(label)

        if img.dtype == 'uint8':
            img = img/255

        return (torch.from_numpy(img).permute(2, 0, 1),torch.from_numpy(mask),torch.from_numpy(maps),torch.from_numpy(eweight),torch.from_numpy(bweight))
    def __len__(self):
        """
        Get the number of items in the dataset.

        Returns:
        int: Number of items in the dataset.
        """
        return self.nitems

def asMinutes(s):
    """
    Convert seconds to a string representing minutes and seconds.

    Parameters:
    s (float): Time in seconds.

    Returns:
    str: Time formatted as 'Xm Ys'.
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    """
    Calculate the elapsed time and estimated remaining time.

    Parameters:
    since (float): Start time in seconds.
    percent (float): Progress percentage (0 to 1).

    Returns:
    str: Elapsed and remaining time formatted as 'Elapsed (- Remaining)'.
    """
    now = time.time()
    s = now - since
    es = s / (percent+.00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def make_maps(label):
    """
    Create feature maps and boundary weight for a given label.

    This function generates horizontal and vertical gradient maps, and computes boundary weights for each region in the label.

    Parameters:
    label (numpy.ndarray): Label image with segmented regions.

    Returns:
    tuple: Tuple containing the maps and the boundary weight.
    """
    maps = np.zeros((2,)+label.shape,np.float32)
    weight = np.ones(label.shape)
    rgs = regionprops(label)
    for rg in rgs:
        ymin,xmin,ymax,xmax = rg.bbox
        shape = rg.image.shape
        if (ymin==0)|(xmin==0)|(ymax==label.shape[0])|(xmax==label.shape[1]):
            weight[ymin:ymax,xmin:xmax] = 0
        else:
            maps[0,ymin:ymax,xmin:xmax] += rg.image * np.linspace(-1,1,shape[1])
            maps[1,ymin:ymax,xmin:xmax] += rg.image * np.linspace(-1,1,shape[0]).reshape((shape[0],1))
    return maps,weight

def dice_loss(pred, true, smooth=1e-3):
    """
    Compute the Dice loss between predicted and true labels.

    Parameters:
    pred (torch.Tensor): Predicted labels, assumed to be of shape NxHxWxC.
    true (torch.Tensor): True labels, assumed to be of shape NxHxWxC.
    smooth (float, optional): Smoothing factor to avoid division by zero. Default is 1e-3.

    Returns:
    torch.Tensor: Computed Dice loss.
    """
    inse = torch.sum(pred * true, (0, 1, 2))
    l = torch.sum(pred, (0, 1, 2))
    r = torch.sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.sum(loss)
    return loss

def grad_kernel(size=11):
    """
    Create a gradient kernel for computing image gradients.

    This function generates a kernel for computing horizontal and vertical gradients
    in an image. The kernel size is adjustable.

    Parameters:
    size (int, optional): Size of the kernel. Default is 11.

    Returns:
    torch.Tensor: A 4D tensor representing the gradient kernel.
    """

    temp = torch.arange(-size // 2 + 1,size // 2 + 1,dtype=torch.float32,device="cuda",requires_grad=False,)
    temp = torch.outer(temp,torch.ones_like(temp))
    stemp = temp*temp
    return (temp/(stemp+stemp.T+1.0e-15)).view(1,1,size,size)

class Criterion(nn.Module):
    def __init__(self,class_weight,edge_weight,grad_weight,hv_weight,dice_weight,crossentropy_weight):
        """
        Initialize the Criterion object for computing loss.

        Parameters:
        class_weight (torch.Tensor): Weights for each class for cross-entropy loss.
        edge_weight (float): Weight for edge loss.
        grad_weight (float): Weight for gradient loss.
        hv_weight (float): Weight for horizontal and vertical map loss.
        dice_weight (float): Weight for dice loss.
        crossentropy_weight (float): Weight for cross-entropy loss.
        """

        super(Criterion, self).__init__()
        self.critCEntropy = nn.CrossEntropyLoss(weight = class_weight, ignore_index = -100 , reduction='none')
        self.critGrad = nn.MSELoss(reduction='none')
        self.crithv = nn.MSELoss(reduction='none')
        self.edge_weight=edge_weight
        self.class_weight=class_weight
        self.grad_weight=grad_weight
        self.hv_weight=hv_weight
        self.dice_weight=dice_weight
        self.crossentropy_weight=crossentropy_weight
        self.kernel = grad_kernel()
    
    def forward(self,x_pred,hvm_pred,y,hvmaps,y_weight,hv_weight):
        """
        Compute the loss components for model training.

        Parameters:
        x_pred (torch.Tensor): Predicted class scores from the model.
        hvm_pred (torch.Tensor): Predicted horizontal and vertical maps from the model.
        y (torch.Tensor): True class labels.
        hvmaps (torch.Tensor): True horizontal and vertical maps.
        y_weight (torch.Tensor): Weights for the true class labels.
        hv_weight (torch.Tensor): Weights for the horizontal and vertical maps.

        Returns:
        tuple: Tuple containing the horizontal and vertical map loss, gradient loss,
               cross-entropy loss, and dice loss.
        """
        # Cross-entropy loss (Lc)
        loss_matrix = self.critCEntropy(x_pred, y)
        lossCEntropy = (loss_matrix * (self.edge_weight**y_weight)).mean() #can skip if edge weight==1
        
        # dice loss (Ld)
        lossD=dice_loss(x_pred.argmax(1),y)
        
        # HV map loss (La)
        lossHV=self.crithv(hvm_pred.squeeze(),hvmaps)
        weight = self.class_weight[y]*hv_weight
        lossHV[:,0] *= weight
        lossHV[:,1] *= weight
        lossHV.permute(0,2,3,1)[(y==0)&(x_pred.argmax(1)==1)]=0
        lossHV=lossHV.mean()
        
        #gradient loss (Lb)
        grad=torch.cat((F.conv2d(hvmaps[:,0].unsqueeze(1),self.kernel.permute(0,1,3,2),padding='same'),F.conv2d(hvmaps[:,1].unsqueeze(1),self.kernel,padding='same')),axis=1)
        grad_pred=torch.cat((F.conv2d(hvm_pred[:,0].unsqueeze(1),self.kernel.permute(0,1,3,2),padding='same'),F.conv2d(hvm_pred[:,1].unsqueeze(1),self.kernel,padding='same')),axis=1)
        
        lossGrad=self.critGrad(grad_pred,grad)
        lossGrad[:,0] *= weight
        lossGrad[:,1] *= weight
        lossGrad.permute(0,2,3,1)[(y==0)&(x_pred.argmax(1)==1)]=0
        lossGrad=lossGrad.mean()
        
        return self.hv_weight*lossHV, self.grad_weight*lossGrad, self.crossentropy_weight*lossCEntropy, self.dice_weight*lossD




def main_train(args) -> None:
    """
    Main function to train the HoverFast model.

    This function sets up the training parameters, initializes the model, loads the dataset, 
    and performs training and validation over the specified number of epochs.

    Parameters:
    args (argparse.Namespace): Parsed command-line arguments.
    """
    datapath = args.dataset_path
    dataname = args.dataname
    outdir = args.outdir
    batch_size = args.batch_size
    n_process = min(batch_size,os.cpu_count())
    num_epochs = args.epoch
    depth = args.depth  # Depth of the network 
    wf = args.width  # wf (int): number of filters in the first layer is 2**wf

    # UNet params
    n_classes= 2 # Number of classes in the data mask to predict
    in_channels= 3 # Input channels of the data, RGB = 3
    padding= True # Whether to use padding
    batch_norm = True # Whether to use batch normalization between layers
    up_mode = 'upconv' # 'upconv' for transpose convolution or 'upsample' for interpolation
    conv_block = "msunet"  # Convolutional block type, either 'unet' or 'msunet'

    # Training parameters
    edge_weight = 1.1  # Boost edge values based on original UNet paper
    phases = ["train", "test"]  # Phases for training and testing
    validation_phases = ["test"]  # Phases for validation
    grad_weight, hv_weight, dice_weight, crossentropy_weight = (1/10, 14, 1/6, 1)  # Loss weights

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = HoverFast(
        n_classes=n_classes, in_channels=in_channels, padding=padding,
        depth=depth, wf=wf, up_mode=up_mode, batch_norm=batch_norm,
        conv_block=conv_block
    ).to(device, memory_format=torch.channels_last)

    # Load dataset and DataLoader
    dataset={}
    dataLoader={}
    for phase in phases:
        dataset[phase]=Dataset(os.path.join(datapath,dataname)+f"_{phase}.pytable",device, transforms= randaugment,edge_weight=edge_weight)
        dataLoader[phase]=DataLoader(dataset[phase], batch_size=batch_size,shuffle=True, num_workers=n_process, pin_memory=True, drop_last=True)

    optim = torch.optim.Adam(model.parameters()) 
    class_weight=dataset["train"].numpixels[1,:]
    f=np.sum(class_weight)/class_weight
    class_weight=f/np.sum(f)
    class_weight = torch.from_numpy(class_weight).type('torch.FloatTensor').to(device)

    print(f'class weight: {class_weight}') # Display class weights

    criterion = Criterion(class_weight,edge_weight,grad_weight,hv_weight,dice_weight,crossentropy_weight)
    bcm = BinaryConfusionMatrix().to(device)

    writer=SummaryWriter(os.path.join(outdir,f"hoverfast_{dataname}_"+datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M"))) # Open the tensorboard visualiser

    best_loss_on_test = np.Infinity
    edge_weight=torch.tensor(edge_weight).to(device)
    start_time = time.time()
    for epoch in range(num_epochs):
        for phase in phases:
            stats={}
            stats['loss']={}
            for stat in ['total_loss','hv_loss','grad_loss','crossEntropy_loss','dice_loss']:
                stats['loss'][stat] = 0
            stats['cmatrix'] = torch.zeros((n_classes,n_classes)).to(device)

            if phase == 'train':
                model.train()  
            else: 
                model.eval()   

            for _ , (X, y, hvmaps, y_weight, b_weight) in enumerate(tqdm(dataLoader[phase],leave=False)): 
                X = X.type('torch.FloatTensor').to(device, memory_format=torch.channels_last)
                y = y.type('torch.LongTensor').to(device)
                hvmaps = hvmaps.to(device)
                y_weight = y_weight.to(device)
                b_weight = b_weight.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    
                    x_pred,hvm_pred = model(X)
                    
                    losses = criterion(x_pred,hvm_pred,y,hvmaps,y_weight,b_weight)
                    loss = sum(losses)

                    if phase=="train":
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        train_loss = loss
                    
                    stats['loss']['total_loss']+=loss.detach()
                    stats['loss']['hv_loss']+=losses[0].detach()
                    stats['loss']['grad_loss']+=losses[1].detach()
                    stats['loss']['crossEntropy_loss']+=losses[2].detach()
                    stats['loss']['dice_loss']+=losses[3].detach()

                    if phase in validation_phases:
                        
                        predflat=x_pred.argmax(axis=1).flatten()
                        targetflat=y.flatten()
                        
                        stats['cmatrix']+= bcm(predflat, targetflat).detach()
                
            n_batches=len(dataLoader[phase])
            print(n_batches)
            stats['loss']['total_loss']=(stats['loss']['total_loss']/n_batches).cpu().numpy()
            stats['loss']['hv_loss']=(stats['loss']['hv_loss']/n_batches).cpu().numpy()
            stats['loss']['grad_loss']=(stats['loss']['grad_loss']/n_batches).cpu().numpy()
            stats['loss']['crossEntropy_loss']=(stats['loss']['crossEntropy_loss']/n_batches).cpu().numpy()
            stats['loss']['dice_loss']=(stats['loss']['dice_loss']/n_batches).cpu().numpy()
            
            if phase in validation_phases:
                stats['cmatrix']=(stats['cmatrix']/stats['cmatrix'].sum()).cpu().numpy()

            # Save metrics to tensorboard
            writer.add_scalars(f'{phase}/loss', stats['loss'], epoch)
            if phase in validation_phases:
                writer.add_scalar(f'{phase}/accuracy', stats['cmatrix'].trace(), epoch)
                writer.add_scalar(f'{phase}/precision', stats['cmatrix'][1,1]/stats['cmatrix'][:,1].sum(), epoch)
                writer.add_scalar(f'{phase}/recall', stats['cmatrix'][1,1]/stats['cmatrix'][1].sum(), epoch)
                writer.add_scalar(f'{phase}/specificity', stats['cmatrix'][0,0]/stats['cmatrix'][0].sum(), epoch)
                writer.add_scalar(f'{phase}/negative predictive value', stats['cmatrix'][0,0]/stats['cmatrix'][:,0].sum(), epoch)
            
            if phase == 'train':
                train_loss = stats['loss']['total_loss']
            current_loss = stats['loss']['total_loss']

        print('%s ([%d/%d] %d%%), train loss: %.4f test loss: %.4f' % (timeSince(start_time, (epoch+1) / num_epochs), 
                                                    epoch+1, num_epochs ,(epoch+1) / num_epochs * 100, train_loss, current_loss),end="")    


        if current_loss < best_loss_on_test:
            best_loss_on_test = current_loss
            print("  **")
            state = {'epoch': epoch + 1,
            'model_dict': model.state_dict(),
            'optim_dict': optim.state_dict(),
            'best_loss_on_test': current_loss,
            'n_classes': n_classes,
            'in_channels': in_channels,
            'padding': padding,
            'depth': depth,
            'wf': wf,
            'up_mode': up_mode,
            'batch_norm': batch_norm,
            'conv_block': conv_block }


            torch.save(state, f"{outdir}/{dataname}_best_model.pth")
        else:
            print("")
