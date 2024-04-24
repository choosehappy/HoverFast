from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations import *
import numpy as np
from skimage.color import rgb2hed, hed2rgb
import numbers

class HEDJitterAugmentation(ImageOnlyTransform):
    def __init__(self,alpha,beta,always_apply=False,p=0.5):
        super(HEDJitterAugmentation, self).__init__(always_apply, p)
        if isinstance(alpha,numbers.Number):
            self.alpha = (-alpha,alpha)
        elif isinstance(alpha,tuple):
            if alpha[0]<=alpha[1]:
                self.alpha = alpha
            else:
                raise ValueError
        else:
            raise ValueError
        if isinstance(beta,numbers.Number):
            self.beta = (-beta,beta)
        elif isinstance(beta,tuple):
            if beta[0]<=beta[1]:
                self.beta = beta
            else:
                raise ValueError
        else:
            raise ValueError
        self.cap = np.array([1.87798274, 1.13473037, 1.57358807])

    def adjust_HED(self,img):
        img = np.array(img)
        
        alpha = np.random.uniform(1+self.alpha[0], 1+self.alpha[1], (1, 3))
        betti = np.random.uniform(self.beta[0], self.beta[1], (1, 3))

        alpha[0,2]=min(alpha[0,2],2*alpha[0,:2].prod()/alpha[0,:2].sum())

        s = rgb2hed(img)/self.cap
        s = alpha * (s + betti)
        nimg = hed2rgb(s*self.cap)

        return (255*nimg).clip(0,255).astype(np.uint8)

    def apply(self, image,**params):
        # Apply your custom color augmentation function to the image
        augmented_image = self.adjust_HED(image)

        # Return the augmented image and the unchanged mask
        return augmented_image
    
def randaugment():
    p=0.8

    aug_always = [HEDJitterAugmentation((-0.4,0.4),(-0.005,0.01),p=p),
                  VerticalFlip(p=p),
                  HorizontalFlip(p=p),
                  RandomResizedCrop(1024, 1024, scale=(0.95, 1.05), always_apply=True, p=p),
                  Rotate(p=1, value=(255,255,255),crop_border=True),
                  RandomSizedCrop((512,512), 512,512)]

    aug_intensity =[RandomBrightnessContrast(p=p, brightness_limit=0.2, contrast_limit=0.2),
                RandomGamma(p=p, gamma_limit = (65,140),eps =1e-7)]

    blur = [Blur(blur_limit=5, p=0.3),
            MotionBlur (p=0.3,blur_limit=(3,5))]

    aug_noise = [GaussNoise(p=p, var_limit=(120,600)), 
                 ISONoise(p=p,intensity=(0.1,0.4), color_shift=(0.05,0.2)),
                 MultiplicativeNoise(p=p,multiplier=(0.75,1.25),elementwise=True)]

    noise_ops = np.random.choice(aug_noise, 1, replace=False).tolist()
    intensity_ops = np.random.choice(aug_intensity, 1, replace=False).tolist()
    blur_ops = np.random.choice(blur, 1, replace=False).tolist()
    transforms = Compose(aug_always + intensity_ops + blur_ops + noise_ops)
    
    return transforms
