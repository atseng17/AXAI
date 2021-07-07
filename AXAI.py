import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
from skimage.segmentation import quickshift
import os

class AXAI():
    def __init__(self, model, inputs, loss_criterion, std, mean):
        """Init function.
        Args:
            model: Model to explain
            inputs: Data to explain
            loss_criterion: Loss function of the model
            std: Std of dataset used in preprocessing
            mean: Mean of dataset used in preprocessing
        """
        self.model = model
        self.inputs = inputs
        self.loss_criterion = loss_criterion
        self.std = std
        self.mean = mean
        
    def tensor2cuda(self, tensor):
        """
        AXAI utilises the GPU if it detects there are available GPUs on the users machine. 
        Args:
            tensor: Pytorch tensor
        """
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor

    def gen_adv(self):
        """
        Return:
            diff: This is the generation of adverserial input using PGDM. 
        """
        self.model.eval()
        with torch.no_grad():
            adv_examples=[]
            data = self.tensor2cuda(self.inputs)
            output = self.model(data)
            pred = torch.max(output, dim=1)[1] 
            with torch.enable_grad():
                adv_data, diff_tmp= self.pgdm(net=self.model, x=data,\
                y=pred, loss_criterion=self.loss_criterion, \
                alpha=0.001, eps=.2, steps=20, radius=.2, norm=2)         
            diff = diff_tmp.squeeze().detach().cpu().numpy()
        return diff
    
    def Attack_and_Filter(self):
        """
        Return:
            Filtered_Attacks: .Attack_and_Filter() Wraps up .gen_adv() and .threshold()
        """
        Attacks = self.gen_adv()
        Filtered_Attacks = self.threshold(Attacks)
        return Filtered_Attacks
    
    def explain(self, K=5, kernel_size=8, max_dist=10, 
                 ratio=.1):
        """
        Performs image segmentation via QuickShift (Vedaldi et al., 2008), and maps the 
        filtered adverserial attacks back to the original image. The K argument is the 
        explanation length, i.e. the number of explanable image segments one desires to 
        show. kernel_size, max_dist, and ratio are three input parameters for QuickShift. 
        The user is suggested to play with the four arguments in the .explain() method. 
        On how the parameters would affect the explanation results, please refer to our paper. 

        Args:
            K: Explanation length, i.e. the number of explanable features to show.
            kernel_size: Input parameters for QuickShift.
            max_dist: Input parameters for QuickShift.
            ratio: Input parameters for QuickShift.
        Return:
            Denormalize explaintion
            Original image
        """

        # Pulls out the filtered attacks
        Filtered_Attacks = self.Attack_and_Filter()
        
        # Image segmentation process
        data_org = self.inputs.squeeze().detach().cpu().numpy()
        image = np.transpose(data_org, (1, 2, 0))
        segments_orig = quickshift(image, kernel_size=kernel_size,
                                   max_dist=max_dist, ratio=ratio)

        values, counts = np.unique(segments_orig, return_counts=True)
        attack_frequency=[]
        attack_intensity=[]

        for i in range(len(values)):
            segments_orig_loc=segments_orig==values[i]
            tmp = np.logical_and(segments_orig_loc,Filtered_Attacks)
            attack_frequency.append(np.sum(tmp))
            attack_intensity.append(np.sum(tmp)/counts[i])
            
        # Mapping process
        top_attack = np.sort(attack_intensity)[::-1][:K]
        zero_filter = np.zeros(np.array(attack_intensity).shape, 
                               dtype=bool)
        for i in range(len(top_attack)):
            intensity_filter = attack_intensity == top_attack[i]
            zero_filter = zero_filter+intensity_filter

        strongly_attacked_list = values[zero_filter]
        un_slightly_attacked_list = np.delete(values, 
                                              strongly_attacked_list)
        strongly_attacked_image = copy.deepcopy(image)
        for x in un_slightly_attacked_list:
            strongly_attacked_image[segments_orig == x] = (255,255,255)
        
        # Make the original_img in the desired format
        original_img = np.transpose(self.inputs.squeeze().
                                    detach().cpu().numpy(), (1, 2, 0))

        return self.denormalize(strongly_attacked_image),\
               self.denormalize(original_img)

            
    def threshold(self,diff, percentage=15):
        """
        As mentoned in our paper, a theshold is defined to filter out unuseful features. 
        diff is the adverserial attack on the original image. The suggested value of the 
        percentage argument is also explained in our paper. 
        Args:
            diff: Adverserial attacks generated via PGDM
            percentage: the threshold filtering out unuseful features
        Return:
            Filtered attacks
        """
        dif_total_1 = copy.deepcopy(diff[0])
        dif_total_2 = copy.deepcopy(diff[1])
        dif_total_3 = copy.deepcopy(diff[2])
        thres_1_1=np.percentile(dif_total_1, percentage)
        thres_1_2=np.percentile(dif_total_1, 100-percentage)
        mask_1_2 = (dif_total_1 >= thres_1_1) &\
                    (dif_total_1 < thres_1_2)
        dif_total_1[mask_1_2] = 0
        
        thres_2_1=np.percentile(dif_total_2, percentage)
        thres_2_2=np.percentile(dif_total_2, 100-percentage)
        mask_2_2 = (dif_total_2 >= thres_2_1) &\
                    (dif_total_2 < thres_2_2)
        dif_total_2[mask_2_2] = 0

        thres_3_1=np.percentile(dif_total_3, percentage)
        thres_3_2=np.percentile(dif_total_3, 100-percentage)
        mask_3_2 = (dif_total_3 >= thres_3_1) &\
                    (dif_total_3 < thres_3_2)
        dif_total_3[mask_3_2] = 0        
        dif_total = dif_total_1+dif_total_2+dif_total_3

        return dif_total

    def pgdm(self,net, x, y, loss_criterion, alpha, eps, steps,\
             radius, norm): 
        """
        Projected Gradient Descent Method (PGDM) (Madry et al., 2017) 
        """
        pgd = x.new_zeros(x.shape)
        adv_x = x + pgd
        for step in range(steps):
            pgd = pgd.detach()
            x = x.detach()
            adv_x = adv_x.clone().detach()
            adv_x.requires_grad = True 
            preds = net(adv_x)
            net.zero_grad()
            loss = loss_criterion(preds, y)
            loss.backward(create_graph=False, retain_graph=False)
            adv_x_grad = adv_x.grad
            scaled_adv_x_grad = adv_x_grad/adv_x_grad.\
                                view(adv_x.shape[0], -1)\
                                .norm(norm, dim=-1).view(-1, 1, 1, 1)

            pgd = pgd + (alpha*scaled_adv_x_grad)

            mask = pgd.view(pgd.shape[0], -1).norm(norm, dim=1) <= eps
            scaling_factor = pgd.view(pgd.shape[0], -1).\
                             norm(norm, dim=1)
            scaling_factor[mask] = eps
            pgd *= eps / scaling_factor.view(-1, 1, 1, 1)
            adv_x = x + pgd 
        return adv_x, pgd
    
    def denormalize(self,norm_img):
        """
        Denormalizes the image that is normalized during preprocessing, 
        this is just for showing the explanations wihtout normalization. 
        Args:
            norm_img: Normalized image
        Return:
            Denormalized image
        """
        return norm_img*self.std+self.mean
    
    def plotter(self,Explanations,save_path=None,save=False):
        """
        Explanations,save_path=None,save
        Plots a side by side image (original image and its corresponding explanation). 
        If save=True then the side by side image is saved as a .png file.
        Args:
            Explanations: The explanations generated by .explain()
            save_path: Save path
            save: Save image based on the user defined path is set to True
        """
        original_img = np.transpose(self.inputs.squeeze().
                                    detach().cpu().numpy(), (1, 2, 0))
        original_img = self.denormalize(original_img)

        fig = plt.figure(figsize=(6,3))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(original_img)
        ax.axis('off')
        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(Explanations)
        ax.axis('off')
        if save:
            plt.savefig(os.path.join(save_path,'_explanation.png'),
                        dpi=300,bbox_inches='tight')