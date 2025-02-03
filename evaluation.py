import sys
sys.path.append('/home/fa578s/Desktop/CSC790_project/atari/')

from atari.play import AtariNet
from atari.ale_env import ALEModern, ALEClassic
import torch
from pathlib import Path
from gzip import GzipFile
from captum.attr import LRP, NoiseTunnel, IntegratedGradients, GradientShap, Occlusion
import os
import cv2
import numpy as np
from pytorchgradcammaster.pytorch_grad_cam.utils.image import preprocess_image
from PyTorchRelevancePropagationmaster.src.lrp import LRPModel
from PyTorchRelevancePropagationmaster.projects.per_image_lrp.visualize import plot_relevance_scores
from PytorchLRPmaster.innvestigator import InnvestigateModel
import seaborn as sns
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from SmoothGrad.smooth_grad import SmoothGrad
from pytorchsmoothgrad.lib.gradients import SmoothGrad
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap



def _load_checkpoint(fpath, device="cpu"):
    fpath = Path(fpath)
    with fpath.open("rb") as file:
        with GzipFile(fileobj=file) as inflated:
            return torch.load(inflated, map_location=device)

def prediction_func(frame):
    input_frame = torch.tensor(frame).permute(0, 3, 1, 2)
    grayscale_channel = input_frame.mean(dim=1, keepdim=True) 
    four_channel = torch.cat([input_frame, grayscale_channel], dim=1)
    input_frame = four_channel.to(torch.uint8)
    qs = model(input_frame)
    return torch.softmax(qs, dim=1).detach().numpy()

def predict(input_tensor, model):

    input_tensor = input_tensor.to(torch.uint8)
    pred = model(input_tensor)
    return torch.softmax(pred, dim=1).detach().numpy()


def Lime_exp(input_tensor, fr, des_path):

    input_tensor = input_tensor.to(torch.float32)

    input_tensor = input_tensor.squeeze(0)
    input = input_tensor.mean(dim=0)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(input.detach(), 
                                            prediction_func, # classification function
                                            top_labels=1, 
                                            hide_color=255, 
                                            num_samples=3000)

    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,
        hide_rest=False,
        num_features=50,
        min_weight=0.001,
    )

    plt.imshow(temp)
    plt.imshow(mask, alpha=0.5, cmap='jet')  # Overlay the mask with some transparency
    plt.axis('off')

    output_file = os.path.join(des_path, f'explanation_{fr // 4}.png')
    plt.savefig(output_file)
    plt.close()

def LRP_exp(input_tensor, fr, des_path, model):
    input_tensor = input_tensor.to(torch.float32)
    lrp_model = LRPModel(model)
    r = lrp_model.forward(input_tensor)
    plot_relevance_scores(x=input_tensor, r=r, name=f'explanation_{fr // 4}.png', outdir=des_path)

def Innvestigate_LRP(model, input_tensor, des_path, fr):
    input_tensor = input_tensor*255
    input_tensor = input_tensor.to(torch.uint8)
    inn_model = InnvestigateModel(model, lrp_exponent=2,
                                    method="e-rule",
                                    beta=.5)
    model_prediction, heatmap = inn_model.innvestigate(in_tensor=input_tensor)
    heatmap = heatmap.squeeze(0)
    heatmap_viz = heatmap.mean(dim=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_viz, cmap='viridis')
    plt.axis('off')
    output_file = os.path.join(des_path, f'explanation_{fr // 4}.png')
    plt.savefig(output_file)
    plt.close()


# def Smoothgrad_exp(model, input_tensor, des_path, fr):
#     input_tensor = input_tensor.to(torch.float32)
#     smooth_grad = SmoothGrad(
#         pretrained_model=model,
#         cuda=None,
#         n_samples=100,
#         magnitude=True)
#     smooth_saliency = smooth_grad(input_tensor)
    #save_as_gray_image(smooth_saliency, os.path.join(des_path, f'explanation_{fr // 4}.png'))
 
# def Smoothgrad_exp(input_tensor, model, des_path, fr):
#     input_tensor = input_tensor.to(torch.float32)
#     smooth_grad = SmoothGrad(model=model, cuda=False, sigma=0.20,
#                              n_samples=100, guided=None)
                             
#     smooth_grad.load_image(input_tensor=input_tensor)
#     prob, idx = smooth_grad.forward()

#     # Generate the saliency images of top 3
#     for i in range(0, 3):
#         smooth_grad.generate(idx=idx[i], filename=os.path.join(des_path, f'explanation_{fr // 4}.png'))
# def Smoothgrad_exp(model, input_tensor, des_path, fr):
#     input_tensor = input_tensor.to(torch.float32)
#     ig = IntegratedGradients(model)
#     nt = NoiseTunnel(ig)
#     attribution = nt.attribute(input_tensor, nt_type='smoothgrad', nt_samples=10, target=3)

# def GradientShap_exp(model, input_tensor, des_path, fr):


#     gradient_shap = GradientShap(model)

#     default_cmap = LinearSegmentedColormap.from_list('custom blue', 
#                                                  [(0, '#ffffff'),
#                                                   (0.25, '#000000'),
#                                                   (1, '#000000')], N=256)

#     # Defining baseline distribution of images
#     rand_img_dist = torch.cat([input_tensor * 0, input_tensor * 1])
#     # temp=np.transpose(input_tensor.squeeze().cpu().numpy(), (1,2,0))
#     # print(temp.shape)

#     attributions_gs = gradient_shap.attribute(input_tensor,
#                                             n_samples=50,
#                                             stdevs=0.0001,
#                                             baselines=rand_img_dist,
#                                             target=5)
#     _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
#                                         np.transpose(input_tensor.squeeze().cpu().detach().numpy(), (1,2,0)),
#                                         ["original_image", "heat_map"],
#                                         ["all", "absolute_value"],
#                                         cmap=default_cmap,
#                                         show_colorbar=True)

def Occlusion_exp(model, input_tensor, des_path, fr):
    
    occlusion = Occlusion(model)

    attributions_occ = occlusion.attribute(input_tensor,
                                        strides = (3, 8, 8),
                                        target=3,
                                        sliding_window_shapes=(3,15, 15),
                                        baselines=0)
 
    attribution_sum = np.sum(attributions_occ.squeeze().cpu().detach().numpy(), axis=0)
    print(np.max(attribution_sum))
    
    _ = viz.visualize_image_attr_multiple(np.transpose(attribution_sum, (1,2,0)),
                                      np.transpose(input_tensor.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      show_colorbar=True,
                                      outlier_perc=2)



#checkpoint = "/home/fa578s/Desktop/CSC790_project/atari/Pong/DQN_modern/atari_DQN_modern_Pong_2_model_49750000.gz"
#checkpoint = "/home/fa578s/Desktop/CSC790_project/atari/Assault/DQN_modern/atari_DQN_modern_Assault_2_model_49750000.gz"
#checkpoint = "/home/fa578s/Desktop/CSC790_project/atari/Tennis/DQN_modern/atari_DQN_modern_Tennis_1_model_43000000.gz"
checkpoint = "/home/fa578s/Desktop/CSC790_project/atari/Breakout/DQN_modern/atari_DQN_modern_Breakout_2_model_49250000.gz"
#checkpoint = "/home/fa578s/Desktop/CSC790_project/atari/SpaceInvaders/DQN_modern/atari_DQN_modern_SpaceInvaders_2_model_49500000.gz"
ckp_path = Path(checkpoint) 
game = ckp_path.parts[-3]

# set env
ALE = ALEModern if "_modern/" in checkpoint else ALEClassic
env = ALE(
    game,
    torch.randint(100_000, (1,)).item(),
    sdl=True,
    device="cpu",
    clip_rewards_val=False,
    
    )


# init model
model = AtariNet(env.action_space.n, distributional="C51_" in checkpoint)
# load state
ckpt = _load_checkpoint(checkpoint)
model.load_state_dict(ckpt["estimator_state"])
model.eval()

frame_path = '/home/fa578s/Desktop/CSC790_project/frames_breakout/'
frames = os.listdir(frame_path)
des_path = '/home/fa578s/Desktop/CSC790_project/Lime_explanation/breakout/'
model_name = 'Lime'

#Added
for fr in range(0, len(frames)-4, 4):
    input_images = []
    for i in range(4):
        input_img = f'/home/fa578s/Desktop/CSC790_project/frames_breakout/_frame{fr + i}.jpg'
        rgb_img = cv2.imread(input_img)
        rgb_img = cv2.resize(rgb_img, (84, 84))
        grayscale_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)/255
        input_images.append(grayscale_image)

    if len(input_images) != 4:
        break

    input_img = np.stack(input_images, axis=0)
    input_tensor = torch.tensor(input_img, requires_grad=True)
    input_tensor = input_tensor.unsqueeze(0)

    if model_name == 'Lime':
        Lime_exp(input_tensor, fr, des_path)

    if model_name =='LRP':
        LRP_exp(input_tensor, fr, des_path, model)

    if model_name=='Innvestigate_LRP':
        Innvestigate_LRP(model, input_tensor, des_path, fr)

    # if model_name=='SmoothGrad':
    #     Smoothgrad_exp(model, input_tensor, des_path, fr)
    # if model_name=='Gradientshap':
    #     GradientShap_exp(model, input_tensor, des_path, fr)
    # if model_name=='Occlusion':
    #     Occlusion_exp(model, input_tensor, des_path, fr)
  


# End added



# input_img1 = '/home/fa578s/Desktop/CSC790_project/frames_pong/_frame6.jpg'
# input_img2 = '/home/fa578s/Desktop/CSC790_project/frames_pong/_frame7.jpg'
# input_img3 = '/home/fa578s/Desktop/CSC790_project/frames_pong/_frame8.jpg'
# input_img4 = '/home/fa578s/Desktop/CSC790_project/frames_pong/_frame9.jpg'


# rgb_img1 = cv2.imread(input_img1)
# rgb_img1 = cv2.resize(rgb_img1, (84, 84))
# grayscale_image1 = cv2.cvtColor(rgb_img1, cv2.COLOR_BGR2GRAY)/255

# rgb_img2 = cv2.imread(input_img2)
# rgb_img2 = cv2.resize(rgb_img2, (84, 84))
# grayscale_image2 = cv2.cvtColor(rgb_img2, cv2.COLOR_BGR2GRAY)/255

# rgb_img3 = cv2.imread(input_img3)
# rgb_img3 = cv2.resize(rgb_img3, (84, 84))
# grayscale_image3 = cv2.cvtColor(rgb_img3, cv2.COLOR_BGR2GRAY)/255

# rgb_img4 = cv2.imread(input_img4)
# rgb_img4 = cv2.resize(rgb_img4, (84, 84))
# grayscale_image4 = cv2.cvtColor(rgb_img4, cv2.COLOR_BGR2GRAY)/255

# input_img = np.stack([grayscale_image1, grayscale_image2, grayscale_image3, grayscale_image4], axis=0)
# input_tensor = torch.tensor(input_img)
# input_tensor = input_tensor.unsqueeze(0)
# input_tensor = input_tensor*255
# # input_tensor = input_tensor.to(torch.float32)
# #input_tensor = input_tensor.mean(dim=-1)
# input_tensor = input_tensor.to(torch.uint8)



# # lrp_model = LRPModel(model)
# # r = lrp_model.forward(input_tensor)
# # plot_relevance_scores(x=input_tensor, r=r, name="input_1", outdir="/home/fa578s/Desktop/CSC790_project/lrp_explanation/")


# inn_model = InnvestigateModel(model, lrp_exponent=2,
#                                 method="e-rule",
#                                 beta=.5) 
# model_prediction, heatmap = inn_model.innvestigate(in_tensor=input_tensor)
# heatmap = heatmap.squeeze(0)
# heatmap_viz = heatmap.mean(dim=0)

# plt.figure(figsize=(10, 8))
# sns.heatmap(heatmap_viz, cmap='viridis')
# plt.savefig('frame1.png')



# input_tensor = input_tensor.squeeze(0)
# input = input_tensor.mean(dim=0)



# explainer = lime_image.LimeImageExplainer()
# explanation = explainer.explain_instance(input, 
#                                          prediction_func, # classification function
#                                          top_labels=1, 
#                                          hide_color=255, 
#                                          num_samples=3000)

# # Visualize the explanation
# temp, mask = explanation.get_image_and_mask(
#     label=explanation.top_labels[0],
#     positive_only=True,
#     hide_rest=False,
#     num_features=50,
#     min_weight=0.001,
# )

# plt.imshow(temp)
# plt.imshow(mask, alpha=0.5, cmap='jet')  # Overlay the mask with some transparency
# plt.axis('off')
# plt.show()




# SmoothGrad
# https://github.com/kazuto1011/smoothgrad-pytorch/blob/master/smooth_grad.py