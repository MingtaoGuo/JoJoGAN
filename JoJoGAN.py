import torch
import torch.optim as optim
import torch.nn.functional as F

from models.psp import pSp
from models.stylegan2.model import Discriminator
from face_detection.yoloface import yolov5

from skimage import transform as trans
from PIL import Image 
import numpy as np

import cv2 
import copy 
from tqdm import tqdm
import argparse
import os 

def crop_with_ldmk(img, ldmk, size=256):
    std_ldmk = np.array([[193, 240], [319, 240],
                         [257, 314], [201, 371],
                         [313, 371]], dtype=np.float32) / 512 * size
    tform = trans.SimilarityTransform()
    tform.estimate(ldmk, std_ldmk)
    M = tform.params[0:2, :]
    cropped = cv2.warpAffine(img, M, (size, size), borderValue=0.0)
    invert_param = cv2.invertAffineTransform(M)
    return cropped, invert_param


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def setup_model(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']

    opts['checkpoint_path'] = checkpoint_path
    opts['device'] = device
    opts = argparse.Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net = net.to(device)
    return net, opts


class Train_JoJoGAN():
    def __init__(self, stylegan2_model="stylegan2-ffhq-config-f.pt", psp_model="e4e_ffhq_encode.pt", yolo_model="yolov5s-face.onnx", device="cuda") -> None:
        self.device = device
        self.discriminator = Discriminator(1024, 2).eval().to(device)
        self.net, self.opts = setup_model(psp_model)
        self.generator = self.net.decoder
        self.generator.train()
        self.original_generator = copy.deepcopy(self.generator)
        self.original_generator.eval()
        self.discriminator.load_state_dict(torch.load(stylegan2_model)["d"], strict=False)
        self.yolonet = yolov5(yolo_model, confThreshold=0.3, nmsThreshold=0.5, objThreshold=0.3)
        self.optim = optim.Adam(self.generator.parameters(), lr=2e-3, betas=(0, 0.99))

    def train(self, styles_dir, save_img_dir, save_model_dir, num_itr, style_mix_idx=7):
        files = os.listdir(styles_dir)
        batchsize = len(files)
        styles_256 = np.zeros([batchsize, 3, 256, 256])
        styles_1024 = np.zeros([batchsize, 3, 1024, 1024])
        for idx, file in enumerate(files):
            img = np.array(Image.open(os.path.join(styles_dir, file)))[..., :3]
            dets = self.yolonet.detect(img)
            dets = self.yolonet.postprocess(img, dets)
            [confidence, bbox, landmark] = dets[0]
            landmark = landmark.reshape([5, 2])
            cropped, inv = crop_with_ldmk(img, landmark, size=1024)
            styles_256[idx] = np.transpose(cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_AREA), axes=[2, 0, 1])
            styles_1024[idx] = np.transpose(cv2.resize(cropped, (1024, 1024), interpolation=cv2.INTER_AREA), axes=[2, 0, 1])
        styles_256_tensor = torch.tensor(styles_256, dtype=torch.float32).to(self.device) / 127.5 - 1.0
        styles_1024_tensor = torch.tensor(styles_1024, dtype=torch.float32).to(self.device) / 127.5 - 1.0

        with torch.no_grad():
            style_latent_codes = get_latents(self.net, styles_256_tensor)

        for i in range(num_itr+1):
            z = torch.randn(batchsize, 512).cuda()
            z_latent = self.generator.style(z)[:, None, :]
            z_latent = torch.repeat_interleave(z_latent, style_latent_codes.shape[1], dim=1)
            styles_mix = style_latent_codes.detach()
            styles_mix[:, style_mix_idx:] = z_latent.detach()[:, style_mix_idx:]

            fake, _ = self.generator([styles_mix], input_is_latent=True, randomize_noise=False, return_latents=True)
            with torch.no_grad():
                real_feat = self.discriminator(styles_1024_tensor)
            fake_feat = self.discriminator(fake)

            loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)])/len(fake_feat)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if i % 100 == 0:
                fake = (fake.permute(0, 2, 3, 1).clamp(-1, 1).detach().cpu().numpy()[0] + 1) * 127.5
                Image.fromarray(np.uint8(fake)).save(f"{save_img_dir}/{str(i)}.png")
                print(f"Iteration: {i}, loss: {loss.item()}")
        torch.save(self.generator.state_dict(), f"{save_model_dir}/stylize_stylegan.pth")

class Infer_JoJoGAN():
    def __init__(self, stylize_model, psp_model="./e4e_ffhq_encode.pt",  device="cuda") -> None:
        self.device = device
        self.net, self.opts = setup_model(psp_model)
        self.generator = self.net.decoder
        self.generator.eval()
        self.original_generator = copy.deepcopy(self.generator)
        self.original_generator.eval()
        self.generator.load_state_dict(torch.load(stylize_model))

    def inference(self, num_sample, save_dir):
        # mean_latent = self.generator.mean_latent(4096)
        for i in tqdm(range(num_sample)):
            z = torch.randn(1, 512).cuda()
            with torch.no_grad():
                # fake1 = self.generator([z], truncation=0.5, truncation_latent=mean_latent)[0]
                fake1 = self.generator([z])[0]
                fake2 = self.original_generator([z])[0]
                fake = torch.cat([fake2, fake1], dim=3)
                fake = (fake.clamp(-1, 1).permute(0, 2, 3, 1).detach().cpu().numpy()[0] + 1) * 127.5
                Image.fromarray(np.uint8(fake)).save(f"{save_dir}/{i}.png")

