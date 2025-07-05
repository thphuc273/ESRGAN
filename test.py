import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from skimage.metrics import structural_similarity as ssim

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_img_folder = 'LR/*'
hr_img_folder = 'HR/*'

psnr_list = []
ssim_list = []

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)
    
    hr_path = osp.join('HR', base.replace('_LR', '_HR') + '.png')
    original = cv2.imread(hr_path, cv2.IMREAD_COLOR)
    
    # if original.shape != output.shape:
    #     print(f'Skipping {base} due to shape mismatch')
    #     continue
    
    psnr_value = cv2.PSNR(original, output)
    psnr_list.append(psnr_value)
    print(f"PSNR của {base}: {psnr_value:.2f} dB")
    
    ssim_value = ssim(original, output, multichannel=True, channel_axis=2)
    ssim_list.append(ssim_value)
    print(f"SSIM của {base}: {ssim_value:.2f} dB")
    
# Tính giá trị trung bình
avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)
print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")