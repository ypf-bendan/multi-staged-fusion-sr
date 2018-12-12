import argparse, os
import torch
from torch.autograd import Variable
from scipy.ndimage import imread
from scipy.misc import imsave
from PIL import Image
import numpy as np
import time, math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="PyTorch VDSR Demo")
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda?")
parser.add_argument("--model", default="checkpoint/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--scale", default=3, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def block_patch(image, scale):
    blocks = []
    (width, height) = image.shape
    block1 = image[0:(int(width/2)+scale), 0:(int(height/2)+scale)]
    block2 = image[(int(width/2)-scale):width, 0:(int(height/2)+scale)]
    block3 = image[0:(int(width/2)+scale), (int(height/2)-scale):height]
    block4 = image[(int(width/2)-scale):width, (int(height/2)-scale):height]

    #block1 = image[0:int(width/2), 0:int(height/2)]
    #block2 = image[int(width/2):width, 0:int(height/2)]
    #block3 = image[0:int(width/2), int(height/2):height]
    #block4 = image[int(width/2):width, int(height/2):height]

    blocks.append(block1)
    blocks.append(block2)
    blocks.append(block3)
    blocks.append(block4)

    return blocks

def crop_by_pixel(image):
    size = image.shape
    if size[0] % 2 != 0:
        image = image[1:size[0], 0:size[1]]

    size = image.shape
    if size[1] % 2 != 0:
        image = image[0:size[0], 1:size[1]]

    return image


def merge_block(image, blocks, scale):
    (width, height) = image.shape
    print(str(width)+" "+ str(height))
    image[0:int(width/2), 0:int(height/2)] = (blocks[0])[0, 0:int(width/2), 0:int(height/2)]
    image[int(width/2):width, 0:int(height/2)] = (blocks[1])[0, scale:int(width/2)+scale, 0:int(height/2)]
    image[0:int(width/2), int(height/2):height] = (blocks[2])[0, 0:int(width/2), scale:int(height/2)+scale]
    image[int(width/2):width, int(height/2):height] = (blocks[3])[0, scale:int(width/2)+scale, scale:int(height/2)+scale]

    return image
    
def colorize(y, ycbcr): 
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img

opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")


model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

im_gt_ycbcr = imread("mf_test/" + opt.image + ".png", mode="YCbCr")
im_b_ycbcr = imread("mf_test/"+ opt.image + "_scale_x"+ str(opt.scale) + ".png", mode="YCbCr")

img_gt_crop = np.zeros((672, 1022, 3), np.uint8)
img_bi_crop = np.zeros((672, 1022, 3), np.uint8)

img_gt_crop[:,:,0] = crop_by_pixel(im_gt_ycbcr[:,:,0])
img_gt_crop[:,:,1] = crop_by_pixel(im_gt_ycbcr[:,:,1])
img_gt_crop[:,:,2] = crop_by_pixel(im_gt_ycbcr[:,:,2])

img_bi_crop[:,:,0] = crop_by_pixel(im_b_ycbcr[:,:,0])
img_bi_crop[:,:,1] = crop_by_pixel(im_b_ycbcr[:,:,1])
img_bi_crop[:,:,2] = crop_by_pixel(im_b_ycbcr[:,:,2])

im_gt_ycbcr = img_gt_crop
im_b_ycbcr = img_bi_crop
    
im_gt_y = im_gt_ycbcr[:,:,0].astype(float)
im_b_y = im_b_ycbcr[:,:,0].astype(float)


print(im_gt_y.shape)
print(im_b_y.shape)

psnr_bicubic = PSNR(im_gt_y, im_b_y,shave_border=opt.scale)

im_input = im_b_y/255.

blocks = []
generate = []

generate_image = np.zeros(im_gt_y.shape)
generate_image = generate_image.astype(float)
print(generate_image.shape)

blocks = block_patch(im_input, 3)

for i in range(4):

    im_input = Variable(torch.from_numpy(blocks[i]).float()).view(1, -1, blocks[i].shape[0], blocks[i].shape[1])

    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    out = model(im_input)
    elapsed_time = time.time() - start_time

    out = out.cpu()

    im_h_y = out.data[0].numpy().astype(np.float32)

    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.

    generate.append(im_h_y) 
    print(im_h_y.shape)  
    del out,   im_h_y   

im_h_y = merge_block(generate_image, generate, 3)

psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=opt.scale)

im_h = colorize(im_h_y, im_b_ycbcr)
im_gt = Image.fromarray(im_gt_ycbcr, "YCbCr").convert("RGB")
im_b = Image.fromarray(im_b_ycbcr, "YCbCr").convert("RGB")

print("Scale=",opt.scale)
print("PSNR_predicted=", psnr_predicted)
print("PSNR_bicubic=", psnr_bicubic)
print("It takes {}s for processing".format(elapsed_time))

fig = plt.figure()
ax = plt.subplot("131")
ax.imshow(im_gt)
ax.set_title("GT")

ax = plt.subplot("132")
ax.imshow(im_b)
ax.set_title("Input(bicubic)")

ax = plt.subplot("133")
ax.imshow(im_h)
ax.set_title("Output(vdsr)")

imsave('pic_test/img061_MFNet_x3_gt.bmp', im_gt)
imsave('pic_test/img061_MFNet_x3_bi.bmp', im_b)
imsave('pic_test/img061_MFNet_x3.bmp', im_h)
plt.show()
