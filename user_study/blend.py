import glob
import cv2
import os
import numpy as np

def compute_sdf(mask, winsize=2):
    height, width = mask.shape[0:2]
    ycoor = np.tile(np.arange(height).reshape(-1,1),(1,width)).reshape(-1)
    xcoor = np.tile(np.arange(width),(height,1)).reshape(-1)
    features = []
    for y in np.arange(-winsize,winsize+1):
        for x in np.arange(-winsize,winsize+1):
            # outside
            ytmp = ycoor+y
            xtmp = xcoor+x
            ytmp = np.maximum(0,np.minimum(height-1,ytmp))
            xtmp = np.maximum(0,np.minimum(width-1,xtmp))
            features.append(mask[ytmp, xtmp])
    features = np.stack(features, -1)
    features = np.mean(features,-1)
    soft_mask = features.reshape(height,width)

    kernel = np.ones((3,3), np.uint8)
    thin_mask = cv2.erode(mask, kernel, iterations=1)
    thin_mask = cv2.dilate(thin_mask, kernel, iterations=1)
    thin_mask = np.absolute(mask-thin_mask)

    soft_mask = (1-thin_mask)*np.maximum(2*(soft_mask-0.5),0)+thin_mask*soft_mask
    return soft_mask

img_list = glob.glob("./paste_ours/*direct_paste.png")
method = "neural_illu"
for p in img_list:
	prefix = os.path.basename(p).split('-')[0]
	scene_rgb = cv2.imread(f"./paste_ours/{prefix}-scene_rgb.png")
	hole_mask = cv2.imread(f"./paste_ours/{prefix}-hole_mask.png",0)/255
	scene_composite = cv2.imread(f"./{method}/{prefix}_place.png")
	soft_mask = compute_sdf(hole_mask, 2)
	soft_mask = cv2.merge([soft_mask]*3)
	scene_composite = soft_mask*scene_composite+(1-soft_mask)*scene_rgb
	cv2.imwrite(f"./{method}/{prefix}_place_blend.png", scene_composite)