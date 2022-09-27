import io
import numpy as np
import cv2


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])
    
def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = b
        cmap[i, 1] = g
        cmap[i, 2] = r
    return cmap

def fig_to_numpy(fig):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr

def normalize_img(img_, min_, max_):
    img_ = (img_-min_)/(max_-min_)
    return img_

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

def compute_sdf1(mask, n_iter):
    scale = 1.0 / n_iter
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=4)
    mask_out = mask.copy().astype(np.float32)
    for i in range(n_iter):
        mask_new = cv2.dilate(mask, kernel, iterations = 1)
        mask_out[(mask_new - mask) > 0] = scale * (n_iter - i)
        mask = mask_new
    return mask_out


def compute_metrics(img1, img2):
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)
    if len(img1.shape)==3 or img1.shape[2]==3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    l1 = np.average(np.absolute(img1.astype(np.float)-img2.astype(np.float)))
    p = psnr(img1, img2)
    s,_= ssim(img1, img2)
    return l1, p, s


def test():
    path = "/cs/vml-furukawa/user/yimingq/cutpaste/attention/experiments/250_1e6/results/inpaint-ckpt-200/aa/"
    img1 = cv2.imread(path+"0-scene_rgb.png")
    img2 = cv2.imread(path+"0-rgb_output.png")
    a = np.absolute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)-cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    cv2.imwrite(path+"diff.png", a)
    # img1 = (np.ones((5,5,3))*255).astype(np.uint8)
    # img2 = (np.ones((5,5,3))*125).astype(np.uint8)
    #img1, img2 = img1.astype(np.uint8), img2.astype(np.uint8)
    

    print(np.mean(np.absolute(img1-img1)))
    print(compute_metrics(img1, img2))

if __name__ == "__main__":
    test()


