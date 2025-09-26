# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import os
import cv2

# Helpers

def split_bgr(im):
    h = im.shape[0] // 3
    b, g, r = im[:h, :], im[h:2*h, :], im[2*h:3*h, :]
    return b, g, r

def roll2d(im, dx, dy):
    return np.roll(np.roll(im, dy, axis=0), dx, axis=1)

def center_crop(im, crop_frac):
    h, w = im.shape[:2]
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    if ch >= h // 2 or cw >= w // 2:
        return im
    return im[ch:h-ch, cw:w-cw]

def downsample(im):

    im = np.asarray(im, dtype=np.float32)
    out = sk.transform.rescale(im, 0.5, anti_aliasing=True, preserve_range=True)
    return out.astype(np.float32)

def get_overlap(ref, img, dx, dy):
    h, w = ref.shape
    x_start = max(0, dx)
    x_end = min(w, w + dx) if dx < 0 else min(w, w - dx)
    y_start = max(0, dy)
    y_end = min(h, h + dy) if dy < 0 else min(h, h - dy)
    ref_crop = ref[y_start:y_end, x_start:x_end]
    img_crop = img[y_start-dy:y_end-dy, x_start-dx:x_end-dx]
    return ref_crop, img_crop

# Metrics

def ssd(im1, im2): # sum of squared differences, works the same way as L2 norm
    return -np.mean((im1 - im2)**2)

def ncc(im1, im2):
    im1m = im1 - np.mean(im1)
    im2m = im2 - np.mean(im2)
    return np.sum(im1m * im2m) / np.sqrt(np.sum(im1m**2) * np.sum(im2m**2))

def get_metric(metric_name):
    if metric_name == 'ssd':
        return ssd
    elif metric_name == 'ncc':
        return ncc
    else:
        raise ValueError('Unknown metric: ' + metric_name)
    
def best_shift_single_scale(ref, img, radius, metric, border_frac):
    metric_fn = get_metric(metric)
    
    best_score = -np.inf
    best_dx, best_dy = 0, 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # rolled = roll2d(img, dx, dy)
            ref_crop, rolled_crop = get_overlap(ref, img, dx, dy)
            ref_cropped = center_crop(ref_crop, border_frac)
            rolled_cropped = center_crop(rolled_crop, border_frac)
            if ref_cropped.size == 0 or rolled_cropped.size == 0:
                continue
            score = metric_fn(ref_cropped, rolled_cropped)
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy
    return best_dx, best_dy

def pyramid_align(ref, img, metric='ncc', base_radius=15, max_levels=None, stop_at_min=64, border_frac=0.20):
    pyr_ref = [ref]
    pyr_img = [img]
    while True:
        r = pyr_ref[-1]
        if max_levels is not None and len(pyr_ref) >= max_levels:
            break
        if min(r.shape[0], r.shape[1]) <= stop_at_min:
            break
        pyr_ref.append(downsample(r))
        pyr_img.append(downsample(pyr_img[-1]))
    L = len(pyr_ref) - 1
    dx, dy = 0, 0
    for l in range(L, -1, -1):
        r = pyr_ref[l]
        i = pyr_img[l]
        print(f"Pyramid level {l}: shape={r.shape}, radius={max(base_radius // (2**l), 3)}")

        i_est = roll2d(i, dx, dy)
        radius = max(base_radius // (2**l), 3)
        ddx, ddy = best_shift_single_scale(r, i_est, radius, metric, border_frac)
        dx += ddx
        dy += ddy
        print(f"Level {l}: dx={dx}, dy={dy}")
        if l > 0:
            dx *= 2
            dy *= 2
    return dx, dy

def align(moving, reference, use_pyramid=True, metric='ncc', base_radius=15, border_frac=0.20):
    if use_pyramid:
        dx, dy = pyramid_align(reference, moving, metric, base_radius, border_frac=border_frac)
    else:
        dx, dy = best_shift_single_scale(reference, moving, base_radius, metric, border_frac)
    aligned = roll2d(moving, dx, dy)
    return aligned, dx, dy

if __name__ == '__main__':
    # gets all images in the input directory, aligns them, and saves to output directory
    input_dirs = ['cs180_proj1_data', 'cs180_proj1_own_data']  
    output_dir = 'output_images'         

    os.makedirs(output_dir, exist_ok=True)

    for input_dir in input_dirs:
        for fname in os.listdir(input_dir):
            print(f"Processing {fname}...")
            if fname.lower().endswith(('.jpg')):
                im_path = os.path.join(input_dir, fname)
                # im_path = 'cs180_proj1_data/emir.tif'  # Example image path
                im = skio.imread(im_path)
                im = sk.img_as_float(im).astype(np.float32)
                b, g, r = split_bgr(im)
                ab, dx_b_pyr, dy_b_pyr = align(b, g, use_pyramid=False, metric='ncc', base_radius=15, border_frac=0.20)
                ar, dx_r_pyr, dy_r_pyr = align(r, g, use_pyramid=False, metric='ncc', base_radius=15, border_frac=0.20)
                print(f"Blue channel offsets: dy={dy_b_pyr}, dx={dx_b_pyr}")
                print(f"Red channel offset: dy={dy_r_pyr}, dx={dx_r_pyr}")
                im_out = np.dstack([ar, g, ab])
                # out_path = os.path.join(output_dir, f'out_{fname}')
                # skio.imsave(out_path, im_out)
                skio.imshow(im_out)
                skio.show()
            elif fname.lower().endswith(('.tif')):
                
                im_path = os.path.join(input_dir, fname)
                # im_path = 'cs180_proj1_data/emir.tif'  # Example image path
                im = skio.imread(im_path)
                im = sk.img_as_float(im).astype(np.float32)
                b, g, r = split_bgr(im)
                ab, dx_b_pyr, dy_b_pyr = align(b, g, use_pyramid=True, metric='ncc', base_radius=15, border_frac=0.30)
                ar, dx_r_pyr, dy_r_pyr = align(r, g, use_pyramid=True, metric='ncc', base_radius=15, border_frac=0.30)
                print(f"Blue channel offsets: dy={dy_b_pyr}, dx={dx_b_pyr}")
                print(f"Red channel offset: dy={dy_r_pyr}, dx={dx_r_pyr}")
                im_out = np.dstack([ar, g, ab])
                # out_path = os.path.join(output_dir, f'out_{fname}')
                # skio.imsave(out_path, im_out)
                skio.imshow(im_out)
                skio.show()