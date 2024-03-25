# -*- coding: UTF-8 -*-
"""
@author: Luca Bondi (luca.bondi@polimi.it)
@author: Paolo Bestagini (paolo.bestagini@polimi.it)
@author: Nicolò Bonettini (nicolo.bonettini@polimi.it)
Politecnico di Milano 2018
"""

import os
from glob import glob
from multiprocessing import cpu_count, Pool
import itertools

import numpy as np
from PIL import Image

import prnu

def map_dataset(directory):
    structure = {}
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            structure[folder_name] = {
                "flat": [],
                "nat": [],
            }
            flat_folder = os.path.join(folder_path, "flat")
            for file_name in os.listdir(flat_folder):
                if file_name.endswith(".jpg"):
                    structure[folder_name]["flat"].append(os.path.join(flat_folder, file_name))
            nat_folder = os.path.join(folder_path, "nat")
            for file_name in os.listdir(nat_folder):
                if file_name.endswith(".jpg"):
                    structure[folder_name]["nat"].append(os.path.join(nat_folder, file_name))
    return structure

def pce(a, b):
    # Check if the matrices have the same shape
    if a.shape != b.shape:
        raise ValueError("Matrices must have the same shape.")
    # Compute the cross-correlation between the matrices
    correlation = np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(b).conj()).real
    # Find the peak value of the correlation
    peak_value = np.max(correlation)
    # Compute the average correlation energy
    correlation_energy = np.sum(np.abs(correlation)) / np.prod(a.shape)
    # Compute the peak to correlation energy ratio
    pce = peak_value / correlation_energy
    return pce

def main():
    device_set = map_dataset("./data/devices/")
    """
    Main example script. Load a subset of flatfield and natural images from Dresden.
    For each device compute the fingerprint from all the flatfield images.
    For each natural image compute the noise residual.
    Check the detection performance obtained with cross-correlation and PCE
    :return:
    """
    ff_dirlist = np.array(sorted(glob('test/data/ff-jpg/*.JPG')))
    ff_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in ff_dirlist])

    nat_dirlist = np.array(sorted(glob('test/data/nat-jpg/*.JPG')))
    nat_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in nat_dirlist])

    print('Computing fingerprints')
    for device, device_data in device_set.items():
        print(f"Computing fingerprint for {device}")
        fingerprint = prnu.get_fingerprint_from_images(device_data["flat"], slice_size=50)
        device_data["flat_fingerprint"] = fingerprint

        # for nat_img_path in device_data["nat"]:
        #     nat_img = Image.open(nat_img_path)
        #     nat_img = prnu.cut_ctr(np.asarray(nat_img), (224, 224, 3))
        #     nat_fingerprint = prnu.extract_single(nat_img)
        #     cc = prnu.crosscorr_2d(nat_fingerprint, fingerprint)
        #     pce_result = prnu.pce(cc)
        #     print(pce_result)
        # slice_size = 20
        # print(f"Computing fingerprint for {slice_size} test images of {device}")
        # nat_images = []
        # paths = device_data["nat"][:slice_size]
        # for img_path in paths:
        #     nat_images.append(
        #         prnu.cut_ctr(np.asarray(Image.open(img_path)), (224, 224, 3))
        #     )

        # pool = Pool(cpu_count())
        # w = pool.map(prnu.extract_single, nat_images)
        # pool.close()
        # device_data["nat_fingerprint"] = w
    
    img_path = device_set["D18_Apple_iPhone5c"]["nat"][124]
    img = Image.open(img_path)
    img = prnu.cut_ctr(np.asarray(img), (224, 224, 3))
    results = {}
    for device, device_data in device_set.items():
        fingerprint = device_data["flat_fingerprint"]
        nat_fingerprint = prnu.extract_single(img)
        cc = prnu.crosscorr_2d(nat_fingerprint, fingerprint)
        pce = prnu.pce(cc)
        results[device] = pce
    print(results)
    return


    fingerprints_from_flat = np.stack(fingerprints_from_flat, 0)

    print('Computing residuals')

    imgs = []
    for img_path in nat_dirlist:
        imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512, 3))]

    pool = Pool(cpu_count())
    w = pool.map(prnu.extract_single, imgs)
    pool.close()

    w = np.stack(w, 0)

    # Computing Ground Truth
    gt = prnu.gt(fingerprint_device, nat_device)

    print('Computing cross correlation')
    cc_aligned_rot = prnu.aligned_cc(fingerprints_from_flat, w)['cc']

    
    print('Computing statistics cross correlation')
    stats_cc = prnu.stats(cc_aligned_rot, gt)

    print('Computing PCE')
    pce_rot = np.zeros((len(fingerprint_device), len(nat_device)))

    for fingerprint_idx, fingerprint_k in enumerate(fingerprints_from_flat):
        for natural_idx, natural_w in enumerate(w):
            cc2d = prnu.crosscorr_2d(fingerprint_k, natural_w)
            pce_rot[fingerprint_idx, natural_idx] = prnu.pce(cc2d)['pce']

    print('Computing statistics on PCE')
    stats_pce = prnu.stats(pce_rot, gt)

    print('AUC on CC {:.2f}, expected {:.2f}'.format(stats_cc['auc'], 0.98))
    print('AUC on PCE {:.2f}, expected {:.2f}'.format(stats_pce['auc'], 0.81))


if __name__ == '__main__':
    main()
