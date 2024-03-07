import os
import  nibabel as nib
import numpy as np
from esFCM import Segment
from esFCM.utils import read_image_with_info, add_noise_to_image
noise = [0, 20, 40]
biase = [0, 5, 10]

def run_experiment(info_image, mask_image):
    for noise_perc in noise:
        noise_perc = noise_perc / 100
        for bias_ratio in biase:
            bias_ratio = bias_ratio / 100

            im_T1, affine, spacing, padding, header, _ = info_image
            if mask_image is None:
                mask = im_T1>0
            else:
                mask = nib.load(mask_image).get_fdata()>0


            if noise_perc > 0:
                noise_type = 'gauss'
                im_T1 = add_noise_to_image(im_T1, mask, noise_type, noise_perc)
            if bias_ratio > 0:
                x_i, y_i, z_i = im_T1.shape
                x_m, y_m, z_m = np.meshgrid(np.arange(x_i), np.arange(y_i), np.arange(z_i), indexing='ij')
                # Define the bias field equation (linear in this case)
                # You can adjust the parameters as needed for your application
                slope_x = 0.02  # Adjust as needed
                slope_y = 0.02  # Adjust as needed
                slope_z = 0.02  # Adjust as needed

                # Calculate the bias field based on the equation
                bias_field = bias_ratio * np.ptp(im_T1) * (
                            np.sin(slope_x * x_m) + np.sin(slope_y * y_m) + np.sin(slope_z * z_m))
                # Ensure the bias field is within the desired intensity range (0 to 1000)
                bias_field = np.clip(bias_field, 0, 1000)
                im_T1 = im_T1 + bias_field
                im_T1 = np.clip(im_T1, 0, 1000)
                im_T1[~mask] = 0
            im_T1[~(mask > 0)] = 0



            seg_esFCM = Segment(nib.Nifti1Image(im_T1, affine, header), mask=mask, num_tissues=3, max_iter=50, max_fail=4,
                                method='esFCM')


            seg_FCM = Segment(nib.Nifti1Image(im_T1, affine, header), mask=mask, num_tissues=3, max_iter=50, max_fail=4,
                                method='FCM')

            seg_gmm = Segment(nib.Nifti1Image(im_T1, affine, header), mask=mask, num_tissues=3, max_iter=50, max_fail=4,
                                method='gmm')
            """
            To run fsl and ants algorithms
            cmd = 'fast -t 1 -n 3 -v -o {} {}'.format(output_image_fsl, input_image)
            cmd = 'Atropos -d 3 -a {} -i kmeans[3] -x {} -o {} -v'.format(input_image, mask_image, output_image_ants)
            """



if __name__ == '__main__':
    file_im = '../data/input.nii.gz'
    info_image = read_image_with_info(file_im)
    run_experiment(info_image, mask_image=None)