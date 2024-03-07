__AUTHOR__ = 'Bahram Jafrasteh'


from esFCM.FCM import esFCM
from esFCM.FCM import FCM_pure as FCM
from esFCM.utils import *


def run_gmm(img, num_tissues=3):
    from sklearn.mixture import GaussianMixture
    # Number of tissue types
    n_components = num_tissues+1
    im_T1 = img.get_fdata()
    data = im_T1.get_fdata().reshape((-1, 1))
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag')
    try:
        gmm.fit(data)
    except:
        pass

    # Predict tissue types for each voxel
    predicted_labels = gmm.predict(data)

    # Reshape predicted labels back into 3D
    predicted_labels_3d = predicted_labels.reshape(im_T1.shape)

    Centers = gmm.means_.flatten()
    sortedC = Centers.argsort()
    sorted_el = [sortedC[i] for i in range(4)]
    if sorted_el != [0, 1, 2, 3]:
        predicted_labels_3d = label_mapping(predicted_labels_3d, sorted_el)
    return predicted_labels_3d
def Segment(img, mask,  num_tissues=3,max_iter =100, max_fail=3, method='esFCM'):
    """
    img: Nifti image (3D)
    mask: numpy mask (3D)
    num tissue : number of tissues
    max_iter : maximum number of iterations
    max_fail : maximum number of try for bias field correction before failing
    Segmentation
    """
    image_used, pad_zero = remove_zero(img.get_fdata(), 0)
    if mask is None:
        mask = image_used > 0
    else:
        mask = mask[pad_zero[0][0]:pad_zero[0][1] + 1, pad_zero[1][0]:pad_zero[1][1] + 1,
             pad_zero[2][0]:pad_zero[2][1] + 1]
    if method.lower() == 'gmm':
        seg1 =  run_gmm(img, num_tissues)
    elif 'fcm' in method.lower():
        if method.lower() == 'esfcm':
            model = esFCM(image_used, img.affine, num_tissues=num_tissues, max_iter=max_iter,
                      mask=mask, max_fail= max_fail)
        elif method.lower() == 'fcm':
            model = FCM(image_used, img.affine, num_tissues=num_tissues, max_iter=max_iter,
                      mask=mask)

        try:
            model.initialize_fcm(initialization_method='otsu')
        except:
            model.initialize_fcm(initialization_method='kmeans')

        model.fit()
        seg1 = model.predict(use_softmax=True).astype('int')
    else:
        raise ValueError('Method must be "gmm", "fcm" or "esfcm"')
    fina_seg = np.zeros(img.shape)
    fina_seg[pad_zero[0][0]:pad_zero[0][1] + 1, pad_zero[1][0]:pad_zero[1][1] + 1,
    pad_zero[2][0]:pad_zero[2][1] + 1] = seg1
    return nib.Nifti1Image(fina_seg, img.affine, img.header)