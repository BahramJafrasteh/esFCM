from nibabel.affines import apply_affine
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
import torch.nn.functional as F
from skimage.measure import label as label_connector
import nibabel as nib


def remove_zero(f_data, value=0):
    """
    Remove non segmented areas from image
    :param f_data:
    :param value:
    :return:
    """

    xs, ys, zs = np.where(f_data > value)  # find zero values
    tol = 4

    min_max = []
    for x in [xs, ys, zs]:
        minx = min(x) - tol if min(x) - tol > 1 else min(x)
        maxx = max(x) + tol if max(x) + tol < f_data.shape[0] - 1 else max(x)
        min_max.append([minx, maxx])
    f_data = f_data[min_max[0][0]:min_max[0][1] + 1, min_max[1][0]:min_max[1][1] + 1, min_max[2][0]:min_max[2][1] + 1]

    return f_data, min_max

def label_mapping(curr_label, sorted_el):
    seg_label = np.zeros_like(curr_label)
    for curr_l, new_l in enumerate(sorted_el):
        ind_curr = curr_label == new_l
        seg_label[ind_curr] = curr_l
    return seg_label
def add_noise_to_image(im_t1, mask, noise_type, noise_perc):
    if noise_type == "gauss":
        row, col, ch = im_t1.shape
        mean = 0
        sigma = noise_perc*np.std(im_t1)
        np.random.seed(0)
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        gauss[~mask]=0
        im_t1[~mask]=0
        im_t1 = im_t1 + gauss
    return im_t1
def read_image_with_info(T1):
    imT1_t = nib.load(T1)
    subject = os.path.basename(T1)
    if imT1_t.ndim == 4:
        imT1_t = nib.funcs.four_to_three(imT1_t)[0]
    imT1 = imT1_t.get_fdata().copy()
    header = imT1_t.header
    affine = imT1_t.affine
    spacing = imT1_t.header['pixdim'][1:4]
    a, b = 0, 1000
    dif = b - a
    mindata = imT1.min()
    maxdata = imT1.max()
    im_T1 = a + (((imT1 - mindata) * dif) / (maxdata - mindata))
    padding = im_T1.min()

    return im_T1, affine, spacing, padding, header, subject

class BiasCorrection(object):
    def __init__(self):
        pass
    def set_info(self, target, reference, weight, biasfield, padding, mask, affine):
        self.target = target
        self.reference = reference
        self.weight = weight
        self.biasfield = biasfield
        self.padding = padding
        self.mask = mask
        self.affine = affine
        self.scaler = RobustScaler()
        self.scalerW = StandardScaler()
        self.scalery = StandardScaler()



    def _weighted_leas_square(self, imcoord, realdcoord, bias, weights,
                              sample_every= None):

        if sample_every is not None:
            vecB = bias[::sample_every]
            weight = weights[::sample_every]
            realdcoord = realdcoord[::sample_every, :]
        else:
            vecB = bias
            weight = weights

        A = self.biasfield.fit_transform(realdcoord)


        A = self.scaler.fit_transform(A)

        weight = (weight - weight.min())/np.ptp(weight)
        vecB = self.scalery.fit_transform(vecB.reshape(-1, 1)).squeeze()
        WLS = LinearRegression()

        WLS.fit(A, vecB, sample_weight=1-weight)
        return WLS
    def normalize(self, fi, source):
        a, b = source.min(), source.max()
        dif = b - a
        mindata = fi.min()
        maxdata = fi.max()
        filtered_image = a + (((fi - mindata) * dif) / (maxdata - mindata))
        return filtered_image
    def Apply(self, x, weight=None):
        # apply bias field correction on image
        ind_non_padd = (x != self.padding)* (self.mask==1)
        coord = np.argwhere(ind_non_padd)
        world = apply_affine(self.affine, coord)
        A = self.biasfield.transform(world)

        A = self.scaler.transform(A)
        res = self.scalery.inverse_transform(self.coef.predict(A).reshape(-1,1)).squeeze()

        if weight is not None:
            x[ind_non_padd] = x[ind_non_padd] -weight[ind_non_padd]*res
        else:
            x[ind_non_padd] = x[ind_non_padd] -res
        return x

    def Run(self):
        index_selected = (self.target!= self.padding)* (self.mask==1)
        imcoord = np.argwhere(index_selected)
        realdcoord = apply_affine(self.affine, imcoord)


        bias= (self.target[index_selected] - self.reference[index_selected])#/np.median(self.target[self.target>0])

        weights = self.weight[index_selected]
        # wheighted least square for bias field which polynomial here
        self.coef = self._weighted_leas_square(imcoord, realdcoord, bias=bias, weights=weights, sample_every=None)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))/(np.sqrt(2*np.pi)*sigma) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window_3D(window_size, channel):
    from torch.autograd import Variable
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def ssim3D(i1, i2, window, window_size, channel, contrast=True, L=1):
    img1 =torch.from_numpy(i1).unsqueeze(0).unsqueeze(0).to(torch.float)
    img2 = torch.from_numpy(i2).unsqueeze(0).unsqueeze(0).to(torch.float)
    mux = F.conv3d(img1, window, padding=window_size // 2, groups=channel) #Overall Mean Luminance im1
    muy = F.conv3d(img2, window, padding=window_size // 2, groups=channel)#Overall Mean Luminance im2
    mux_sq = mux.pow(2)
    muy_sq = muy.pow(2)
    # Constants for SSIM calculation
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2


    mux_muy = mux * muy

    sigmax_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mux_sq
    sigmax_sq = np.clip(sigmax_sq, 0, sigmax_sq.max())
    sigmay_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - muy_sq
    sigmay_sq = np.clip(sigmay_sq, 0, sigmay_sq.max())
    sigmaxy = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mux_muy
    # structural similarity
    #ssim_map = (sigmaxy + C1) / (sigmax_sq.sqrt() * sigmay_sq.sqrt() + C1)
    #Luminance
    #ssim_map = (2 * mux * muy + C1) / (mux** 2 + muy** 2 + C1)#contrast
    if contrast:
        ssim_map = (2 * sigmax_sq.sqrt() * sigmay_sq.sqrt() + C1) / (sigmax_sq + sigmay_sq + C1)
    else:
        ssim_map = ((2 * mux_muy + C1) * (2 * sigmaxy + C2)) / ((mux_sq + muy_sq + C1) * (sigmax_sq + sigmay_sq + C2))
    ssim_map = ssim_map.squeeze().detach().cpu().numpy()
    return ssim_map

def LargestCC(segmentation, connectivity=3):
    """
    Get largets connected components
    """
    ndim = 3
    if segmentation.ndim == 4:
        segmentation = segmentation.squeeze(-1)
        ndim = 4
    labels = label_connector(segmentation, connectivity=connectivity)
    frequency = np.bincount(labels.flat)
    # frequency = -np.sort(-frequency)
    return labels, frequency



def neighborhood_conv(output, kerenel_size=3, direction ='x',sqr2dist=False):
    # compute neighborhood pixels
    if sqr2dist: # average of all neighborhood pixels in 3D
        kernel_used = torch.from_numpy(np.zeros(kerenel_size).astype(np.float32))  # mean
        threed_kernel = torch.einsum('i,j,k->ijk', kernel_used, kernel_used, kernel_used)  # mean

        if direction=='xz':
            threed_kernel[0][1][[0,2]] = 1 #xz
            threed_kernel[2][1][[0, 2]] = 1 #xz
        elif direction=='xy':
            threed_kernel[0][0][1] = 1 #xy
            threed_kernel[0][2][1] = 1 #xy

            threed_kernel[2][0][1] = 1 #xy
            threed_kernel[2][2][1] = 1 #xy
        elif direction == 'yz':
            threed_kernel[1][0][[0,2]] =1 # yz
            threed_kernel[1][2][[0,2]] =1 # yz
        elif direction == 'xyz':
            threed_kernel[0][1][[0,2]] = 1 #xz
            threed_kernel[2][1][[0, 2]] = 1 #xz

            threed_kernel[0][0][1] = 1 #xy
            threed_kernel[0][2][1] = 1 #xy

            threed_kernel[2][0][1] = 1 #xy
            threed_kernel[2][2][1] = 1 #xy

            threed_kernel[1][0][[0,2]] =1 # yz
            threed_kernel[1][2][[0,2]] =1 # yz
        else:
            raise exit('Direction should be xy, xz, yz or xyz')

    else:
        kernel_used = torch.from_numpy(np.zeros(kerenel_size).astype(np.float32))  # mean
        threed_kernel = torch.einsum('i,j,k->ijk', kernel_used, kernel_used, kernel_used)
        if direction=='x':

            threed_kernel[0][1][1] = 1.0 # left
            threed_kernel[2][1][1] = 1.0 # right
        elif direction=='y':
            threed_kernel[1][0][1] = 1.0 # left
            threed_kernel[1][2][1] = 1.0 # right
        elif direction == 'z':
            threed_kernel[1][1][0] = 1.0 # left
            threed_kernel[1][1][2] = 1.0  # right
        elif direction == 'xyz':
            threed_kernel[0][1][1] = 1.0 # left
            threed_kernel[2][1][1] = 1.0 # right

            threed_kernel[1][0][1] = 1.0 # left
            threed_kernel[1][2][1] = 1.0 # right

            threed_kernel[1][1][0] = 1.0 # left
            threed_kernel[1][1][2] = 1.0  # right
        else:
            raise exit('Direction should be x, y, y or xyz')
    inp_torch = torch.from_numpy(output.astype(np.float32)).unsqueeze(0).permute([4, 0, 1, 2, 3])
    s = F.conv3d(inp_torch,
                 threed_kernel.reshape(1, 1, *threed_kernel.shape), stride=1,
                 padding=len(kernel_used) // 2)
    s= s.permute([1, 2, 3, 4, 0]).squeeze().detach().cpu().numpy()
    if s.ndim==3:
        s = s.reshape(*s.shape,1)
    return s




def rescale_between_a_b(image, a, b):
    nifti_type = False
    if hasattr(image, 'get_fdata'):
        nifti_type = True
        data_im = image.get_fdata().copy()
    else:
        data_im = image.copy()
    dif = b-a
    mindata= data_im.min()
    maxdata = data_im.max()
    data_im = a + (((data_im - mindata) * dif) / (maxdata - mindata))
    if nifti_type:
        return nib.Nifti1Image(data_im, image.affine, image.header)
    else:
        return data_im