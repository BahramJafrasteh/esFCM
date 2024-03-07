import sys
sys.path.append('../')
from esFCM import Segment
import nibabel as nib

if __name__ == '__main__':
    file_im = '../data/input.nii.gz'
    img = nib.load(file_im)
    fina_seg = Segment(img,mask=img.get_fdata()>0, num_tissues=3, max_iter=50, max_fail=4)
    fina_seg.to_filename(sys.argv[2])
