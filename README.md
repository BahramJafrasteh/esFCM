



# Enhanced Spatial Fuzzy C-Means Algorithm for Brain Tissue Segmentation in T1 Images

This repository contains the implementation code and dataset for the paper titled "Enhanced Spatial Fuzzy C-Means algorithm for brain tissue segmentation in T1 images". In this paper, an enhanced spatial Fuzzy C-Means (FCM) algorithm is proposed for accurate segmentation of brain tissues in T1-weighted magnetic resonance imaging (MRI) scans.

## Dataset
The dataset used in this research is comprised of T1-weighted MRI images. It includes:
- IXI dataset: Available from [Brain Development Index](http://brain-development.org/ixi-dataset)
- IBSR dataset: Available from [NITRC](https://www.nitrc.org/projects/ibsr)
- Brainweb dataset: Available from [McGill University](https://brainweb.bic.mni.mcgill.ca/brainweb/)
- HUPM dataset: Available from [University of Cadiz](https://rodin.uca.es/handle/10498/31306)

## Contents
- `brain_tissue_segmentation/`: Contains the implementation code for the enhanced spatial FCM algorithm, FCM and GMM.
- `data/`: The dataset that can be downloaded from the previously mentioned links.
- `example/`: Contains an example and a script to add noise and bias filed to repeat the experiemtns mentioned in the manuscript.

## Usage
1. Clone this repository.
   ```bash
   git clone https://github.com/bahramjafrasteh/esFCM.git
2. Navigate to the repository directory.

3. Install esFCM library
   ```bash
    python setup.py install
4. import the library
   ```bash
   import esFCM
   import nibabel
   from esFCM import Segment
   img = nib.load(file_im)
   segmented_image = Segment(img,mask=img.get_fdata()>0, num_tissues=3, max_iter=50, max_fail=4)
   segmented_image.to_filename('out.nii.gz')
   
## Citation
If you find this work useful in your research, please consider citing:
```bibtex
@article{jafrasteh2024,
  title={Enhanced spatial Fuzzy C-Means algorithm for brain tissue segmentation in T1 images},
  author={Jafrasteh, Bahram and Lubián-Gutiérrez, Manuel and Lubián-López, Simón Pedro and Benavente-Fernández, Isabel},
  journal={Neuroinformatics},
  year={2024},
  volume={-},
  number={-},
  pages={-},
  doi={Under review}
}
```
## License
This project is licensed under the MIT License - see the [LICENSE](License.txt) file for details.