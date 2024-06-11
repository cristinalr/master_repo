import argparse
import os

import numpy as np
import SimpleITK as sitk 

def load_coronal_mip(img_path:str, target_size=(256,256), target_resolution=None, rotation_angle=0):
    """
    Loads a coronal MIP, samples to a target size and applies rotation about z axis
    ----------
    Parameters:
    img_path : path to NIfTI image
    target_size : size of target image pixel grid
    target_resolution : resolution (pixel size) of target image in mm
    rotation_angle : angle (DEGREES) to apply about z axis before MIP calculation

    returns:
    MIP : MIP image with desired rotation, size and resolution
    """
    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    img = sitk.DICOMOrient(img, 'LPI')

    # Rotate image before MIP computation
    sz = img.GetSize()
    centre_idx = [sz[i]/2 for i in range(0,3)]
    centre_xyz = img.TransformContinuousIndexToPhysicalPoint(centre_idx)
    angle_radians = np.pi * rotation_angle/180
    rotation = sitk.Euler3DTransform(centre_xyz,0,0,angle_radians)
    pad_size = [int(np.max(np.array(sz))/2)]*3
    rot_resamp_grid = sitk.ConstantPad(img, pad_size,pad_size,0)
    img_rotate = sitk.Resample(img, rot_resamp_grid, rotation, sitk.sitkLinear)

    # Compute MIP
    mip = sitk.MaximumProjection(img_rotate,1)

    # Resample to target size and dimension
    mip_resamp_grid = sitk.Image(target_size[0],1,target_size[1],mip.GetPixelID())
    mip_resamp_grid.SetDirection(img.GetDirection())
    mip_resamp_grid.SetOrigin(img.GetOrigin())
    if target_resolution is not None:
        mip_resamp_grid.SetSpacing((target_resolution[0],1,target_resolution[1]))
    else:
        pixel_dimensions = img.GetSpacing()
        target_size_3d = [target_size[0], 1, target_size[1]]
        target_resolution = [sz[i]*pixel_dimensions[i]/target_size_3d[i] for i in range(0,3)]
        target_resolution[1] = pixel_dimensions[1]
        mip_resamp_grid.SetSpacing((target_resolution[0],1,target_resolution[1]))
    mip = sitk.Resample(mip, mip_resamp_grid, sitk.Transform(), sitk.sitkLinear) 

    return mip



if __name__=='__main__':

    # Test the MIP computation
    parser = argparse.ArgumentParser('Create coronal MIP with arbitraty rotation about z')
    parser.add_argument('img_path', help='Path to image NIfTI file')
    parser.add_argument('--target_size',help='Target size of MIP in both dimensions', type=int,default=256)
    parser.add_argument('--target_resolution',help='Target resolution (both dimensions) in mm', type=float,default=1.5)
    parser.add_argument('-r','--rotation_angle', help='Rotation angle in degrees', default=0.0, type=float)
    parser.add_argument('-o','--out_dir', help='Output directiory',default=None)

    args = parser.parse_args()

    mip = load_coronal_mip(args.img_path,
                           target_size=(args.target_size,args.target_size),
                           target_resolution=(args.target_resolution,args.target_resolution),
                           rotation_angle=args.rotation_angle)
    
    out_dir = args.out_dir
    if out_dir is not None:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        
        sitk.WriteImage(mip, os.path.join(out_dir,f'rotated_mip_theta_{args.rotation_angle}.nii.gz'))






