import nibabel as nib
import torch

mat = torch.from_numpy(nib.load('resources/gordon333MNI.nii.gz').get_fdata())
w,h,d = mat.shape
cent = []
output = ''
for i in mat.unique():
    if i == 0: continue
    x, y, z = torch.where(mat==i)
    cx = (x-w/2).float().mean()
    cy = (y-h/2-16).float().mean()
    cz = (z-d/2+20).float().mean()
    output += f'{cx}\t{cy}\t{cz}\t1\t1\tROI{i}\n'
with open('resources/gordon333.node', 'w') as f:
    f.write(output)
