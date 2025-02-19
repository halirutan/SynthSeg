import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import scipy.ndimage as ndi
from skimage.morphology import closing, cube
from typing import Union, List
from SynthSeg.logging_utils import get_logger

logger = get_logger("Resample")

def resample_label_image(
    nifti: nib.Nifti1Image,
    resolution_out: Union[float, List[float]],
    out_prefix: str = "resampled_label"
) -> None:
    """
    Resamplet ein Label-Image mit vier unterschiedlichen Methoden (NN, linear, Majority-Vote,
    NN+MorphologicalClosing), speichert die Ergebnisse separat und gibt eine einfache
    Vergleichsmetrik (Dice) zum Original aus.

    *Hardcoded* Version: Ruft man die Funktion normal auf, gibt sie vier NIfTI-Ausgaben aus:
    - <out_prefix>_nn.nii.gz
    - <out_prefix>_lin.nii.gz
    - <out_prefix>_majority.nii.gz
    - <out_prefix>_nn_closed.nii.gz
    """
    if isinstance(resolution_out, float):
        resolution_out = (resolution_out, resolution_out, resolution_out)
    elif isinstance(resolution_out, list) or isinstance(resolution_out, np.ndarray):
        resolution_out = tuple(resolution_out)
    else:
        raise ValueError("resolution_out muss float oder Liste/Tuple aus floats sein")

    original_data = nifti.get_fdata().astype(np.int32)
    original_affine = nifti.affine.copy()
    original_header = nifti.header.copy()
    old_spacing = nifti.header.get_zooms()[:3]  # (dx, dy, dz)

    def _dice_coefficient(arr1: np.ndarray, arr2: np.ndarray) -> float:
        mask1 = (arr1 != 0)
        mask2 = (arr2 != 0)
        inter = np.logical_and(mask1, mask2).sum()
        denom = mask1.sum() + mask2.sum()
        if denom == 0:
            return 1.0
        return 2.0 * inter / denom

    def _sitk_resample(image_sitk: sitk.Image,
                       target_spacing: tuple[float, float, float],
                       interpolator) -> sitk.Image:
        old_sp = image_sitk.GetSpacing()
        old_sz = image_sitk.GetSize()
        new_size = [int(round(osz * (osp / nsp))) for osz, osp, nsp in zip(old_sz, old_sp, target_spacing)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image_sitk.GetDirection())
        resampler.SetOutputOrigin(image_sitk.GetOrigin())
        resampler.SetInterpolator(interpolator)
        return resampler.Execute(image_sitk)

    def _nifti_to_sitk(nifti_in: nib.Nifti1Image) -> sitk.Image:
        dat = nifti_in.get_fdata().astype(np.float32)
        img_sitk = sitk.GetImageFromArray(dat)

        affine = nifti_in.affine
        zooms = nifti_in.header.get_zooms()[:3]

        # Hier in reine float-Liste oder Tupel umwandeln:
        zooms = [float(z) for z in zooms]
        img_sitk.SetSpacing(zooms)

        img_sitk.SetOrigin((affine[0, 3], affine[1, 3], affine[2, 3]))

        direction = [
            float(affine[0, 0]), float(affine[0, 1]), float(affine[0, 2]),
            float(affine[1, 0]), float(affine[1, 1]), float(affine[1, 2]),
            float(affine[2, 0]), float(affine[2, 1]), float(affine[2, 2])
        ]
        img_sitk.SetDirection(direction)

        return img_sitk

    def _sitk_to_nifti(image_sitk: sitk.Image, header_template: nib.Nifti1Header) -> nib.Nifti1Image:
        arr = sitk.GetArrayFromImage(image_sitk)
        new_header = header_template.copy()
        sp = image_sitk.GetSpacing()
        orx = image_sitk.GetOrigin()
        dir_ = image_sitk.GetDirection()
        direction_mat = np.array(dir_).reshape(3, 3)
        affine_out = np.eye(4, dtype=np.float64)
        affine_out[:3, :3] = direction_mat @ np.diag(sp)
        affine_out[:3, 3] = np.array(orx)
        new_header.set_zooms(sp)
        return nib.Nifti1Image(arr, affine_out, new_header)

    def _majority_vote_resample(orig_data: np.ndarray,
                                old_spc: tuple[float, float, float],
                                new_spc: tuple[float, float, float],
                                aff: np.ndarray) -> nib.Nifti1Image:
        hdr = original_header.copy()
        old_sz_z, old_sz_y, old_sz_x = orig_data.shape
        old_sz = [old_sz_z, old_sz_y, old_sz_x]
        new_size = [
            int(round(osz * (osp / nsp))) for osz, osp, nsp in zip(old_sz, old_spc, new_spc)
        ]
        oversample_factor = 2
        intermediate_size_xyz = [dim * oversample_factor for dim in new_size]
        zoom_factors = [
            inter_sz / orig_sz for inter_sz, orig_sz in zip(intermediate_size_xyz, orig_data.shape)
        ]
        temp_data = ndi.zoom(orig_data, zoom=zoom_factors, order=0)
        out_data = np.zeros((new_size[0], new_size[1], new_size[2]), dtype=np.int32)

        for z_new in range(new_size[0]):
            z0 = z_new * oversample_factor
            for y_new in range(new_size[1]):
                y0 = y_new * oversample_factor
                for x_new in range(new_size[2]):
                    x0 = x_new * oversample_factor
                    block = temp_data[z0:z0+oversample_factor,
                                      y0:y0+oversample_factor,
                                      x0:x0+oversample_factor]
                    vals, counts = np.unique(block, return_counts=True)
                    majority_lbl = vals[np.argmax(counts)]
                    out_data[z_new, y_new, x_new] = majority_lbl
        new_affine = aff.copy()
        new_affine[0, 0] = new_spc[0] * np.sign(aff[0, 0])
        new_affine[1, 1] = new_spc[1] * np.sign(aff[1, 1])
        new_affine[2, 2] = new_spc[2] * np.sign(aff[2, 2])
        hdr.set_zooms(new_spc)
        return nib.Nifti1Image(out_data, new_affine, hdr)

    def _morphological_closing_multilabel(arr_in: np.ndarray, selem_size=2) -> np.ndarray:
        arr_out = np.zeros_like(arr_in)
        labels_unique = np.unique(arr_in)
        selem = cube(selem_size)
        for lbl in labels_unique:
            if lbl == 0:
                continue
            mask = (arr_in == lbl).astype(np.uint8)
            closed = closing(mask, selem)
            arr_out[closed > 0] = lbl
        return arr_out

    # ---------------------------
    # 1) NEAREST NEIGHBOR
    # ---------------------------
    sitk_input = _nifti_to_sitk(nifti)
    sitk_nn = _sitk_resample(sitk_input, resolution_out, interpolator=sitk.sitkNearestNeighbor)
    nii_nn = _sitk_to_nifti(sitk_nn, original_header)
    out_nn_path = f"{out_prefix}_nn.nii.gz"
    nib.save(nii_nn, out_nn_path)
    dice_nn = _dice_coefficient(original_data, nii_nn.get_fdata().astype(np.int32))

    # ---------------------------
    # 2) LINEAR (gerundet)
    # ---------------------------
    sitk_lin = _sitk_resample(sitk_input, resolution_out, interpolator=sitk.sitkLinear)
    lin_data_float = sitk.GetArrayFromImage(sitk_lin)
    lin_data_rounded = np.rint(lin_data_float).astype(np.int32)
    tmp_lin_sitk = sitk.GetImageFromArray(lin_data_rounded)
    tmp_lin_sitk.CopyInformation(sitk_lin)
    nii_lin = _sitk_to_nifti(tmp_lin_sitk, original_header)
    out_lin_path = f"{out_prefix}_lin.nii.gz"
    nib.save(nii_lin, out_lin_path)
    dice_lin = _dice_coefficient(original_data, lin_data_rounded)

    # ---------------------------
    # 3) MAJORITY-VOTE
    # ---------------------------
    nii_maj = _majority_vote_resample(
        original_data, old_spacing, resolution_out, original_affine
    )
    out_maj_path = f"{out_prefix}_majority.nii.gz"
    nib.save(nii_maj, out_maj_path)
    dice_maj = _dice_coefficient(original_data, nii_maj.get_fdata().astype(np.int32))

    # ---------------------------
    # 4) NN + MorphologicalClosing
    # ---------------------------
    nn_data = nii_nn.get_fdata().astype(np.int32)
    nn_closed_data = _morphological_closing_multilabel(nn_data, selem_size=2)
    nii_nn_closed = nib.Nifti1Image(nn_closed_data, nii_nn.affine, nii_nn.header)
    out_nn_closed_path = f"{out_prefix}_nn_closed.nii.gz"
    nib.save(nii_nn_closed, out_nn_closed_path)
    dice_nn_closed = _dice_coefficient(original_data, nn_closed_data)

    print("Ergebnisse (Dice-Koeffizient, nur Nicht-Null):")
    print(f"Nearest Neighbor        : {dice_nn:.4f}   => {out_nn_path}")
    print(f"Linear (gerundet)       : {dice_lin:.4f}  => {out_lin_path}")
    print(f"Majority Vote           : {dice_maj:.4f}  => {out_maj_path}")
    print(f"NN + MorphologicalClose : {dice_nn_closed:.4f} => {out_nn_closed_path}")


def do_resample(nifti: nib.Nifti1Image, resolution_out: np.ndarray) -> np.ndarray:
    """
    Bleibt hier erhalten, falls du `resample_label_image_cropped` nutzen willst.
    Sonst kann man es ggf. entfernen.
    """
    header = nifti.header
    dim = header.get_data_shape()
    if len(dim) != 3:
        raise RuntimeError("Image data does not have 3 dimensions")
    resolution_in = header["pixdim"][1:4]
    step_size = resolution_out / resolution_in
    zs = np.arange(0, dim[0], step_size[0]).astype(np.intp)
    ys = np.arange(0, dim[1], step_size[1]).astype(np.intp)
    xs = np.arange(0, dim[2], step_size[2]).astype(np.intp)
    data = nifti.get_fdata()
    data_resampled = data[np.ix_(zs, ys, xs)]
    return data_resampled


def main():
    """Hard-Coded Aufruf: Keine CLI-Parameter mehr nötig."""
    # -- HARDCODED PFAD ZUM INPUT
    image_file = "/Users/julietteburkhardt/SynthSeg/data/training_label_maps/training_seg_02.nii.gz"
    # -- HARDCODED OUTPUT-DIR (Speichert resampled_images unten)
    output_dir = "/Users/julietteburkhardt/SynthSeg/data/training_label_maps"
    # -- HARDCODED ZIEL-AUFLÖSUNG
    resolution = 1.0

    # Hier Laden wir das Input
    if not os.path.isfile(image_file):
        logger.error(f"Eingabe existiert nicht: '{image_file}'")
        return
    nifti = nib.load(image_file)

    # Rufen die Resample-Funktion auf
    out_prefix = os.path.join(output_dir, "training_seg_02_resampled")
    resample_label_image(nifti, resolution, out_prefix=out_prefix)

    logger.info("Fertig mit Hardcoded-Resampling.")


if __name__ == '__main__':
    main()
