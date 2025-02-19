import os
import sys
from dataclasses import dataclass
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom, gaussian_filter

# SynthSeg-Logger
from SynthSeg.logging_utils import get_logger
logger = get_logger("Resample")


@dataclass
class HPCScaleParams:
    """
    Dataclass mit Standardwerten für HPC-Parameter.
    """
    final_dim_z: int = 700
    final_dim_y: int = 700
    final_dim_x: int = 700
    gauss_sigma: float = 10.0


def gaussian_filter_mask(volume_3d: np.ndarray, label: float, sigma: float) -> np.ndarray:
    """
    Erzeugt eine Binärmaske (vol==label) und wendet Gaussian Filter an.
    Gibt ein float32-Array gleicher Größe zurück.
    """
    mask = (volume_3d == label).astype(np.float32)
    filtered = gaussian_filter(mask, sigma=sigma)
    return filtered.astype(np.float32)


def filter_and_argmax(volume_3d: np.ndarray,
                      final_dim=(700,700,700),
                      sigma: float=10.0) -> np.ndarray:
    """
    1) NN-Upscale 'volume_3d' auf final_dim=(z,y,x).
    2) Gauss-Filter pro Label => Argmax => int32-Volume.
    """
    in_z, in_y, in_x = volume_3d.shape
    out_z, out_y, out_x = final_dim

    scale_z = out_z / in_z
    scale_y = out_y / in_y
    scale_x = out_x / in_x

    logger.info(f"Input volume shape  = {volume_3d.shape}")
    logger.info(f"Upscaling-Factor    = (z={scale_z:.3f}, y={scale_y:.3f}, x={scale_x:.3f})")
    logger.info(f"Output volume shape = {final_dim}")

    # NN-Upsample
    upscaled_3d = zoom(volume_3d, zoom=(scale_z, scale_y, scale_x), order=0)
    logger.info(f"NN-Up shape => {upscaled_3d.shape}")

    unique_labels = np.unique(upscaled_3d)
    logger.info(f"Labels im upgesampelten Vol: {unique_labels}")

    final_vol = np.zeros(upscaled_3d.shape, dtype=np.float32)
    max_gauss = np.zeros(upscaled_3d.shape, dtype=np.float32)

    for lab in tqdm(unique_labels, desc="Gauss HPC"):
        gauss_val = gaussian_filter_mask(upscaled_3d, label=lab, sigma=sigma)
        update_mask = gauss_val > max_gauss
        final_vol[update_mask] = lab
        max_gauss[update_mask] = gauss_val[update_mask]

    return final_vol.astype(np.int32)


def main():
    """
    Entry-Point für HPC: nimmt zwei Argumente <input_file> <output_file> aus sys.argv,
    resamplet HPC => (700,700,700).
    Aufrufbar via 'synthSeg-scale-hpc <input> <output>'.
    """
    if len(sys.argv) < 3:
        logger.error(f"Aufruf: {sys.argv[0]} <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.isfile(input_file):
        logger.error(f"Input nicht gefunden: {input_file}")
        sys.exit(1)

    logger.info(f"Input file  => {input_file}")
    logger.info(f"Output file => {output_file}")

    # Standard-HPC-Parameter (könntest du auch optional aus Flags lesen)
    params = HPCScaleParams()

    # 1) Nifti laden
    nifti_in = nib.load(input_file)
    data_in = nifti_in.get_fdata().astype(np.int32)

    # 2) HPC => final_dim
    final_dim = (params.final_dim_z, params.final_dim_y, params.final_dim_x)
    final_data = filter_and_argmax(
        volume_3d=data_in,
        final_dim=final_dim,
        sigma=params.gauss_sigma
    )

    # 3) Header/Affine anpassen
    final_header = nifti_in.header.copy()

    old_zooms = final_header.get_zooms()[:3]
    in_z, in_y, in_x = data_in.shape

    scale_z = params.final_dim_z / in_z
    scale_y = params.final_dim_y / in_y
    scale_x = params.final_dim_x / in_x

    # => new_zooms = old / scale
    new_zooms = (
        old_zooms[0] / scale_z,
        old_zooms[1] / scale_y,
        old_zooms[2] / scale_x
    )
    final_header.set_zooms(new_zooms)

    final_affine = nifti_in.affine.copy()
    final_affine[0,0] /= scale_x
    final_affine[1,1] /= scale_y
    final_affine[2,2] /= scale_z

    final_header.set_qform(final_affine, code=1)
    final_header.set_sform(final_affine, code=1)

    logger.info(f"Input-Auflösung  = {old_zooms}")
    logger.info(f"Output-Auflösung = {new_zooms}")

    # 4) Speichern
    nifti_out = nib.Nifti1Image(final_data, final_affine, final_header)
    nib.save(nifti_out, output_file)

    logger.info("Fertig.")


if __name__ == "__main__":
    main()
