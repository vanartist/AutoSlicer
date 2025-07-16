import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import torch
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
import matplotlib.pyplot as plt
import vtk
import vtk.util.numpy_support as numpy_support
from scipy.ndimage import label
import pydicom
import glob
import json

# Density map for each label (modifiable according to requirements)
DENSITY_MAP = {
    1: 1.05, 2: 1.06, 3: 1.06, 4: 1.02, 5: 1.05, 6: 1.04, 7: 1.04, 8: 1.04, 9: 1.04,
    10: 0.3, 11: 0.3, 12: 0.3, 13: 0.3, 14: 0.3, 15: 1.04, 16: 0.8, 17: 1.04, 18: 1.04,
    19: 1.04, 20: 1.04, 21: 1.05, 22: 1.04, 23: 1.01, 24: 1.01, 25: 1.9, 26: 1.9,
    27: 1.9, 28: 1.9, 29: 1.9, 30: 1.9, 31: 1.9, 32: 1.9, 33: 1.9, 34: 1.9, 35: 1.9,
    36: 1.9, 37: 1.9, 38: 1.9, 39: 1.9, 40: 1.9, 41: 1.9, 42: 1.9, 43: 1.9, 44: 1.9,
    45: 1.9, 46: 1.9, 47: 1.9, 48: 1.9, 49: 1.9, 50: 1.9, 51: 1.06, 52: 1.05, 53: 1.05,
    54: 1.05, 55: 1.05, 56: 1.05, 57: 1.05, 58: 1.05, 59: 1.05, 60: 1.05, 61: 1.06,
    62: 1.05, 63: 1.05, 64: 1.05, 65: 1.05, 66: 1.05, 67: 1.05, 68: 1.05, 69: 1.85,
    70: 1.85, 71: 1.85, 72: 1.85, 73: 1.85, 74: 1.85, 75: 1.85, 76: 1.85, 77: 1.85,
    78: 1.85, 79: 1.04, 80: 1.06, 81: 1.06, 82: 1.06, 83: 1.06, 84: 1.06, 85: 1.06,
    86: 1.06, 87: 1.06, 88: 1.06, 89: 1.06, 90: 1.04, 91: 1.9, 92: 1.85, 93: 1.85,
    94: 1.85, 95: 1.85, 96: 1.85, 97: 1.85, 98: 1.85, 99: 1.1, 100: 1.1, 101: 1.85,
    102: 1.85, 103: 1.85, 104: 1.85, 105: 1.85, 106: 1.85, 107: 1.85, 108: 1.85,
    109: 1.85, 110: 1.85, 111: 1.85, 112: 1.85, 113: 1.85, 114: 1.85, 115: 1.85,
    116: 1.85, 117: 1.8, 118: 0.95
}


class AutoSlicer:
    """
    AutoSlicer performs:
      1. DICOM to NIfTI conversion
      2. TotalSegmentator-based segmentation
      3. Adding a skin label in a given intensity range
      4. Computing mass, volume, inertia, center of mass
      5. Generating a VTK model file
      6. Visualizing the VTK (optional)
    """

    def __init__(self, workspace: str):
        """
        Initializes AutoSlicer with file paths and threshold defaults.

        Args:
            workspace (str): Name of the workspace directory.
        """
        # self.current_dir = os.path.dirname(os.path.abspath(__file__))
        # self.workspace = os.path.join(self.current_dir, workspace)
        self.workspace = workspace
        self._ensure_directory(self.workspace)

        # Default intensity thresholds
        self.lower_ = -96.25
        self.upper_ = 153.46

        # File path configuration
        self.source_volume_path = os.path.join(self.workspace, "CT_Source_Volume.nii.gz")
        self.total_seg_result = os.path.join(self.workspace, "CT_TotalSegmentation.nii.gz")
        self.other_soft_tissue = os.path.join(self.workspace, "CT_SoftTissueLabel0.nii.gz")
        self.final_seg_result = os.path.join(self.workspace, "CT_head_mask.nii.gz")
        self.cropping = os.path.join(self.workspace, "CT_Cropping_seg.nii.gz")
        self.vtk_path_head = os.path.join(self.workspace, "head_visualization.vtk")
        self.vtk_path_neck = os.path.join(self.workspace, "neck_visualization.vtk")
        self.main_axes = os.path.join(self.workspace, "PrincipalAxes_global_reference.mrk.json")
        # The screenshot will be saved here
        self.output_image = os.path.join(self.workspace, "vtk_visualization.png")

        self.Inertia_parameters_file = os.path.join(self.workspace, "head_inertia_parameters_global_reference.json")
        self.Inertia_parameters_file_neck = os.path.join(self.workspace, "neck_inertia_parameters_global_reference.json")

        self.Inertia_parameters_frankfort = os.path.join(self.workspace, "head_inertia_parameters_frankfort_reference.json")
        self.Inertia_parameters_frankfort_neck = os.path.join(self.workspace,
                                                         "neck_inertia_parameters_frankfort_reference.json")
        self.frank_coor = os.path.join(self.workspace, "head_frankfort_reference.mrk.json")
        self.voxel_size_value = [0, 0, 0]
        self.input_folder_path = None
        self.mrk_json = os.path.join(self.workspace, "F.mrk.json")
        self.neck_segmentation = os.path.join(self.workspace, "CT_neck_mask.nii.gz")
        self.neck_reference = os.path.join(self.workspace, "neck_frankfort_reference.mrk.json")
        # unit vector
        self.x_ax = None
        self.y_ax = None
        self.z_ax = None

    @staticmethod
    def _ensure_directory(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def set_density(self, label: int, value: float) -> None:
        if label in DENSITY_MAP:
            DENSITY_MAP[label] = value
        else:
            print(f"Label {label} not found in the density map. Skipping.")

    def set_threshold(self, lower_val: float, upper_val: float) -> None:
        self.lower_ = lower_val
        self.upper_ = upper_val

    # ----------------- NIfTI Creation ----------------- #
    # @staticmethod
    def _get_voxel_size(self, input_folder: str):
        try:
            reader = sitk.ImageSeriesReader()
            dicom_series = reader.GetGDCMSeriesIDs(input_folder)
            if not dicom_series:
                raise ValueError("No DICOM series found.")

            series_file_names = reader.GetGDCMSeriesFileNames(input_folder, dicom_series[0])
            reader.SetFileNames(series_file_names)
            image = reader.Execute()

            voxel_size = image.GetSpacing()
            self.voxel_size_value = voxel_size
            print(f"Voxel size (x, y, z): {voxel_size}")
            unit = "mm"
            print(f"Assumed unit: {unit}")
            return voxel_size, unit

        except Exception as e:
            print(f"Error reading voxel size: {e}")
            return None, None

    def _dicom_to_nifti(self, input_folder: str, output_path: str) -> None:
        try:
            reader = sitk.ImageSeriesReader()
            self._get_voxel_size(input_folder)

            dicom_series = reader.GetGDCMSeriesIDs(input_folder)
            if not dicom_series:
                raise ValueError("No DICOM series found.")

            print(f"Found {len(dicom_series)} DICOM series.")
            for series_id in dicom_series:
                series_file_names = reader.GetGDCMSeriesFileNames(input_folder, series_id)
                reader.SetFileNames(series_file_names)
                image = reader.Execute()
                sitk.WriteImage(image, output_path)
                print(f"Converted series {series_id} to {output_path}")

            print("All DICOM series converted successfully.")

        except Exception as e:
            print(f"Error during DICOM->NIfTI: {e}")

    def create_nifti(self, input_folder: str) -> None:
        self._dicom_to_nifti(input_folder, self.source_volume_path)

    # ----------------- Segmentation ----------------- #
    def _segment_image(self, input_image_path: str, output_path: str) -> None:
        device = "gpu" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        try:
            input_img = nib.load(input_image_path)
            print("Running TotalSegmentator...")
            output_img = totalsegmentator(
                input=input_img,
                task="total",
                ml=True,
                device=device
            )
            nib.save(output_img, output_path)
            print(f"Segmentation done. Saved to: {output_path}")
        except Exception as e:
            print(f"Error in segmentation: {e}")
            raise

    def _label_segmentation(self, seg_result_path: str, labeled_output_path: str) -> None:
        try:
            print("Loading segmentation with labels...")
            seg_nifti_img, label_map_dict = load_multilabel_nifti(seg_result_path)

            label_img = seg_nifti_img.get_fdata().astype(int)
            label_nifti = nib.Nifti1Image(label_img, seg_nifti_img.affine, seg_nifti_img.header)
            label_nifti.header["descrip"] = "Label Map for 3D Slicer"

            for label, description in label_map_dict.items():
                label_nifti.header.extensions.append(
                    nib.nifti1.Nifti1Extension(4, f"{label}: {description}".encode("utf-8"))
                )

            nib.save(label_nifti, labeled_output_path)
            print(f"Labeled segmentation saved to: {labeled_output_path}")
        except Exception as e:
            print(f"Error labeling segmentation: {e}")
            raise

    def create_segmentation(self) -> None:
        self._segment_image(self.source_volume_path, self.total_seg_result)
        self._label_segmentation(self.total_seg_result, self.other_soft_tissue)

    # ----------------- Skin Labeling ----------------- #
    @staticmethod
    def _load_nifti_file(file_path: str):
        nifti_img = nib.load(file_path)
        return nifti_img.get_fdata(), nifti_img.affine

    @staticmethod
    def _save_nifti_file(data: np.ndarray, affine: np.ndarray, file_path: str) -> None:
        new_nifti_img = nib.Nifti1Image(data, affine)
        nib.save(new_nifti_img, file_path)

    def _add_skin_label(self, src_volume: np.ndarray, seg_result: np.ndarray) -> np.ndarray:
        """
        Use self.lower_ and self.upper_ to define the intensity range for 'skin',
        then label it as 118 in the final segmentation array.
        """
        skin_candidate = (src_volume >= self.lower_) & (src_volume <= self.upper_)
        unlabeled_area = (seg_result == 0)
        skin_label_area = skin_candidate & unlabeled_area

        updated_seg = seg_result.copy()
        updated_seg[skin_label_area] = 118
        return updated_seg

    def create_soft_tissue_segmentation(self) -> None:
        seg_result, seg_affine = self._load_nifti_file(self.other_soft_tissue)
        source_volume, _ = self._load_nifti_file(self.source_volume_path)

        updated_seg = self._add_skin_label(source_volume, seg_result)
        self._save_nifti_file(updated_seg, seg_affine, self.final_seg_result)
        # self._save_nifti_file(updated_seg, seg_affine, self.neck_segmentation)
        print(f"Updated segmentation saved: {self.final_seg_result}")

    # ----------------- Inertia + Center of Mass ----------------- #
    @staticmethod
    def _get_segmentation_labels(seg_file: str) -> list:
        nifti_img = nib.load(seg_file)
        seg_data = nifti_img.get_fdata()
        unique_labels = np.unique(seg_data)
        return [int(x) for x in unique_labels.tolist()]

    def _extract_patient_metadata(self, input_folder):

        try:
            dicom_files = glob.glob(os.path.join(input_folder, "*.dcm"))
            if not dicom_files:
                print("No DICOM files found in folder.")
                return {}

            ds = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)

            return {
                "PatientID": getattr(ds, "PatientID", ""),
                "PatientSex": getattr(ds, "PatientSex", ""),
                "PatientAge": getattr(ds, "PatientAge", ""),
                "StudyInstanceUID": getattr(ds, "StudyInstanceUID", ""),
                "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", ""),
                "SOPInstanceUID": getattr(ds, "SOPInstanceUID", ""),
            }
        except Exception as e:
            print(f"Error extracting patient metadata: {e}")
            return {}

    def _calculate_inertia_parameters(self, seg_file: str,out_put_inertial) -> dict:
        """
        Calculates volume, mass, inertia, and center of mass across all labeled structures.
        Outputs include real-world physical DICOM-space values and principal moments of inertia.
        """
        voxel_size = self.voxel_size_value
        labels = self._get_segmentation_labels(seg_file)
        nifti_img = nib.load(seg_file)
        seg_data = nifti_img.get_fdata()
        affine = nifti_img.affine

        voxel_vol_cm3 = np.prod(voxel_size) / 1000.0

        total_mass = 0.0
        total_volume = 0.0
        total_inertia_tensor = np.zeros((3, 3))

        acc_mass_times_centroid = np.zeros(3)
        acc_mass_times_voxel = np.zeros(3)
        acc_mass_times_dicom = np.zeros(3)
        acc_mass = 0.0

        for label in labels:
            if label == 0 or label not in DENSITY_MAP:
                continue

            mask = (seg_data == label)
            num_voxels = mask.sum()
            if num_voxels == 0:
                continue

            density = DENSITY_MAP[label]
            volume = num_voxels * voxel_vol_cm3
            mass = (volume * density) / 1000.0  # kg

            coords_ijk = np.array(np.where(mask)).T
            coords_dicom = nib.affines.apply_affine(affine, coords_ijk)
            centroid_dicom = coords_dicom.mean(axis=0)
            voxel_centroid = coords_ijk.mean(axis=0)
            centroid_mm = voxel_centroid * voxel_size

            acc_mass_times_centroid += mass * centroid_mm
            acc_mass_times_voxel += mass * voxel_centroid
            acc_mass_times_dicom += mass * centroid_dicom
            acc_mass += mass

            # Inertia tensor in DICOM coordinate system
            inertia_tensor = np.zeros((3, 3))
            for pt in coords_dicom:
                rel = pt - centroid_dicom
                x, y, z = rel
                inertia_tensor += np.array([
                    [y ** 2 + z ** 2, -x * y, -x * z],
                    [-x * y, x ** 2 + z ** 2, -y * z],
                    [-x * z, -y * z, x ** 2 + y ** 2]
                ]) * mass / num_voxels

            total_volume += volume
            total_mass += mass
            total_inertia_tensor += inertia_tensor

        if acc_mass > 0:
            global_com_image = acc_mass_times_centroid / acc_mass
            global_com_voxel = acc_mass_times_voxel / acc_mass
            global_com_dicom = acc_mass_times_dicom / acc_mass
        else:
            global_com_image = np.zeros(3)
            global_com_voxel = np.zeros(3)
            global_com_dicom = np.zeros(3)

        # Eigendecomposition of the total inertia tensor (principal MOI and directions)
        eigvals, eigvecs = np.linalg.eigh(total_inertia_tensor)
        principal_moi = eigvals.tolist()
        principal_axes = eigvecs.tolist()

        result = {
            "T1": {
                "name": "Volume",
                "value": total_volume,
                "unit": "cm³",
                "explanation": "Total segmented volume."
            },
            "T2": {
                "name": "Mass",
                "value": total_mass,
                "unit": "kg",
                "explanation": "Total mass estimated from volume and label-specific density."
            },
            "T3": {
                "name": "Total Inertia Tensor (DICOM space) Global Reference",
                "value": total_inertia_tensor.tolist(),
                "unit": "kg·mm²",
                "explanation": "Inertia tensor in DICOM patient coordinate system (RAS), relative to global center of mass."
            },
            "T4": {
                "name": "Total Inertia Tensor (DICOM space) Global Reference",
                "value": (total_inertia_tensor / 100.0).tolist(),
                "unit": "kg·cm²",
                "explanation": "Same as T3, converted from mm² to cm²."
            },
            "T5": {
                "name": "Center of Mass (Image Space)",
                "value": global_com_image.tolist(),
                "unit": "mm",
                "explanation": "Center of mass in image physical space. Not orientation-corrected."
            },
            "T6": {
                "name": "Center of Mass (Voxel Index)",
                "value": global_com_voxel.tolist(),
                "unit": "IJK",
                "explanation": "Center of mass in raw voxel index coordinates (I, J, K)."
            },
            "T7": {
                "name": "Center of Mass (DICOM Space)",
                "value": global_com_dicom.tolist(),
                "unit": "mm",
                "explanation": "True center of mass in DICOM patient RAS(XYZ) coordinate system."
            },
            "T8": {
                "name": "Principal Moments of Inertia",
                "value": principal_moi,
                "unit": "kg·mm²",
                "explanation": "Indicates how resistant the structure is to rotation around each of its three main directions."
            },
            "T9": {
                "name": "Principal Axes (columns: x,y,z)",
                "value": principal_axes,
                "unit": "unit vectors",
                "explanation": "Each column is a unit direction vector of a principal axis in DICOM RAS space."
            }
        }

        patient_info = self._extract_patient_metadata(self.input_folder_path)
        output_dict = {
            "Patient Metadata": {k: patient_info.get(k, "") for k in [
                "PatientID", "PatientSex", "PatientAge", "StudyInstanceUID",
                "SeriesInstanceUID", "SOPInstanceUID"
            ]},
            "Inertia Analysis Result": result
        }

        with open(out_put_inertial, "w", encoding="utf-8") as json_out:
            json.dump(output_dict, json_out, ensure_ascii=False, indent=2)

        markups_data = {
            "markups": [
                {
                    "type": "Fiducial",
                    "label": "Center_of_Mass",
                    "controlPoints": [{
                        "position": global_com_dicom.tolist(),
                        "label": "COM",
                        "description": "Center of mass"
                    }]
                }
            ],
            "version": "4.11",
            "coordinateSystem": "RAS"
        }

        axes = np.array(principal_axes).T  # shape: (3, 3)
        length = 400  # mm
        for i in range(3):
            direction = axes[i]
            start = (global_com_dicom - direction * length / 2).tolist()
            end = (global_com_dicom + direction * length / 2).tolist()

            markups_data["markups"].append({
                "type": "Line",
                "label": f"PrincipalAxis_{i + 1}",
                "controlPoints": [
                    {"position": start, "label": "P1", "description": "Start"},
                    {"position": end, "label": "P2", "description": "End"}
                ]
            })

        json_path = self.main_axes
        # with open(json_path, "w", encoding="utf-8") as f:
        #     json.dump(markups_data, f, indent=2)
        return result

    def correct_dicom_affine(self, nifti_input_path, nifti_output_path=None):
        """
        Corrects the affine matrix of a NIfTI file to standard DICOM RAS orientation
        (X: left→right, Y: posterior→anterior, Z: inferior→superior).

        Args:
            nifti_input_path (str): Path to the input NIfTI file (.nii.gz)
            nifti_output_path (str, optional): Path to save corrected NIfTI. If None, overwrites input.
        """
        import nibabel as nib
        import numpy as np

        # Load original NIfTI
        img = nib.load(nifti_input_path)
        data = img.get_fdata()
        affine = img.affine

        print(f"Original affine matrix:\n{affine}")

        # Check affine orientation
        codes = nib.aff2axcodes(affine)
        print(f"Original axis codes: {codes}")

        # Create identity affine
        new_affine = np.eye(4)

        # Copy voxel spacing
        voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        new_affine[0, 0] = voxel_sizes[0]
        new_affine[1, 1] = voxel_sizes[1]
        new_affine[2, 2] = voxel_sizes[2]

        # Set origin (0,0,0) for now (or could keep old origin if needed)
        new_affine[:3, 3] = [0, 0, 0]

        # Check if flipping is needed
        flip_x = codes[0] != 'R'
        flip_y = codes[1] != 'A'
        flip_z = codes[2] != 'S'

        # Flip data if necessary
        if flip_x:
            print("Flipping X axis (Left↔Right)")
            data = np.flip(data, axis=0)
        if flip_y:
            print("Flipping Y axis (Anterior↔Posterior)")
            data = np.flip(data, axis=1)
        if flip_z:
            print("Flipping Z axis (Superior↔Inferior)")
            data = np.flip(data, axis=2)

        # Save corrected image
        corrected_img = nib.Nifti1Image(data, new_affine, header=img.header)

        if nifti_output_path is None:
            nifti_output_path = nifti_input_path  # Overwrite original

        nib.save(corrected_img, nifti_output_path)
        print(f"Affine corrected NIfTI saved to: {nifti_output_path}")
    def get_VTK_file(self,mask_file,out_put_path):
        """
        Generates a VTK file from the final segmentation, with a 'Density' array
        stored per label as a point data array in the polygon mesh.
        """
        seg_file = mask_file
        output_file_name = out_put_path

        seg_reader = vtk.vtkNIFTIImageReader()
        seg_reader.SetFileName(seg_file)
        seg_reader.Update()
        seg_image = seg_reader.GetOutput()

        seg_array = numpy_support.vtk_to_numpy(seg_image.GetPointData().GetScalars())
        unique_labels = np.unique(seg_array)

        density_lut = {}
        for label in unique_labels:
            if label == 0:
                continue
            density_lut[label] = DENSITY_MAP.get(int(label), 1.0)

        append_filter = vtk.vtkAppendPolyData()

        for label in unique_labels:
            if label == 0:
                continue

            thresh = vtk.vtkImageThreshold()
            thresh.SetInputData(seg_image)
            thresh.ThresholdBetween(label, label)
            thresh.SetInValue(1)
            thresh.SetOutValue(0)
            thresh.Update()

            cast_filter = vtk.vtkImageCast()
            cast_filter.SetInputConnection(thresh.GetOutputPort())
            cast_filter.SetOutputScalarTypeToUnsignedChar()
            cast_filter.Update()

            contour = vtk.vtkMarchingCubes()
            contour.SetInputConnection(cast_filter.GetOutputPort())
            contour.SetValue(0, 0.5)
            contour.Update()

            smooth = vtk.vtkSmoothPolyDataFilter()
            smooth.SetInputConnection(contour.GetOutputPort())
            smooth.SetNumberOfIterations(30)
            smooth.SetRelaxationFactor(0.1)
            smooth.FeatureEdgeSmoothingOff()
            smooth.BoundarySmoothingOn()
            smooth.Update()

            fill_holes = vtk.vtkFillHolesFilter()
            fill_holes.SetInputConnection(smooth.GetOutputPort())
            fill_holes.SetHoleSize(1000.0)
            fill_holes.Update()

            polydata = fill_holes.GetOutput()
            density = density_lut[label]

            density_array = vtk.vtkFloatArray()
            density_array.SetName("Density")
            density_array.SetNumberOfComponents(1)
            density_array.SetNumberOfTuples(polydata.GetNumberOfPoints())
            density_array.FillComponent(0, density)
            polydata.GetPointData().AddArray(density_array)

            append_filter.AddInputData(polydata)

        append_filter.Update()

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(append_filter.GetOutputPort())
        cleaner.Update()

        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(output_file_name)
        writer.SetInputConnection(cleaner.GetOutputPort())
        writer.Write()

    def calculate_inertia(self):
        inertial_head =  self._calculate_inertia_parameters(self.final_seg_result,self.Inertia_parameters_file)
        inertial_neck = self._calculate_inertia_parameters(self.neck_segmentation,self.Inertia_parameters_file_neck)
        return inertial_head,inertial_neck


    def run_automation(self, input_folder: str):
        """
        One-click process:
          1. Convert DICOM -> NIfTI
          2. Perform segmentation
          3. Add skin label
          4. Compute inertia/center of mass
          5. Generate VTK file

        Returns a dict of computed volume, mass, inertia, and center of mass.
        """
        self.input_folder_path = input_folder

        self.create_nifti(input_folder)
        # self.correct_dicom_affine(self.source_volume_path)
        self.create_segmentation()
        self.create_soft_tissue_segmentation()

        # Further process of the data
        # remove medical devices
        self.keep_largest_component_only_target(118.0)
        # to make sure the C1 is cleaned
        self.keep_largest_component_only_target(50.0)
        self.keep_largest_component_only_target(90.0)
        self.keep_largest_component_only_target(91.0)
        self.keep_largest_component_only_target(44.0)
        # Cut head
        self.cut_below_plane_defined_by_points(offset=10)
        # remove the rest part of tissue
        self.crop_x_axis_by_target(self.final_seg_result,
                                   self.final_seg_result,
                                   target_label=91.0,
                                   offset=15)
        self.keep_largest_component_only_target(118.0)
        # remove other labels
        self.filter_segmentation_labels(self.final_seg_result,
                                        self.final_seg_result,
                                        [118.0, 91.0, 90.0])

        if os.path.exists(self.neck_segmentation):
            self.filter_segmentation_labels(
                self.neck_segmentation,
                self.neck_segmentation,
                [44.0, 45.0, 46.0, 47.0, 48.0,
                 49.0, 50.0, 57.0, 58.0, 79.0, 86.0, 87.0, 118.0]
            )
        else:
            print(f"File {self.neck_segmentation} does not exist. Skipping filtering.")

        self.get_VTK_file(self.final_seg_result,self.vtk_path_head)
        self.get_VTK_file(self.neck_segmentation,self.vtk_path_neck)
        result_head,result_neck = self.calculate_inertia()
        # self.save_segmentation_slices_as_images(output_prefix="CT_seg")

        # Cleanup temporary files if desired
        try:
            if os.path.exists(self.other_soft_tissue):
                os.remove(self.other_soft_tissue)
                print(f"Deleted temp file: {self.other_soft_tissue}")
            if os.path.exists(self.total_seg_result):
                os.remove(self.total_seg_result)
                print(f"Deleted temp file: {self.total_seg_result}")
        except Exception as e:
            print(f"Error deleting temp files: {e}")

        return result_head,result_neck

    @staticmethod
    def center_crop_or_pad(image, center, output_size=(256, 256)):
        """
        Crop or pad a 2D image around a given center to a fixed output size.

        Args:
            image (np.ndarray): 2D input image (e.g., segmentation slice).
            center (tuple): (y, x) coordinates of the desired center.
            output_size (tuple): Desired output image size (height, width).

        Returns:
            np.ndarray: Centered, cropped/padded image.
        """
        H, W = image.shape
        out_h, out_w = output_size
        cy, cx = center

        # Define the bounding box
        y1 = int(cy - out_h // 2)
        y2 = y1 + out_h
        x1 = int(cx - out_w // 2)
        x2 = x1 + out_w

        # Create a blank output canvas
        output = np.zeros((out_h, out_w), dtype=image.dtype)

        # Determine valid source and destination regions
        src_y1 = max(y1, 0)
        src_y2 = min(y2, H)
        src_x1 = max(x1, 0)
        src_x2 = min(x2, W)

        dst_y1 = src_y1 - y1
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        dst_x1 = src_x1 - x1
        dst_x2 = dst_x1 + (src_x2 - src_x1)

        output[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

        return output

    def save_segmentation_slices_as_images(self, output_prefix="slice", output_size=(256, 256)):
        """
        Save centered slices of the final segmentation in 3 orthogonal planes:
        axial (top-down), sagittal (left-right), and coronal (front-back).

        The saved PNG images will be centered on the region of interest and padded to output_size.
        """

        # Load segmentation
        seg = nib.load(self.final_seg_result)
        data = seg.get_fdata()

        # Get coordinates of foreground voxels
        coords = np.argwhere(data > 0)
        if coords.shape[0] == 0:
            print("No foreground labels found in segmentation.")
            return

        # Compute center of mass
        center_x = int(np.mean(coords[:, 0]))
        center_y = int(np.mean(coords[:, 1]))
        center_z = int(np.mean(coords[:, 2]))

        # Define output file paths
        paths = {
            "sagittal_left_right": os.path.join(self.workspace, f"{output_prefix}_sagittal_LR.png"),
            "coronal_front_back": os.path.join(self.workspace, f"{output_prefix}_coronal_FB.png"),
            "axial_top_bottom": os.path.join(self.workspace, f"{output_prefix}_axial_TB.png"),
        }

        # Extract slices
        sag = np.rot90(data[center_x, :, :])
        cor = np.rot90(data[:, center_y, :])
        axi = np.rot90(data[:, :, center_z])

        # Calculate center for each slice (2D)
        def get_slice_center(slice_2d):
            nonzero = np.argwhere(slice_2d > 0)
            if len(nonzero) == 0:
                return (slice_2d.shape[0] // 2, slice_2d.shape[1] // 2)
            return tuple(np.mean(nonzero, axis=0).astype(int)[::-1])  # (y, x)

        sag_center = get_slice_center(sag)
        cor_center = get_slice_center(cor)
        axi_center = get_slice_center(axi)

        # Center-crop or pad
        sag_cropped = self.center_crop_or_pad(sag, sag_center, output_size)
        cor_cropped = self.center_crop_or_pad(cor, cor_center, output_size)
        axi_cropped = self.center_crop_or_pad(axi, axi_center, output_size)

        # Save images using colorful colormap
        plt.imsave(paths["sagittal_left_right"], sag_cropped, cmap='nipy_spectral')
        plt.imsave(paths["coronal_front_back"], cor_cropped, cmap='nipy_spectral')
        plt.imsave(paths["axial_top_bottom"], axi_cropped, cmap='nipy_spectral')

        print("Centered segmentation preview images saved:")
        for name, path in paths.items():
            print(f"  {name}: {path}")

    def filter_segmentation_labels(self, input_path, output_path, target_labels):
        """
        Keep only the specified label values in a segmentation file.

        :param input_path: Path to the input .nii.gz segmentation file
        :param output_path: Path to save the filtered .nii.gz file
        :param target_labels: A list of label values to keep, e.g., [1, 3, 5]
        """
        # Load the segmentation image
        img = nib.load(input_path)
        data = img.get_fdata()

        # Create an empty array with the same shape
        filtered_data = np.zeros_like(data)

        # Copy over only the target labels
        for label in target_labels:
            if np.any(data == label):
                filtered_data[data == label] = label
            else:
                print(f"Label {label} not found in data, skipping.")

        # Save the filtered image
        filtered_img = nib.Nifti1Image(filtered_data, affine=img.affine, header=img.header)
        nib.save(filtered_img, output_path)

        print(f"Filtered segmentation saved to: {output_path}")

    def keep_largest_component_only_target(self, target_label, neck_path=None):
        # Load the original segmentation file
        if neck_path == None:
            seg = nib.load(self.final_seg_result)
        else:
            seg = nib.load(self.neck_segmentation)

        data = seg.get_fdata()
        affine = seg.affine
        header = seg.header

        # Make a copy of the original data to preserve all other labels
        new_data = data.copy()

        # Create a binary mask for the target label
        if not np.any(data == target_label):
            print(f"Label {target_label} not found in the image. Skipping operation.")
            return

        mask = (data == target_label).astype(np.uint8)

        # Perform 3D connected component labeling
        labeled_array, num_features = label(mask)
        if num_features == 0:
            print(f"No connected components found for label {target_label}")
            return

        # Find the label of the largest connected component
        counts = np.bincount(labeled_array.flatten())
        counts[0] = 0  # Ignore background
        max_label = np.argmax(counts)

        # Create a mask that only includes the largest connected component
        cleaned_mask = (labeled_array == max_label)

        # Set all other voxels with the target label to background (0)
        new_data[(data == target_label) & (~cleaned_mask)] = 0

        # Save the cleaned segmentation to a new file
        new_seg = nib.Nifti1Image(new_data.astype(data.dtype), affine, header)
        if neck_path == None:
            nib.save(new_seg, self.final_seg_result)
            print(f"Kept only the largest connected component for label {target_label}, other labels preserved.")
            print(f"Cleaned segmentation saved to: {self.final_seg_result}")
        else:
            nib.save(new_seg, self.neck_segmentation)
            print(f"Kept only the largest connected component for label {target_label}, other labels preserved.")
            print(f"Cleaned segmentation saved to: {self.neck_segmentation}")

    def crop_x_and_y_by_target(self, input_path, output_path, x_min, x_max, y_min, y_max, target_label,
                               vertebra_surface=44):
        """
        Crop a NIfTI image along X and Y axes.
        X-axis is cropped using given x_min and x_max.
        Y-axis is determined based on target label location within the image.

        Args:
            input_path (str): Path to input .nii.gz file.
            output_path (str): Path to save cropped .nii.gz file.
            x_min (int): Minimum index along X-axis to crop.
            x_max (int): Maximum index along X-axis to crop.
            target_label (int): Label value to locate along Y-axis.
            offset_x (int): Extra voxels added to x crop (already applied usually).
            offset_y (int): Extra voxels to expand y crop region.
        """
        img = nib.load(input_path)
        data = img.get_fdata()
        affine = img.affine

        # First crop along X-axis
        cropped_data_x = data[x_min:x_max + 1, :, :]
        # Final crop along Y-axis
        cropped_data = cropped_data_x[:, y_min:y_max + 1, :]

        # Update affine
        new_affine = affine.copy()
        new_affine[:3, 3] += affine[:3, 0] * x_min  # x translation
        new_affine[:3, 3] += affine[:3, 1] * y_min  # y translation

        # Find target_label (e.g., 44) in the cropped image
        target_mask = (cropped_data == vertebra_surface)

        if np.any(target_mask):
            # Get voxel coordinates (in cropped image)
            coords = np.column_stack(np.where(target_mask))  # shape (N, 3)

            # Map voxel indices to world coordinates
            voxel_to_world = lambda ijk: affine[:3, :3] @ np.array((ijk[0] + x_min, ijk[1] + y_min, ijk[2])) + affine[
                                                                                                               :3, 3]
            world_coords = np.array([voxel_to_world(coord) for coord in coords])

            # Determine Z axis direction (upward or downward)
            z_direction = np.sign(affine[2, 2])

            # Find the ground point based on real-world Z
            if z_direction > 0:
                # Z increases upward, so ground = minimum z
                ground_idx = coords[np.argmin(world_coords[:, 2])]
            else:
                # Z increases downward, so ground = maximum z
                ground_idx = coords[np.argmax(world_coords[:, 2])]

            print(f"Lowest point of label {target_label} found at voxel index (after XY crop): {ground_idx}")

            # Cut the volume below the ground_idx along Z
            z_cut = ground_idx[2]

            for z in range(cropped_data.shape[2]):
                if (z < z_cut and z_direction > 0) or (z > z_cut and z_direction < 0):
                    cropped_data[:, :, z] = 0

            print(f"All voxels below plane Z={z_cut} have been cleared.")
        else:
            print(f"Label {target_label} not found in cropped image. Skipping Z-axis cutting.")

        cropped_img = nib.Nifti1Image(cropped_data, affine=new_affine, header=img.header)
        nib.save(cropped_img, output_path)

        print(f"Second image cropped (X: {x_min}-{x_max}, Y: {y_min}-{y_max}) and saved to: {output_path}")

    def crop_x_axis_by_target(self, input_path, output_path, target_label, offset=10, offset_y=25):
        """
        Crop a NIfTI image along the X-axis based on a target label's left-right boundaries,
        and update the affine to preserve the original world-space position.

        Args:
            input_path (str): Path to the input .nii.gz file.
            output_path (str): Path to save the cropped .nii.gz file.
            target_label (int): Label value to locate along X-axis.
            offset (int): Number of voxels to expand the crop on both sides.
        """
        # Load image
        img = nib.load(input_path)
        data = img.get_fdata()
        affine = img.affine

        # Create a mask for the target label
        target_mask = (data == target_label)
        coords = np.argwhere(target_mask)

        if coords.shape[0] == 0:
            print(f"Target label {target_label} not found in image.")
            return

        # Get X-axis min and max
        x_min = np.min(coords[:, 0])
        x_max = np.max(coords[:, 0])

        # Apply offset
        x_min = max(x_min - offset, 0)
        x_max = min(x_max + offset, data.shape[0] - 1)

        # Find Y-axis min and max
        y_min = np.min(coords[:, 1])
        y_max = np.max(coords[:, 1])

        # Apply offset
        y_min = max(y_min - offset_y, 0)
        y_max = min(y_max + offset_y, data.shape[1] - 1)

        # Crop data along X-axis
        cropped_data = data[x_min:x_max + 1, :, :]

        # Update affine to preserve spatial location
        new_affine = affine.copy()
        new_affine[:3, 3] += affine[:3, 0] * x_min

        # Save the cropped image
        cropped_img = nib.Nifti1Image(cropped_data, affine=new_affine, header=img.header)
        nib.save(cropped_img, output_path)

        print(f"Cropped image saved to: {output_path}")
        print(f"X range: {x_min} to {x_max} (with offset = {offset})")
        print(f"Y range: {y_min} to {y_max} (with offset = {offset_y})")

        self.crop_x_and_y_by_target(
            input_path=self.neck_segmentation,
            output_path=self.neck_segmentation,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            target_label=target_label,
        )

    # estimate the Skin HU range
    def get_hu_from_dicom(self, input_folder):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(input_folder)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        image_np = sitk.GetArrayFromImage(image).astype(np.int16)
        hu_data = image_np

        return hu_data

    def estimate_skin_HU(self, input_folder, lower_q=5, upper_q=95):
        hu_volume = self.get_hu_from_dicom(input_folder=input_folder)

        valid = hu_volume[(hu_volume > -300) & (hu_volume < 400)]

        lower = np.percentile(valid, lower_q)
        upper = np.percentile(valid, upper_q)

        print(f"Estimated optimal skin HU range: {lower:.2f} to {upper:.2f}")
        return lower, upper

    @staticmethod
    def get_lowest_point(mask):
        """Get the lowest point (minimum z) in the largest connected component"""
        labeled_array, num_features = label(mask)
        if num_features == 0:
            return None
        counts = np.bincount(labeled_array.flatten())
        counts[0] = 0
        max_lbl = np.argmax(counts)
        coords = np.argwhere(labeled_array == max_lbl)
        min_z = np.min(coords[:, 2])
        lowest_points = coords[coords[:, 2] == min_z]
        return lowest_points[0]

    @staticmethod
    def get_highest_point(mask):
        """Get the highest point (maximum z) in the largest connected component"""
        labeled_array, num_features = label(mask)
        if num_features == 0:
            return None
        counts = np.bincount(labeled_array.flatten())
        counts[0] = 0
        max_lbl = np.argmax(counts)
        coords = np.argwhere(labeled_array == max_lbl)
        max_z = np.max(coords[:, 2])
        highest_points = coords[coords[:, 2] == max_z]
        return highest_points[0]

    def cut_below_plane_defined_by_points(self,
                                          target_label_base=50,
                                          target_label_slope1=50,
                                          target_label_slope2=91,
                                          axis='x',
                                          offset=5
                                          ):
        """
        Cut a volume by a slanted plane defined by two points and a base horizontal plane.
        Save both parts (above and below the plane) as separate files.

        :param target_label_base: Label for base horizontal plane
        :param target_label_slope1: Label for highest point of slanted plane
        :param target_label_slope2: Label for lowest point of slanted plane
        :param axis: 'x' or 'y', slanting direction
        :param offset: z-axis offset for slope steepness
        :param output_path_above: Path to save above-plane NIfTI file
        :param output_path_below: Path to save below-plane NIfTI file
        """
        # 1. Load image

        img = nib.load(self.final_seg_result)
        data = img.get_fdata()
        affine = img.affine
        axcodes = nib.aff2axcodes(affine)
        print(f"Axis directions (axcodes): {axcodes}")
        target_mask_c1 = (data == target_label_base)
        target_mask_c2 = (data == 49) if np.any(data == 49) else np.zeros_like(data, dtype=bool)
        target_mask_spinal_cord = (data == 79) if np.any(data == 79) else np.zeros_like(data, dtype=bool)
        # 2. Extract key points
        base_point = self.get_lowest_point((data == target_label_base).astype(np.uint8))
        pt1 = self.get_highest_point((data == target_label_slope1).astype(np.uint8))
        pt2 = self.get_lowest_point((data == target_label_slope2).astype(np.uint8))

        if base_point is None or pt1 is None or pt2 is None:
            print("Valid region not found for one or more labels.")
            return

        print(f"Lowest point of horizontal plane (label {target_label_base}): {base_point}")
        print(f"Start point of slope (highest, label {target_label_slope1}): {pt1}")
        print(f"End point of slope (lowest, label {target_label_slope2}): {pt2}")

        base_z = base_point[2]
        pt2[2] = pt2[2] - offset

        # 3. Define slanted cutting surface
        if axis == 'x':
            y1, z1 = pt1[1], pt1[2]
            y2, z2 = pt2[1], pt2[2]
            slope = (z2 - z1) / (y2 - y1 + 1e-6)
            y0 = y1
            get_z_cut = lambda x, y: slope * (y - y0) + z1
            print(f"Slope direction: y-z plane (perpendicular to x-axis), slope = {slope:.4f}")
        elif axis == 'y':
            x1, z1 = pt1[0], pt1[2]
            x2, z2 = pt2[0], pt2[2]
            slope = (z2 - z1) / (x2 - x1 + 1e-6)
            x0 = x1
            get_z_cut = lambda x, y: slope * (x - x0) + z1
            print(f"Slope direction: x-z plane (perpendicular to y-axis), slope = {slope:.4f}")
        else:
            raise ValueError("Parameter 'axis' must be either 'x' or 'y'")

        # 4. Prepare two outputs
        above_data = np.zeros_like(data)
        below_data = np.zeros_like(data)
        count_above = 0
        count_below = 0

        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                z_cut = get_z_cut(x, y)
                z_threshold = min(base_z, z_cut)
                for z in range(data.shape[2]):
                    if data[x, y, z] != 0:
                        if z >= z_threshold:
                            above_data[x, y, z] = data[x, y, z]
                            count_above += 1
                        else:
                            below_data[x, y, z] = data[x, y, z]
                            count_below += 1

        print(f"Total voxels saved above cutting plane: {count_above}")
        print(f"Total voxels saved below cutting plane: {count_below}")

        below_data[target_mask_c1] = target_label_base
        # Process label 49
        if np.any(target_mask_c2):
            below_data[below_data == 49] = 0
            below_data[target_mask_c2] = 49

        # Process label 79
        if np.any(target_mask_spinal_cord):
            below_data[below_data == 79] = 0
            below_data[target_mask_spinal_cord] = 79

        print(f"Target label {target_label_base} added back to below segmentation.")
        nib.save(nib.Nifti1Image(above_data, affine), self.final_seg_result)
        nib.save(nib.Nifti1Image(below_data, affine), self.neck_segmentation)
        print(f"Above-plane part saved to: {self.final_seg_result}")
        print(f"Below-plane part saved to: {self.neck_segmentation}")

    def apply_target_mask_from_source_to_target(self,
                                                source_image_path,
                                                target_image_path,
                                                output_image_path,
                                                target_label,
                                                assign_value=None
                                                ):
        """
        Apply a mask extracted from the source image to the target image.

        Args:
            source_image_path (str): Path to the source NIfTI file (.nii.gz) used to extract the target mask.
            target_image_path (str): Path to the target NIfTI file where the mask will be applied.
            output_image_path (str): Path to save the resulting NIfTI file.
            target_label (int): Label value in the source image to define the mask.
            assign_value (float or int, optional): Value to assign to the masked region in the target image.
                                                   If None, the original target values are kept.
        """
        # Load source and target images
        source_img = nib.load(source_image_path)
        target_img = nib.load(target_image_path)

        source_data = source_img.get_fdata()
        target_data = target_img.get_fdata()

        # Ensure the source and target images are aligned
        assert source_data.shape == target_data.shape, "Error: Source and target images must have the same shape."
        assert np.allclose(source_img.affine,
                           target_img.affine), "Error: Source and target images must have the same affine matrix."

        # Create a mask from the source image based on the target label
        mask = (source_data == target_label)

        # Apply the mask to the target image
        updated_data = target_data.copy()
        if assign_value is not None:
            updated_data[mask] = assign_value

        # Save the updated target image
        updated_img = nib.Nifti1Image(updated_data, affine=target_img.affine, header=target_img.header)
        nib.save(updated_img, output_image_path)

        print(f"Processing complete. Output saved to: {output_image_path}")

    def build_head_reference(self):

        def ras_extreme_anatomical(data, affine, label, mode="lowest", z_positive_up=True):
            vox = np.argwhere(data == label)
            if vox.size == 0:
                return None
            ras_coords = nib.affines.apply_affine(affine, vox)
            z_idx = np.argmin(ras_coords[:, 2]) if mode == "lowest" else np.argmax(ras_coords[:, 2])
            if not z_positive_up:
                z_idx = np.argmax(ras_coords[:, 2]) if mode == "lowest" else np.argmin(ras_coords[:, 2])
            return ras_coords[z_idx]

        def detect_frankfort_plane(ct_path):
            print(f"[INFO] Running TotalSegmentator on: {ct_path}")
            seg_img = totalsegmentator(
                input=ct_path,
                task="head_glands_cavities",
                ml=True,
                device="gpu" if torch.cuda.is_available() else "cpu"
            )

            data, aff = seg_img.get_fdata(), seg_img.affine

            z_dir = aff[:3, 2]
            z_positive_up = z_dir[2] > 0
            print(f"[INFO] Affine Z-axis direction: {z_dir}")
            print(f"[INFO] Anatomical superior = {'increasing Z' if z_positive_up else 'decreasing Z'}\n")

            ID_EYE_L = 1
            ID_PORION_L = 17
            ID_PORION_R = 16

            f1 = ras_extreme_anatomical(data, aff, ID_PORION_L, mode="highest", z_positive_up=z_positive_up)
            f3 = ras_extreme_anatomical(data, aff, ID_PORION_R, mode="highest", z_positive_up=z_positive_up)
            f2 = ras_extreme_anatomical(data, aff, ID_EYE_L, mode="lowest", z_positive_up=z_positive_up)

            if any(x is None for x in (f1, f2, f3)):
                raise RuntimeError(
                    "Missing key anatomical structures: eye_left / auditory_canal_left / auditory_canal_right")

            print("Frankfort plane key points (RAS):")
            print(f"F-1 (Porion Left - Superior) : {f1}")
            print(f"F-2 (Orbitale - Inferior)    : {f2}")
            print(f"F-3 (Porion Right - Superior): {f3}\n")
            return f1, f2, f3

        F1, F2, F3 = detect_frankfort_plane(self.source_volume_path)

        # Step 4: Define Frankfort coordinate system
        origin = (F1 + F3) / 2.0

        # x_axis = F3 - F1
        # # x_axis = F1 - F3
        # x_axis = x_axis / np.linalg.norm(x_axis)
        #
        # y_temp = F2 - origin
        # y_temp = y_temp - np.dot(y_temp, x_axis) * x_axis
        # y_axis = y_temp / np.linalg.norm(y_temp)
        #
        # z_axis = np.cross(x_axis, y_axis)
        # z_axis = z_axis / np.linalg.norm(z_axis)

        y_axis = F3 - F1
        y_axis = y_axis / np.linalg.norm(y_axis)

        x_temp = F2 - origin
        x_temp = x_temp - np.dot(x_temp, y_axis) * y_axis
        x_axis = x_temp / np.linalg.norm(x_temp)

        z_axis = np.cross(y_axis, x_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        self.x_ax = x_axis
        self.y_ax = y_axis
        self.z_ax = z_axis

        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        R_T = R.T

        with open(self.Inertia_parameters_file, 'r') as f:
            json_inertia = json.load(f)

        com_ras = np.array(json_inertia["Inertia Analysis Result"]["T7"]["value"])
        com_frankfort = R_T @ (com_ras - origin)

        inertia_tensor_ras = np.array(json_inertia["Inertia Analysis Result"]["T3"]["value"])
        inertia_tensor_frankfort = R_T @ inertia_tensor_ras @ R

        principal_axes_ras = np.array(json_inertia["Inertia Analysis Result"]["T9"]["value"]).T
        principal_axes_frankfort = R_T @ principal_axes_ras
        principal_moments = json_inertia["Inertia Analysis Result"]["T8"]["value"]

        mass_kg = json_inertia["Inertia Analysis Result"]["T2"]["value"]
        volume_cm3 = json_inertia["Inertia Analysis Result"]["T1"]["value"]

        frankfort_results = {
            "T1": {
                "name": "Volume",
                "value": volume_cm3,
                "unit": "cm³",
                "explanation": "Total segmented volume."
            },
            "T2": {
                "name": "Mass",
                "value": mass_kg,
                "unit": "kg",
                "explanation": "Total mass estimated from volume and label-specific density."
            },
            "T7_Frankfort": {
                "name": "Center of Mass (Frankfort Coordinate System)",
                "value": com_frankfort.tolist(),
                "unit": "mm",
                "explanation": "Center of mass transformed into Frankfort anatomical coordinate system."
            },
            "T3_Frankfort": {
                "name": "Total Inertia Tensor (Frankfort Coordinate System, mm²)",
                "value": inertia_tensor_frankfort.tolist(),
                "unit": "kg·mm²",
                "explanation": "Inertia tensor in Frankfort coordinate system using millimeter units."
            },
            "T4_Frankfort": {
                "name": "Total Inertia Tensor (Frankfort Coordinate System, cm²)",
                "value": [[v / 100.0 for v in row] for row in inertia_tensor_frankfort.tolist()],
                "unit": "kg·cm²",
                "explanation": "Inertia tensor in Frankfort coordinate system converted to square centimeters."
            },
            "T8": {
                "name": "Principal Moments of Inertia",
                "value": principal_moments,
                "unit": "kg·mm²",
                "explanation": "Resistance of the structure to rotation around each of its three principal axes."
            },
            "T9_Frankfort": {
                "name": "Principal Axes (columns: x,y,z) in Frankfort Frame",
                "value": [list(col) for col in zip(*principal_axes_frankfort)],
                "unit": "unit vectors",
                "explanation": "Each column is a unit direction vector of a principal axis in Frankfort coordinate system."
            }
        }

        metadata = json_inertia.get("Patient Metadata", {})
        output_json = {
            "Patient Metadata": metadata,
            "Inertia Analysis Result (Frankfort Space)": frankfort_results
        }

        output_path = self.Inertia_parameters_frankfort
        with open(output_path, "w") as f_out:
            json.dump(output_json, f_out, indent=2)
        print(f"Frankfort-based inertia results written to: {output_path}")

        # Step 7: Export Frankfort axes and key points (.mrk.json)
        frankfort_axes_markups = {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
            "markups": []
        }

        axis_names = ["Frankfort_X", "Frankfort_Y", "Frankfort_Z"]
        axis_labels = ["X (Left→Right)", "Y (Back→Front)", "Z (Up)"]
        axis_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        axis_vectors = [x_axis, y_axis, z_axis]
        length = 500.0

        for name, vec, color, label in zip(axis_names, axis_vectors, axis_colors, axis_labels):
            start = origin.tolist()
            end = (origin + vec * length).tolist()

            line_markup = {
                "type": "Line",
                "coordinateSystem": "RAS",
                "controlPoints": [
                    {
                        "id": "0",
                        "label": f"{name}_start",
                        "position": start,
                        "selected": True,
                        "visibility": True,
                        "positionStatus": "defined"
                    },
                    {
                        "id": "1",
                        "label": label,
                        "position": end,
                        "selected": True,
                        "visibility": True,
                        "positionStatus": "defined"
                    }
                ],
                "display": {
                    "color": color,
                    "opacity": 1.0,
                    "lineThickness": 0.5,
                    "textScale": 2.5,
                    "glyphType": "Sphere3D",
                    "glyphScale": 1.5,
                    "glyphSize": 3.0,
                    "pointLabelsVisibility": True
                },
                "labelFormat": "%N",
                "locked": False
            }
            frankfort_axes_markups["markups"].append(line_markup)

        frankfort_axes_path = self.frank_coor
        with open(frankfort_axes_path, "w") as f_out:
            json.dump(frankfort_axes_markups, f_out, indent=2)
        print(f"Frankfort axes markups written to: {frankfort_axes_path}")

    def neck_frankfort_reference_cal(self, T: np.ndarray,
                                     frame_name: str = "Frankfort"):
        """
        Transforms inertial parameters from RAS space to a specified coordinate system,
        and writes all parameters into a unified JSON file.

        Parameters
        ----------
        T : np.ndarray
            4x4 homogeneous transformation matrix (RAS → new coordinate system)
        input_json_path : str
            Path to the original JSON file containing inertial analysis results
        output_json_path : str
            Output path for the JSON file containing the transformed parameters
        frame_name : str
            Name of the coordinate system (used for field naming)
        """

        input_json_path = self.Inertia_parameters_file_neck
        output_json_path = self.Inertia_parameters_frankfort_neck
        with open(input_json_path, 'r') as f:
            json_inertia = json.load(f)

        R = T[0:3, 0:3]  # Rotation matrix from world (RAS) to the new frame
        R_T = R  # Since T transforms from world to the new frame, R is used directly
        origin = -R_T.T @ T[0:3, 3]  # Origin position for calculating the offset

        # 提取原始参数
        raw = json_inertia["Inertia Analysis Result"]
        volume_cm3 = raw["T1"]["value"]
        mass_kg = raw["T2"]["value"]
        com_ras = np.array(raw["T7"]["value"])
        inertia_tensor_ras = np.array(raw["T3"]["value"])
        principal_moments = raw["T8"]["value"]
        principal_axes_ras = np.array(raw["T9"]["value"]).T  # (3, 3)

        com_transformed = R_T @ (com_ras - origin)
        inertia_tensor_transformed = R_T @ inertia_tensor_ras @ R_T.T
        principal_axes_transformed = R_T @ principal_axes_ras

        frankfort_results = {
            "T1": {
                "name": "Volume",
                "value": volume_cm3,
                "unit": "cm³",
                "explanation": "Total segmented volume."
            },
            "T2": {
                "name": "Mass",
                "value": mass_kg,
                "unit": "kg",
                "explanation": "Total mass estimated from volume and label-specific density."
            },
            f"T7_{frame_name}": {
                "name": f"Center of Mass ({frame_name} Coordinate System)",
                "value": com_transformed.tolist(),
                "unit": "mm",
                "explanation": f"Center of mass transformed into {frame_name} anatomical coordinate system."
            },
            f"T3_{frame_name}": {
                "name": f"Total Inertia Tensor ({frame_name} Coordinate System, mm²)",
                "value": inertia_tensor_transformed.tolist(),
                "unit": "kg·mm²",
                "explanation": f"Inertia tensor in {frame_name} coordinate system using millimeter units."
            },
            f"T4_{frame_name}": {
                "name": f"Total Inertia Tensor ({frame_name} Coordinate System, cm²)",
                "value": [[v / 100.0 for v in row] for row in inertia_tensor_transformed.tolist()],
                "unit": "kg·cm²",
                "explanation": f"Inertia tensor in {frame_name} coordinate system converted to square centimeters."
            },
            "T8": {
                "name": "Principal Moments of Inertia",
                "value": principal_moments,
                "unit": "kg·mm²",
                "explanation": "Resistance of the structure to rotation around each of its three principal axes."
            },
            f"T9_{frame_name}": {
                "name": f"Principal Axes (columns: x,y,z) in {frame_name} Frame",
                "value": [list(col) for col in zip(*principal_axes_transformed)],
                "unit": "unit vectors",
                "explanation": f"Each column is a unit direction vector of a principal axis in {frame_name} coordinate system."
            }
        }

        output_json = {
            "Patient Metadata": json_inertia.get("Patient Metadata", {}),
            f"Inertia Analysis Result ({frame_name} Space)": frankfort_results
        }

        with open(output_json_path, "w") as f_out:
            json.dump(output_json, f_out, indent=2)

        print(f"[INFO] Inertia parameters in '{frame_name}' frame written to: {output_json_path}")
        return frankfort_results

    def build_frankfort_transform(self,
                                  x_axis: np.ndarray,
                                  y_axis: np.ndarray,
                                  z_axis: np.ndarray,
                                  new_origin: np.ndarray,
                                  coordinate_name_prefix: str = "Frankfort",
                                  axis_length: float = 500.0) -> np.ndarray:
        """
        Constructs a homogeneous transformation matrix for the Frankfort coordinate system, with an optional output of coordinate axis annotations in JSON format for 3D visualization.

        Parameters
        ----------
        x_axis : np.ndarray
            Unit vector of the X-axis in the Frankfort coordinate system (shape: [3,])
        y_axis : np.ndarray
            Unit vector of the Y-axis in the Frankfort coordinate system (shape: [3,])
        z_axis : np.ndarray
            Unit vector of the Z-axis in the Frankfort coordinate system (shape: [3,])
        new_origin : np.ndarray
            New origin of the coordinate system (shape: [3,])
        json_output_path : str
            If provided, the coordinate axes will be exported as a JSON file for 3D visualization
        coordinate_name_prefix : str
            Prefix for the coordinate axis labels, e.g., "Frankfort" or "COM_Frame"
        axis_length : float
            Length of the coordinate axis arrows (unit: mm)

        Returns
        -------
        T : np.ndarray
            4x4 homogeneous transformation matrix (from RAS space to the translated local Frankfort coordinate system)
        """

        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        T = np.eye(4)
        T[0:3, 0:3] = R.T
        T[0:3, 3] = -R.T @ new_origin
        json_output_path = self.neck_reference
        if json_output_path:
            axis_names = [f"{coordinate_name_prefix}_X", f"{coordinate_name_prefix}_Y", f"{coordinate_name_prefix}_Z"]
            axis_labels = ["X (Left→Right)", "Y (Back→Front)", "Z (Up)"]
            axis_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
            axis_vectors = [x_axis, y_axis, z_axis]

            markups = {
                "markups": [],
                "coordinateSystem": "RAS",
                "version": "4.11",
                "type": "Markups"
            }

            for name, vec, color, label in zip(axis_names, axis_vectors, axis_colors, axis_labels):
                start = new_origin.tolist()
                end = (new_origin + vec * axis_length).tolist()

                line_markup = {
                    "type": "Line",
                    "coordinateSystem": "RAS",
                    "controlPoints": [
                        {
                            "id": "0",
                            "label": f"{name}_start",
                            "position": start,
                            "selected": True,
                            "visibility": True,
                            "positionStatus": "defined"
                        },
                        {
                            "id": "1",
                            "label": label,
                            "position": end,
                            "selected": True,
                            "visibility": True,
                            "positionStatus": "defined"
                        }
                    ],
                    "display": {
                        "color": color,
                        "opacity": 1.0,
                        "lineThickness": 0.5,
                        "textScale": 2.5,
                        "glyphType": "Sphere3D",
                        "glyphScale": 1.5,
                        "glyphSize": 3.0,
                        "pointLabelsVisibility": True
                    },
                    "labelFormat": "%N",
                    "locked": False
                }

                markups["markups"].append(line_markup)

            with open(json_output_path, "w") as f:
                json.dump(markups, f, indent=2)
            print(f"[INFO] Frankfort axes JSON written to: {json_output_path}")

        return T
    def build_neck_reference(self):
        points = self.get_ras_bounding_points(target_label=44.0)
        T = self.build_frankfort_transform(x_axis=self.x_ax,
                                       y_axis=self.y_ax,
                                       z_axis=self.z_ax,
                                       new_origin=points["y_mid_point"])
        res = self.neck_frankfort_reference_cal(T)

        return res

    def get_ras_bounding_points(self, target_label):
        img = nib.load(self.neck_segmentation)
        data = img.get_fdata()
        affine = img.affine

        mask = (data == target_label)
        if not np.any(mask):
            return None

        coords = np.array(np.where(mask))  # shape (3, N)

        x_idx = coords[0]
        y_idx = coords[1]
        z_idx = coords[2]

        def get_point(axis_idx, other1, other2, axis=0, mode='min'):
            if mode == 'min':
                val = axis_idx.min()
            else:
                val = axis_idx.max()
            mask = (axis_idx == val)
            i = np.where(mask)[0][0]
            point = np.array([x_idx[i], y_idx[i], z_idx[i]])
            ras_point = affine @ np.append(point, 1)
            return ras_point[:3]

        points = {
            'x_min_point': get_point(x_idx, y_idx, z_idx, axis=0, mode='min'),
            'x_max_point': get_point(x_idx, y_idx, z_idx, axis=0, mode='max'),
            'y_min_point': get_point(y_idx, x_idx, z_idx, axis=1, mode='min'),
            'y_max_point': get_point(y_idx, x_idx, z_idx, axis=1, mode='max'),
            'z_min_point': get_point(z_idx, x_idx, y_idx, axis=2, mode='min'),
            'z_max_point': get_point(z_idx, x_idx, y_idx, axis=2, mode='max'),
        }
        if points['x_min_point'] is not None and points['x_max_point'] is not None and points[
            'y_max_point'] is not None:
            x_mid = (points['x_min_point'][0] + points['x_max_point'][0]) / 2
            y_val = points['y_max_point'][1]
            z_val = points['y_max_point'][2]
            points['y_mid_point'] = np.array([x_mid, y_val, z_val])
        else:
            points['y_mid_point'] = None

        return points


# Helper methods for visualization
def _decimate_polydata(polydata, reduction=0.3):
    decimator = vtk.vtkDecimatePro()
    decimator.SetInputData(polydata)
    decimator.SetTargetReduction(reduction)
    decimator.PreserveTopologyOn()
    decimator.Update()
    return decimator.GetOutput()


def _transform_to_origin(polydata):
    bounds = polydata.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    transform = vtk.vtkTransform()
    transform.Translate(-xmin, -ymin, -zmin)

    tf_filter = vtk.vtkTransformPolyDataFilter()
    tf_filter.SetInputData(polydata)
    tf_filter.SetTransform(transform)
    tf_filter.Update()

    return tf_filter.GetOutput(), dx, dy, dz, transform


def _create_line_actor(pt1, pt2, color=(0.7, 0.7, 0.7), width=2.0):
    line_source = vtk.vtkLineSource()
    line_source.SetPoint1(pt1)
    line_source.SetPoint2(pt2)
    line_source.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(line_source.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(width)
    return actor


def visualize_with_coordinate_axes(original_com=None, decimate_ratio=0.0, vtk_set=None, output_image=None):
    """
    Reads the self.vtk_path VTK model and visualizes it with:
      1) Model aligned so that the bounding box min corner is at (0,0,0)
      2) vtkCubeAxesActor for annotated axes
      3) If original_com is provided, displays a sphere at the center of mass
         (plus lines projecting to x=0, y=0, and z=0)
      4) decimate_ratio can reduce the polygon count to speed up rendering
      5) A screenshot is saved to self.output_image, capturing the entire model
    """

    # ----------------- Read the VTK mesh ----------------- #
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_set)
    reader.Update()
    polydata = reader.GetOutput()

    # ----------------- Optional decimation ----------------- #
    if decimate_ratio > 0:
        polydata = _decimate_polydata(polydata, decimate_ratio)

    # ----------------- Translate geometry to origin ----------------- #
    transformed_polydata, dx, dy, dz, transform = _transform_to_origin(polydata)

    # ----------------- Create actor from polydata ----------------- #
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(transformed_polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.2)  # Make main actor partially transparent

    # ----------------- Create renderer, add actor ----------------- #
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)  # dark background

    # ----------------- Add Cube Axes ----------------- #
    cube_axes = vtk.vtkCubeAxesActor()
    cube_axes.SetBounds(0, dx, 0, dy, 0, dz)
    cube_axes.SetCamera(renderer.GetActiveCamera())
    cube_axes.SetXTitle("X Axis (mm)")
    cube_axes.SetYTitle("Y Axis (mm)")
    cube_axes.SetZTitle("Z Axis (mm)")
    # Some VTK versions might require .SetFlyModeToStatic() or .SetFlyModeToStaticEdges()
    cube_axes.SetFlyModeToStaticEdges()
    for i in range(3):
        cube_axes.GetTitleTextProperty(i).SetColor(1, 1, 1)
        cube_axes.GetLabelTextProperty(i).SetColor(1, 1, 1)
    renderer.AddActor(cube_axes)

    # ----------------- Optionally add COM sphere + lines ----------------- #
    if original_com is not None:
        tx, ty, tz = transform.GetPosition()
        cx_new = original_com[0] + tx
        cy_new = original_com[1] + ty
        cz_new = original_com[2] + tz

        # 1) Create a bigger, brighter sphere at COM
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(cx_new, cy_new, cz_new)
        sphere_source.SetRadius(10.0)  # e.g., 10.0 mm radius
        sphere_source.Update()

        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())
        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # bright red
        renderer.AddActor(sphere_actor)

        # 2) Create lines from COM to the bounding planes
        line_com_x = _create_line_actor(
            (cx_new, cy_new, cz_new),
            (0, cy_new, cz_new),
            color=(0.0, 1.0, 0.0),  # bright green
            width=3.0
        )
        line_com_y = _create_line_actor(
            (cx_new, cy_new, cz_new),
            (cx_new, 0, cz_new),
            color=(0.0, 1.0, 0.0),
            width=3.0
        )
        line_com_z = _create_line_actor(
            (cx_new, cy_new, cz_new),
            (cx_new, cy_new, 0),
            color=(0.0, 1.0, 0.0),
            width=3.0
        )
        renderer.AddActor(line_com_x)
        renderer.AddActor(line_com_y)
        renderer.AddActor(line_com_z)

        # 3) Add text overlay for COM
        text_actor = vtk.vtkTextActor()
        text_info = (
            f"Original COM: ({original_com[0]:.2f}, {original_com[1]:.2f}, {original_com[2]:.2f})\n"
            f"New COM: ({cx_new:.2f}, {cy_new:.2f}, {cz_new:.2f})"
        )
        text_actor.SetInput(text_info)
        text_actor.GetTextProperty().SetFontSize(18)
        text_actor.GetTextProperty().SetColor(1, 1, 1)
        text_actor.SetDisplayPosition(10, 10)
        renderer.AddActor2D(text_actor)

    # ----------------- Create a window + in
    # teractor ----------------- #
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1024, 768)  # Larger window for a bigger screenshot
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # ----------------- Ensure entire model is in view ----------------- #
    renderer.ResetCamera()

    # ----------------- Render the scene first ----------------- #
    render_window.Render()
    if output_image != None:
        # ----------------- Take a screenshot ----------------- #
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(render_window)
        window_to_image_filter.Update()

        png_writer = vtk.vtkPNGWriter()
        png_writer.SetFileName(output_image)
        png_writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        png_writer.Write()
        print(f"Screenshot saved to: {output_image}")

    # ----------------- Start interactive window ----------------- #
    print("VTK window started. Press 'q' or close the window to exit.")
    interactor.Start()


if __name__ == "__main__":
    file_list = [
        r"D:\DICOM_CHILD\15\0\15_0770"
    ]

    name_list = [
        "15_F_15_0770",
    ]

    for dicom_folder, name in zip(file_list, name_list):
        print(f"\nProcessing {name} in folder {dicom_folder}")
        if not os.path.exists(dicom_folder):
            print(f"[ERROR] Path does not exist: {dicom_folder}")
            continue
        slicer = AutoSlicer(name)
        lower, upper = slicer.estimate_skin_HU(dicom_folder)
        slicer.set_threshold(lower, upper)
        results = slicer.run_automation(dicom_folder)
        res1 = slicer.build_head_reference()
        res2 = slicer.build_neck_reference()
        print("[Head Ref]:", res1)
        print("[Neck Ref]:", res2)
        print("[Segmentation Stats]:", results)