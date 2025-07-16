from autoslicer import AutoSlicer

if __name__ == "__main__":
    file_path = r"D:\DICOM_CHILD\15\0\15_0770"
    workspace = "DICOM_result"

    slicer = AutoSlicer(workspace)
    lower, upper = slicer.estimate_skin_HU(file_path)
    slicer.set_threshold(lower, upper)
    results = slicer.run_automation(file_path)
    res1 = slicer.build_head_reference()
    res2 = slicer.build_neck_reference()
    print("[Head Ref]:", res1)
    print("[Neck Ref]:", res2)
    print("[Segmentation Stats]:", results)