"""DICOM synthetic test - verifies pydicom roundtrip."""
import tempfile
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from src.preprocessing import load_image

def make_synthetic_dicom(path):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0"*128)
    arr = (np.random.rand(64,64)*255).astype(np.uint16)
    ds.PixelData = arr.tobytes()
    ds.Rows, ds.Columns = 64,64
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(path)
    return path

def test_synthetic_dicom_load():
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        make_synthetic_dicom(tmp.name)
        img = load_image(tmp.name)
        assert img.size[0] > 0

def test_quality_check_on_pil():
    from PIL import Image
    from src.quality_check import check_blur, check_contrast
    im = Image.new("RGB", (224,224), "gray")
    ok, score = check_blur(im)
    assert isinstance(score, float)
    ok2, std = check_contrast(im)
    assert isinstance(std, float)
