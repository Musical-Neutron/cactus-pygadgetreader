#!/usr/bin/env python3

import os
import struct
import sys
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

# Import the Header class
from readgadget.modules.header import Header


@pytest.fixture
def mock_args():
    return {
        "nth": 2,
        "single": 1,
        "debug": 1,
        "units": 1,
        "blockordering": "romeel",
        "suppress": 1,
        "double": 1,
        "UnitMass_in_g": 2e33,
        "UnitLength_in_cm": 1e21,
        "UnitVelocity_in_cm_per_s": 1e6,
    }


@pytest.fixture
def gadget1_binary_header():
    # Construct a mock Gadget-1 format header
    header_data = struct.pack("I", 256)  # Skip block size
    header_data += struct.pack("6I", 10, 20, 0, 0, 5, 0)  # npartThisFile
    header_data += struct.pack("6d", 0, 1e10, 0, 0, 5e9, 0)  # massTable
    header_data += struct.pack("d", 0.5)  # time
    header_data += struct.pack("d", 1.0)  # redshift
    header_data += struct.pack("i", 1)  # flag_sfr
    header_data += struct.pack("i", 1)  # flag_fb
    header_data += struct.pack("6I", 100, 200, 0, 0, 50, 0)  # npartTotal
    header_data += struct.pack("i", 1)  # flag_cool
    header_data += struct.pack("i", 8)  # nfiles
    header_data += struct.pack("d", 100.0)  # boxsize
    header_data += struct.pack("d", 0.3)  # Omega0
    header_data += struct.pack("d", 0.7)  # OmegaLambda
    header_data += struct.pack("d", 0.7)  # HubbleParam
    header_data += struct.pack("i", 1)  # flag_age
    header_data += struct.pack("i", 1)  # flag_metals
    header_data += struct.pack("6I", 0, 0, 0, 0, 0, 0)  # npartTotalHW
    header_data += struct.pack("i", 0)  # flag_entropy
    header_data += struct.pack("i", 0)  # flag_doubleprecision
    header_data += struct.pack("i", 1)  # flag_potential
    header_data += struct.pack("i", 0)  # flag_fH2
    header_data += struct.pack("i", 0)  # flag_tmax
    header_data += struct.pack("i", 0)  # flag_delaytime
    # Pad to 256 bytes
    header_data += b"\x00" * (256 - len(header_data))
    header_data += struct.pack("I", 256)  # End skip block
    return header_data


def test_setVars(mock_args):
    header = Header.__new__(Header)
    header.args = mock_args
    header.setVars()

    assert header.nth == 2
    assert header.singleFile is True
    assert header.debug is True
    assert header.units is True
    assert header.suppress is True
    assert header.double is True
    assert header.UnitMass_in_g == 2e33
    assert header.UnitLength_in_cm == 1e21
    assert header.UnitVelocity_in_cm_per_s == 1e6


def test_calcRhoCrit(mock_args):
    header = Header.__new__(Header)
    header.args = mock_args
    header.setVars()
    header.HubbleParam = 0.7
    header.Omega0 = 0.3
    header.OmegaLambda = 0.7
    header.redshift = 1.0
    header.calcRhoCrit()
    expected = (
        3
        * (0.7 * 100 * np.sqrt(0.7 + 0.3 * (1 + 1) ** 3) / 3.08567758e19) ** 2
        / (8 * np.pi * 6.674e-8)
    )
    assert np.isclose(header.rhocrit, expected)


@pytest.mark.parametrize(
    "snap_passed, f_type",
    [
        ("snap.hdf5", "hdf5"),
        ("snap.bin", "tipsy"),
        ("snap", "gadget"),
    ],
)
def test_detectFileType_single(snap_passed, f_type):
    # Create Header without triggering file operations
    header = Header.__new__(Header)
    header.filenum = 0
    header.args = {"single": 1}

    # Test HDF5 detection
    header.snap_passed = snap_passed
    header.setVars()
    assert header.detectFileType() == f_type
    assert header.snap == snap_passed


@pytest.mark.parametrize(
    "snap_passed, f_type",
    [
        ("snap", "gadget"),
        ("snap.0", "gadget"),
        ("snap.hdf5", "hdf5"),
        ("snap.0.hdf5", "hdf5"),
        ("snap.bin", "tipsy"),
    ],
)
@patch("os.path.isfile")
def test_detectFileType_multifile_first_pass(mock_isfile, snap_passed, f_type):
    header = Header.__new__(Header)
    header.snap_passed = snap_passed
    header.filenum = 3
    header.args = {"single": False}
    header.setVars()

    split_s_passed = snap_passed.split(".")
    if len(split_s_passed) > 1:
        if not split_s_passed[-1].isnumeric():
            s_passed = split_s_passed[0] + "." + split_s_passed[-1]
        else:
            s_passed = split_s_passed[0]
    else:
        s_passed = split_s_passed[0]

    mock_isfile.side_effect = lambda x: x in [s_passed]
    assert header.detectFileType() == f_type
    assert header.snap == s_passed


@pytest.mark.parametrize(
    "snap_passed, f_type",
    [
        ("snap", "gadget"),
        ("snap.0", "gadget"),
        ("snap.hdf5", "hdf5"),
        ("snap.0.hdf5", "hdf5"),
    ],
)
@patch("os.path.isfile")
def test_detectFileType_multifile_second_pass(mock_isfile, snap_passed, f_type):
    header = Header.__new__(Header)
    header.snap_passed = snap_passed
    header.filenum = 3
    header.args = {"single": False}
    header.setVars()

    split_s_passed = snap_passed.split(".")
    if len(split_s_passed) > 1:
        if not split_s_passed[-1].isnumeric():
            s_passed = (
                split_s_passed[0] + "." + str(header.filenum) + "." + split_s_passed[-1]
            )
        else:
            s_passed = split_s_passed[0] + "." + str(header.filenum)
    else:
        s_passed = split_s_passed[0] + "." + str(header.filenum)

    mock_isfile.side_effect = lambda x: x in [s_passed]
    assert header.detectFileType() == f_type
    assert header.snap == s_passed


@patch("builtins.open", new_callable=mock_open)
@patch("header.gadget1.skip", return_value=256)
def test_read_gadget_header_gadget1(mock_skip, mock_file, gadget1_binary_header):
    mock_file.return_value.read.return_value = gadget1_binary_header[
        4:
    ]  # Skip initial block size
    header = Header("dummy_snap", 0, {})
    header.fileType = "gadget1"
    header.read_gadget_header()

    assert header.npartThisFile.tolist() == [10, 20, 0, 0, 5, 0]
    assert header.massTable.tolist() == [0, 1e10, 0, 0, 5e9, 0]
    assert header.time == 0.5
    assert header.redshift == 1.0
    assert header.flag_sfr == 1
    assert header.flag_fb == 1
    assert header.npartTotal.tolist() == [100, 200, 0, 0, 50, 0]
    assert header.flag_cool == 1
    assert header.nfiles == 8
    assert header.boxsize == 100.0
    assert header.Omega0 == 0.3
    assert header.OmegaLambda == 0.7
    assert header.HubbleParam == 0.7
    assert header.flag_potential == 1


@patch("h5py.File")
def test_read_hdf5_header(mock_h5py):
    mock_file = MagicMock()
    mock_h5py.return_value = mock_file
    mock_header = MagicMock()
    mock_file.__getitem__.return_value = mock_header
    mock_header.attrs = {
        "NumPart_ThisFile": [10, 20, 0, 0, 5, 0],
        "MassTable": [0, 1e10, 0, 0, 5e9, 0],
        "Time": 0.5,
        "Redshift": 1.0,
        "Flag_Sfr": 1,
        "Flag_Feedback": 1,
        "NumPart_Total": [100, 200, 0, 0, 50, 0],
        "Flag_Cooling": 1,
        "NumFilesPerSnapshot": 8,
        "BoxSize": 100.0,
        "Omega0": 0.3,
        "OmegaLambda": 0.7,
        "HubbleParam": 0.7,
        "Flag_StellarAge": 1,
        "Flag_Metals": 1,
        "NumPart_Total_HighWord": [0, 0, 0, 0, 0, 0],
    }
    mock_file.__contains__.side_effect = lambda x: x == "PartType0/Potential"

    header = Header("snap.hdf5", 0, {})
    header.read_hdf5_header()

    assert header.npartThisFile.tolist() == [10, 20, 0, 0, 5, 0]
    assert header.massTable.tolist() == [0, 1e10, 0, 0, 5e9, 0]
    assert header.time == 0.5
    assert header.redshift == 1.0
    assert header.flag_sfr == 1
    assert header.flag_fb == 1
    assert header.npartTotal.tolist() == [100, 200, 0, 0, 50, 0]
    assert header.flag_cool == 1
    assert header.nfiles == 8
    assert header.boxsize == 100.0
    assert header.Omega0 == 0.3
    assert header.OmegaLambda == 0.7
    assert header.HubbleParam == 0.7
    assert header.flag_potential == 1


@patch("builtins.open", new_callable=mock_open)
def test_read_tipsy_header(mock_file):
    tipsy_header = struct.pack(
        "d",
        0.5,  # time
        "i",
        100,  # ntotal
        "i",
        3,  # ndim
        "i",
        30,  # ngas
        "i",
        60,  # ndark
        "i",
        10,  # nstar
        "f",
        0.0,  # alignment
    )
    mock_file.return_value.read.return_value = tipsy_header

    header = Header("snap.bin", 0, {})
    header.read_tipsy_header()

    assert header.npartThisFile.tolist() == [30, 60, 0, 0, 10, 0]
    assert header.time == 0.5
    assert header.redshift == 1.0  # 1/0.5 - 1


def test_header_init_gadget(mock_args, gadget1_binary_header):
    with patch("builtins.open", mock_open(read_data=gadget1_binary_header)):
        with patch("header.gadget1.skip", return_value=256):
            header = Header("dummy_snap", 0, mock_args)

            # Check header_vals
            assert header.header_vals["ngas"] == 100
            assert header.header_vals["ndm"] == 200
            assert header.header_vals["time"] == 0.5
            assert header.header_vals["redshift"] == 1.0
            assert header.header_vals["boxsize"] == 100.0
            assert header.header_vals["O0"] == 0.3
            assert header.header_vals["Ol"] == 0.7
            assert header.header_vals["h"] == 0.7
            assert header.header_vals["flag_potential"] == 1
            assert header.BLOCKORDER is not None
