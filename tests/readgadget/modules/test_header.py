#!/usr/bin/env python3

import os
import struct
import sys
import tempfile
from unittest.mock import MagicMock, mock_open, patch

import h5py
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
def common_header_values():
    return {
        "npartThisFile": (
            np.array([10, 20, 0, 0, 5, 0], dtype=np.uint32),
            "NumPart_ThisFile",
        ),
        "massTable": (np.array([0, 1e10, 0, 0, 5e9, 0], dtype=np.float64), "MassTable"),
        "time": (np.float64(0.5), "Time"),
        "redshift": (np.float64(1.0), "Redshift"),
        "flag_sfr": (np.int32(1), "Flag_Sfr"),
        "flag_fb": (np.int32(1), "Flag_Feedback"),
        "npartTotal": (
            np.array([100, 200, 0, 0, 50, 0], dtype=np.uint32),
            "NumPart_Total",
        ),
        "flag_cool": (np.int32(1), "Flag_Cooling"),
        "nfiles": (np.int32(8), "NumFilesPerSnapshot"),
        "boxsize": (np.float64(100.0), "BoxSize"),
        "Omega0": (np.float64(0.3), "Omega0"),
        "OmegaLambda": (np.float64(0.7), "OmegaLambda"),
        "HubbleParam": (np.float64(0.7), "HubbleParam"),
        "flag_age": (np.int32(1), "Flag_StellarAge"),
        "flag_metals": (np.int32(1), "Flag_Metals"),
        "npartTotalHW": (
            np.array([0, 0, 0, 0, 0, 0], dtype=np.uint32),
            "NumPart_Total_Highword",
        ),
        "flag_entropy": (np.int32(0), None),
        "flag_doubleprecision": (np.int32(0), None),
        "flag_potential": (np.int32(1), None),
        "flag_fH2": (np.int32(0), None),
        "flag_tmax": (np.int32(0), None),
        "flag_delaytime": (np.int32(0), None),
    }


@pytest.fixture
def binary_header_order():
    return [
        "npartThisFile",
        "massTable",
        "time",
        "redshift",
        "flag_sfr",
        "flag_fb",
        "npartTotal",
        "flag_cool",
        "nfiles",
        "boxsize",
        "Omega0",
        "OmegaLambda",
        "HubbleParam",
        "flag_age",
        "flag_metals",
        "npartTotalHW",
        "flag_entropy",
        "flag_doubleprecision",
        "flag_potential",
        "flag_fH2",
        "flag_tmax",
        "flag_delaytime",
    ]


def create_binary_header(
    header_values, field_order, header_prepend=None, block_size=256
):
    header_data = b""
    if header_prepend is not None:
        header_data += struct.pack("i", 8)
        header_data += struct.pack("4s", b"HEAD")
        header_data += struct.pack(
            "I", sum(get_value(header_values, "npartThisFile")) * 3 * 4
        )
        header_data += struct.pack("I", 0)
    header_data += struct.pack("I", block_size)

    for key in field_order:
        value = get_value(header_values, key)
        data_dtype = value.dtype if hasattr(value, "dtype") else type(value)

        if np.issubdtype(data_dtype, np.signedinteger):
            pack_fmt = "i"
        elif np.issubdtype(data_dtype, np.unsignedinteger):
            pack_fmt = "I"
        else:
            pack_fmt = "d"

        if isinstance(value, np.ndarray):
            pack_fmt = f"{len(value)}{pack_fmt}"
            header_data += struct.pack(pack_fmt, *value)
        else:
            header_data += struct.pack(pack_fmt, value)

    # Pad to initial_block_size bytes and add closing block size
    current_len = len(header_data) - (4 if header_prepend is None else 20)
    header_data += b"\x00" * (block_size - current_len)
    header_data += struct.pack("I", block_size)  # End skip block

    return header_data


def add_dummy_data(header_data, npart_values, gadget2=False):
    # Add dummy data block (positions) to allow data type detection
    ntot = sum(npart_values)  # Sum of npartThisFile
    nbytes = ntot * 3 * 4  # 3 coordinates * 4 bytes per float32

    if gadget2:
        header_data += struct.pack("I", 8)
        header_data += struct.pack("4s", b"POS")
        header_data += struct.pack("I", nbytes)
        header_data += struct.pack("I", 0)

    header_data += struct.pack("I", nbytes)  # Data block size
    header_data += b"\x00" * nbytes  # Dummy position data
    header_data += struct.pack("I", nbytes)  # End data block size
    return header_data


@pytest.fixture
def gadget1_binary_header(common_header_values, binary_header_order):
    base_header = create_binary_header(common_header_values, binary_header_order)

    return add_dummy_data(base_header, get_value(common_header_values, "npartThisFile"))


@pytest.fixture
def gadget2_binary_header(common_header_values, binary_header_order):
    base_header = create_binary_header(
        common_header_values, binary_header_order, header_prepend=True
    )

    return add_dummy_data(
        base_header, get_value(common_header_values, "npartThisFile"), gadget2=True
    )


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


class HeaderTestBase:
    FORMAT = None
    HEADER_FIXTURE = None

    @pytest.fixture
    def header_file(self, request):
        header_data = request.getfixturevalue(self.HEADER_FIXTURE)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(header_data)
            tmp.flush()
            yield tmp.name
            os.unlink(tmp.name)

    def test_header_values(self, header_file, common_header_values):
        header = Header.__new__(Header)
        header.snap = header_file
        header.fileType = self.FORMAT
        header.read_gadget_header()

        for key, expected_value in common_header_values.items():
            actual_value = getattr(header, key)
            if hasattr(actual_value, "tolist"):
                actual_value = actual_value.tolist()
                expected_value = expected_value[0].tolist()
            assert actual_value == expected_value, f"Mismatch in {key}"


class TestGadget1Header(HeaderTestBase):
    FORMAT = "gadget1"
    HEADER_FIXTURE = "gadget1_binary_header"


class TestGadget2Header(HeaderTestBase):
    FORMAT = "gadget2"
    HEADER_FIXTURE = "gadget2_binary_header"


# # @patch("builtins.open", new_callable=mock_open)
# # @patch("readgadget.modules.gadget1.skip", return_value=256)
# # def test_read_gadget_header_gadget1(mock_skip, mock_file, gadget1_binary_header):
# def test_read_gadget_header_gadget1(gadget1_binary_header):
#     # mock_file.return_value.read.return_value = gadget1_binary_header[
#     #     4:
#     # ]  # Skip initial block size

#     with tempfile.NamedTemporaryFile(delete=False) as tmp:
#         tmp.write(bytearray(gadget1_binary_header))
#         tmp_name = tmp.name

#     try:
#         header = Header.__new__(Header)
#         header.snap = tmp_name
#         header.fileType = "gadget1"
#         header.read_gadget_header()

#         assert header.npartThisFile.tolist() == [10, 20, 0, 0, 5, 0]
#         assert header.massTable.tolist() == [0, 1e10, 0, 0, 5e9, 0]
#         assert header.time == 0.5
#         assert header.redshift == 1.0
#         assert header.flag_sfr == 1
#         assert header.flag_fb == 1
#         assert header.npartTotal.tolist() == [100, 200, 0, 0, 50, 0]
#         assert header.flag_cool == 1
#         assert header.nfiles == 8
#         assert header.boxsize == 100.0
#         assert header.Omega0 == 0.3
#         assert header.OmegaLambda == 0.7
#         assert header.HubbleParam == 0.7
#         assert header.flag_potential == 1
#     finally:
#         # Clean up the temporary file
#         os.unlink(tmp_name)


def get_value(common_values, key):
    """Get the value from common_header_values"""
    return common_values[key][0]


def get_hdf5_attr_name(common_values, key):
    """Get the HDF5 attribute name from common_header_values"""
    return common_values[key][1] if len(common_values[key]) > 1 else None


def get_hdf5_attrs(common_values):
    """Get a dictionary of HDF5 attribute names and values"""
    return {
        get_hdf5_attr_name(common_values, k): get_value(common_values, k)
        for k in common_values
        if get_hdf5_attr_name(common_values, k) is not None
    }


@patch("h5py.File")
def test_read_hdf5_header(common_header_values):
    # Create temporary HDF5 file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as tmp:
        tmp_name = tmp.name

    try:
        # Create HDF5 file with header data
        with h5py.File(tmp_name, "w") as f:
            header_group = f.create_group("Header")

            # Set HDF5 attributes using common values
            hdf5_attrs = get_hdf5_attrs(common_header_values)
            for attr_name, attr_value in hdf5_attrs.items():
                header_group.attrs[attr_name] = attr_value

            # Create datasets for flags that depend on data presence
            if get_value(common_header_values, "flag_potential"):
                f.create_dataset("PartType0/Potential", data=np.array([1.0, 2.0, 3.0]))
            if get_value(common_header_values, "flag_fH2"):
                f.create_dataset("PartType0/FractionH2", data=np.array([0.1, 0.2, 0.3]))
            if get_value(common_header_values, "flag_tmax"):
                f.create_dataset(
                    "PartType0/TemperatureMax", data=np.array([100.0, 200.0, 300.0])
                )
            if get_value(common_header_values, "flag_delaytime"):
                f.create_dataset("PartType0/DelayTime", data=np.array([1.0, 2.0, 3.0]))

        # Test header reading
        header = Header.__new__(Header)
        header.snap = tmp_name
        header.read_hdf5_header()

        # Assertions
        assert header.fileType == "hdf5"

        # Check all values
        for key in common_header_values:
            if key in ["flag_entropy", "flag_doubleprecision"]:
                continue  # Skip these as they're not in HDF5

            expected_value = get_value(common_header_values, key)
            actual_value = getattr(header, key)

            if hasattr(actual_value, "tolist"):
                actual_value = actual_value.tolist()
                expected_value = expected_value.tolist()

            assert actual_value == expected_value, f"Mismatch in {key}"
    finally:
        os.unlink(tmp_name)

    # mock_file = MagicMock()
    # mock_h5py.return_value = mock_file
    # mock_header = MagicMock()
    # mock_file.__getitem__.return_value = mock_header
    # mock_header.attrs = {
    #     "NumPart_ThisFile": [10, 20, 0, 0, 5, 0],
    #     "MassTable": [0, 1e10, 0, 0, 5e9, 0],
    #     "Time": 0.5,
    #     "Redshift": 1.0,
    #     "Flag_Sfr": 1,
    #     "Flag_Feedback": 1,
    #     "NumPart_Total": [100, 200, 0, 0, 50, 0],
    #     "Flag_Cooling": 1,
    #     "NumFilesPerSnapshot": 8,
    #     "BoxSize": 100.0,
    #     "Omega0": 0.3,
    #     "OmegaLambda": 0.7,
    #     "HubbleParam": 0.7,
    #     "Flag_StellarAge": 1,
    #     "Flag_Metals": 1,
    #     "NumPart_Total_HighWord": [0, 0, 0, 0, 0, 0],
    # }
    # mock_file.__contains__.side_effect = lambda x: x == "PartType0/Potential"

    # header = Header("snap.0.hdf5", 0, {})
    # header.read_hdf5_header()

    # assert header.npartThisFile.tolist() == [10, 20, 0, 0, 5, 0]
    # assert header.massTable.tolist() == [0, 1e10, 0, 0, 5e9, 0]
    # assert header.time == 0.5
    # assert header.redshift == 1.0
    # assert header.flag_sfr == 1
    # assert header.flag_fb == 1
    # assert header.npartTotal.tolist() == [100, 200, 0, 0, 50, 0]
    # assert header.flag_cool == 1
    # assert header.nfiles == 8
    # assert header.boxsize == 100.0
    # assert header.Omega0 == 0.3
    # assert header.OmegaLambda == 0.7
    # assert header.HubbleParam == 0.7
    # assert header.flag_potential == 1


# @patch("builtins.open", new_callable=mock_open)
def test_read_tipsy_header(common_header_values):
    # Extract values for Tipsy format
    ngas = get_value(common_header_values, "npartThisFile")[0]
    ndark = get_value(common_header_values, "npartThisFile")[1]
    nstar = get_value(common_header_values, "npartThisFile")[4]
    time = get_value(common_header_values, "time")
    tipsy_header = struct.pack("d", time)
    tipsy_header += struct.pack("i", ngas + ndark + nstar)
    tipsy_header += struct.pack("i", 3)
    tipsy_header += struct.pack("i", ngas)
    tipsy_header += struct.pack("i", ndark)
    tipsy_header += struct.pack("i", nstar)
    tipsy_header += struct.pack("f", 0.0)

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp.write(tipsy_header)
        tmp_name = tmp.name

    try:
        header = Header.__new__(Header)
        header.snap = tmp_name
        header.read_tipsy_header()

        # Assertions
        assert header.fileType == "tipsy"
        assert header.npartThisFile == [ngas, ndark, 0, 0, nstar, 0]
        assert header.time == time
        assert header.redshift == 1.0 / time - 1.0
        assert header.flag_sfr == 1
        assert header.flag_fb == 1
        assert header.flag_cool == 1
        assert header.nfiles == 1
        assert header.boxsize == 0.0
        assert header.Omega0 == 0.0
        assert header.OmegaLambda == 0.0
        assert header.HubbleParam == 0.0
        assert header.flag_age == 1
        assert header.flag_metals == 1
        assert header.npartTotalHW == [0, 0, 0, 0, 0, 0]
        assert header.flag_potential == 1
        assert header.flag_fH2 == 0
        assert header.flag_tmax == 1
        assert header.flag_delaytime == 1
    finally:
        os.unlink(tmp_name)

    # mock_file.return_value.read.return_value = tipsy_header

    # header = Header("snap.bin", 0, {})
    # header.read_tipsy_header()

    # assert header.npartThisFile.tolist() == [30, 60, 0, 0, 10, 0]
    # assert header.time == 0.5
    # assert header.redshift == 1.0  # 1/0.5 - 1


def test_header_init_gadget(mock_args, gadget1_binary_header):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(gadget1_binary_header)
        tmp_name = tmp.name

    try:
        # Initialize Header with the temporary file
        header = Header(tmp_name, 0, mock_args)

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

        # Additional checks for mock_args
        assert header.nth == mock_args["nth"]
        assert header.singleFile == mock_args["single"]
        assert header.debug == mock_args["debug"]
        assert header.units == mock_args["units"]
        assert header.suppress == mock_args["suppress"]
        assert header.double == mock_args["double"]
        assert header.UnitMass_in_g == mock_args["UnitMass_in_g"]
        assert header.UnitLength_in_cm == mock_args["UnitLength_in_cm"]
        assert header.UnitVelocity_in_cm_per_s == mock_args["UnitVelocity_in_cm_per_s"]
    finally:
        # Clean up the temporary file
        os.unlink(tmp_name)
