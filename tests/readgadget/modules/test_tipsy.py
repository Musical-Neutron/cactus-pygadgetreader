#!/usr/bin/env python3


import numpy as np
import pytest

from readgadget.modules.tipsy import (
    NMETALS,
    auxgasstruct,
    auxstarstruct,
    dmstruct,
    gasstruct,
    starstruct,
    tipsy_auxread,
    tipsy_binread,
    tipsy_pids,
    tipsy_read,
)


# Mock header object for testing
class MockHeader:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# Fixture to create temporary TIPSY files
@pytest.fixture
def setup_tipsy_files(tmp_path):
    # Create base file paths
    base_path = tmp_path / "test_snapshot"
    bin_path = base_path.with_suffix(".bin")
    aux_path = base_path.with_suffix(".aux")
    pid_path = base_path.with_suffix(".idnum")

    # Create test data
    ngas = 2
    ndm = 3
    nstar = 1
    npart = [ngas, ndm, 0, 0, nstar]

    # Create binary data
    gas_bin = np.zeros(ngas, dtype=gasstruct)
    gas_bin["mass"] = [1.0, 2.0]
    gas_bin["pos"] = [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]]
    gas_bin["vel"] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    gas_bin["rho"] = [0.7, 0.8]
    gas_bin["u"] = [0.9, 1.0]
    gas_bin["hsml"] = [1.1, 1.2]
    gas_bin["metallicity"] = [0.01, 0.02]
    gas_bin["pot"] = [-1.0, -2.0]

    dm_bin = np.zeros(ndm, dtype=dmstruct)
    dm_bin["mass"] = [10.0, 20.0, 30.0]
    dm_bin["pos"] = [[10.1, 10.2, 10.3], [20.1, 20.2, 20.3], [30.1, 30.2, 30.3]]
    dm_bin["vel"] = [[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]]
    dm_bin["hsml"] = [1.3, 2.3, 3.3]
    dm_bin["pot"] = [-10.0, -20.0, -30.0]

    star_bin = np.zeros(nstar, dtype=starstruct)
    star_bin["mass"] = [100.0]
    star_bin["pos"] = [[100.1, 100.2, 100.3]]
    star_bin["vel"] = [[10.0, 10.1, 10.2]]
    star_bin["metallicity"] = [0.1]
    star_bin["age"] = [5.0]
    star_bin["hsml"] = [10.3]
    star_bin["pot"] = [-100.0]

    # Create auxiliary data
    gas_aux = np.zeros(ngas, dtype=auxgasstruct)
    gas_aux["metalarray"] = [[0.01, 0.02, 0.03, 0.04], [0.02, 0.03, 0.04, 0.05]]
    gas_aux["sfr"] = [0.1, 0.2]
    gas_aux["tmax"] = [1.0, 2.0]
    gas_aux["delaytime"] = [0.5, 1.0]
    gas_aux["ne"] = [0.1, 0.2]
    gas_aux["nh"] = [0.3, 0.4]
    gas_aux["nspawn"] = [1, 2]

    star_aux = np.zeros(nstar, dtype=auxstarstruct)
    star_aux["metalarray"] = [[0.1, 0.2, 0.3, 0.4]]
    star_aux["age"] = [5.0]
    star_aux["tmax"] = [10.0]
    star_aux["nspawn"] = [1]

    # Create particle IDs
    gas_pids = np.array([100, 200], dtype=np.int32)
    dm_pids = np.array([300, 400, 500], dtype=np.int32)
    star_pids = np.array([600], dtype=np.int32)

    # Write binary file
    with open(bin_path, "wb") as f:
        gas_bin.tofile(f)
        dm_bin.tofile(f)
        star_bin.tofile(f)

    # Write auxiliary file
    with open(aux_path, "wb") as f:
        gas_aux.tofile(f)
        star_aux.tofile(f)

    # Write PID file
    with open(pid_path, "wb") as f:
        gas_pids.tofile(f)
        dm_pids.tofile(f)
        star_pids.tofile(f)

    return {
        "base_path": base_path,
        "ngas": ngas,
        "ndm": ndm,
        "nstar": nstar,
        "npart_file": npart,
        "gas_bin": gas_bin,
        "dm_bin": dm_bin,
        "star_bin": star_bin,
        "gas_aux": gas_aux,
        "star_aux": star_aux,
        "gas_pids": gas_pids,
        "dm_pids": dm_pids,
        "star_pids": star_pids,
    }


# Test TIPSY binread function
@pytest.mark.parametrize(
    "ptype, bin_string, struct_dict",
    [(0, "gas_bin", gasstruct), (1, "dm_bin", dmstruct), (4, "star_bin", starstruct)],
)
def test_tipsy_binread(setup_tipsy_files, ptype, bin_string, struct_dict):
    files = setup_tipsy_files
    bin_path = files["base_path"].with_suffix(".bin")

    with open(bin_path, "rb") as f:
        # Test gas particles
        header = MockHeader(npartThisFile=files["npart_file"])

        # Test each field in gasstruct
        for field in struct_dict.names:
            header.reading = field
            f.seek(0)  # Reset file pointer
            field_data = tipsy_binread(f, header, ptype)
            assert np.array_equal(field_data, files[bin_string][field])


# Test TIPSY auxread function
@pytest.mark.parametrize(
    "ptype, bin_string, struct_dict",
    [(0, "gas_aux", auxgasstruct), (4, "star_aux", auxstarstruct)],
)
def test_tipsy_auxread(setup_tipsy_files, ptype, bin_string, struct_dict):
    files = setup_tipsy_files
    aux_path = files["base_path"].with_suffix(".aux")

    with open(aux_path, "rb") as f:
        # Test gas particles
        header = MockHeader(npartThisFile=files["npart_file"])

        # Test each field in gasstruct
        for field in struct_dict.names:
            header.reading = field
            f.seek(0)  # Reset file pointer
            field_data = tipsy_auxread(f, header, ptype)
            assert np.array_equal(field_data, files[bin_string][field])


# Test TIPSY pids function
@pytest.mark.parametrize(
    "ptype, bin_string",
    [(0, "gas_pids"), (1, "dm_pids"), (4, "star_pids")],
)
def test_tipsy_pids(setup_tipsy_files, ptype, bin_string):
    files = setup_tipsy_files
    pid_path = files["base_path"].with_suffix(".idnum")

    with open(pid_path, "rb") as f:
        # Test gas particles
        header = MockHeader(npartThisFile=files["npart_file"])
        field_data = tipsy_pids(f, header, ptype)
        assert np.array_equal(field_data, files[bin_string])


# Test TIPSY read dispatch function
@pytest.mark.parametrize(
    "ptype, bin_string, struct_dict",
    [
        (0, "gas_bin", gasstruct),
        (1, "dm_bin", dmstruct),
        (4, "star_bin", starstruct),
        (0, "gas_aux", auxgasstruct),
        (4, "star_aux", auxstarstruct),
    ],
)
def test_tipsy_read_bin_and_aux(setup_tipsy_files, ptype, bin_string, struct_dict):
    files = setup_tipsy_files
    file_path = files["base_path"]
    bin_filepath = file_path.with_suffix(".bin")

    original_f = open(bin_filepath, "rb")

    # Test each field in gasstruct
    header = MockHeader(
        npartThisFile=files["npart_file"],
        f=open(bin_filepath, "rb"),
        snap_passed=file_path,
    )
    for field in struct_dict.names:
        header.reading = field
        original_f.seek(0)  # Reset file pointer
        data = tipsy_read(original_f, header, ptype)
        expected = files[bin_string][field]
        assert np.array_equal(data, expected)


@pytest.mark.parametrize(
    "ptype, bin_string",
    [(0, "gas_pids"), (1, "dm_pids"), (4, "star_pids")],
)
def test_tipsy_read_pids(setup_tipsy_files, ptype, bin_string):
    files = setup_tipsy_files
    file_path = files["base_path"]
    bin_filepath = file_path.with_suffix(".bin")

    original_f = open(bin_filepath, "rb")

    # Test each field in gasstruct
    header = MockHeader(
        npartThisFile=files["npart_file"],
        f=open(bin_filepath, "rb"),
        snap_passed=file_path,
        reading="pid",
    )
    data = tipsy_read(original_f, header, ptype)
    expected = files[bin_string]
    assert np.array_equal(data, expected)
