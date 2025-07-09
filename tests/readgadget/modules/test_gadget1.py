#!/usr/bin/env python3
import os
import tempfile
from io import BytesIO

import numpy as np
import pytest

from readgadget.modules.gadget1 import (
    errorcheck,
    gadget_read,
    gadget_readage,
    gadget_readgasprop,
    gadget_readgasstarprop,
    gadget_readmass,
    gadget_readmetals,
    gadget_readpid,
    gadget_readposvel,
    gadget_readpotentials,
    skip,
)
from readgadget.modules.gadget_blockordering import BLOCKORDERING0


# Fixture to patch module-level dictionaries
@pytest.fixture(autouse=True)
def patch_names(monkeypatch):
    monkeypatch.setattr("readgadget.modules.common.getTfactor", lambda ne, h: 2.0)
    monkeypatch.setattr("readgadget.modules.gadget1.METALFACTOR", 1.0)


# Mock Header Class for Testing
class MockHeader:
    def __init__(
        self,
        npartThisFile,
        dataType=np.float32,
        convert=1.0,
        massTable=None,
        flag_metals=1,
        BLOCKORDER=None,
        header_vals=None,
        debug=False,
        reading=None,
        units=None,
    ):
        self.BLOCKORDER = BLOCKORDER if BLOCKORDER else {}
        self.convert = convert
        self.dataType = dataType
        self.debug = debug
        self.flag_metals = flag_metals
        self.header_vals = header_vals if header_vals else {}
        self.massTable = massTable if massTable is not None else [0] * 6
        self.npartThisFile = npartThisFile
        self.reading = reading
        self.units = units


# Fixture for a mock binary file with skips
@pytest.fixture
def mock_gadget_file():
    """Creates real temporary files with Gadget-format skips, supports multiple blocks"""
    files = []

    def _create(blocks, dtypes=None):
        # If single block passed, convert to list
        if not isinstance(blocks, list):
            blocks = [blocks]

        # Handle dtypes - default to float32
        if dtypes is None:
            dtypes = [np.float32] * len(blocks)
        elif not isinstance(dtypes, list):
            dtypes = [dtypes] * len(blocks)

        # Create temp file
        fd, path = tempfile.mkstemp()
        files.append(path)

        with os.fdopen(fd, "wb") as f:
            for i, data in enumerate(blocks):
                dtype = dtypes[i]
                data_arr = np.array(data, dtype=dtype)
                skip_size = np.array([data_arr.nbytes], dtype=np.uint32)

                # Write skip header + data + skip footer
                f.write(skip_size.tobytes())
                f.write(data_arr.tobytes())
                f.write(skip_size.tobytes())

        return open(path, "rb")

    yield _create

    # Cleanup: Close and delete temp files
    for path in files:
        try:
            os.remove(path)
        except OSError:
            pass


# Test skip()
def test_skip(mock_gadget_file):
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mock_file = mock_gadget_file(data, dtypes=np.float32)
    assert skip(mock_file) == data.nbytes


# Test errorcheck()
def test_errorcheck_exits_on_mismatch(capsys):
    with pytest.raises(SystemExit):
        errorcheck(10, 20, "test_block")
    captured = capsys.readouterr()
    assert (
        "issue with before/after skips - block test_block >> 10 vs 20" in captured.out
    )


# Test gadget_readposvel()
def test_gadget_readposvel(mock_gadget_file):
    # Setup header: 2 particles of type 1
    h = MockHeader(npartThisFile=[0, 2, 0, 0, 0, 0])
    # Create mock file: 2 particles * 3 coordinates
    pos_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    mock_file = mock_gadget_file(pos_data)

    # Read positions for type 1 (index=1)
    result = gadget_readposvel(mock_file, h, 1)
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.allclose(result, expected)


# Test gadget_readpid() with uint32 and uint64
@pytest.mark.parametrize(
    "dtype, expected_type", [(np.uint32, np.uint32), (np.uint64, np.uint64)]
)
def test_gadget_readpid_valid(dtype, expected_type, mock_gadget_file):
    # Setup header: 2 particles of type 1
    h = MockHeader(npartThisFile=[0, 2, 0, 0, 0, 0])
    # Create mock file with PIDs
    pid_data = np.array([123, 456], dtype=dtype)
    mock_file = mock_gadget_file(pid_data, dtype)

    result = gadget_readpid(mock_file, h, 1)
    assert result.dtype == expected_type
    assert np.array_equal(result, pid_data)


@pytest.mark.parametrize("dtype", [(np.uint32), (np.uint64)])
def test_gadget_readpid_invalid(dtype, mock_gadget_file):
    # Setup header: 2 particles of type 1
    h = MockHeader(npartThisFile=[0, 2, 0, 0, 0, 0])
    # Create mock file with PIDs
    pid_data = np.array([123, 456, 1258], dtype=dtype)
    mock_file = mock_gadget_file(pid_data, dtype)

    with pytest.raises(SystemExit):
        gadget_readpid(mock_file, h, 1)


# Test gadget_readmass() with and without mass block
@pytest.mark.parametrize(
    "ptype, npart, mtable, mdata, expected",
    [
        (
            0,
            [0, 2, 0, 0, 1, 0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 20.0, 30.0],
            [1.0],
        ),
        (
            1,
            [0, 2, 0, 0, 1, 0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 20.0, 30.0],
            [10.0, 20.0],
        ),
        (
            4,
            [0, 2, 0, 0, 1, 0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 20.0, 30.0],
            [30.0],
        ),
    ],
)
def test_gadget_readmass_with_mass_block(
    mock_gadget_file, ptype, npart, mtable, mdata, expected
):
    # Mass block exists (massTable=0 for type 1)
    h = MockHeader(npartThisFile=npart, massTable=mtable)
    mass_data = np.array(mdata, dtype=np.float32)

    mock_file = mock_gadget_file(mass_data)
    result = gadget_readmass(mock_file, h, ptype)
    assert np.allclose(result, np.array(expected))


def test_gadget_readmass_without_mass_block():
    # Mass block absent (massTable has fixed mass for type 1)
    h = MockHeader(
        npartThisFile=[0, 2, 0, 0, 0, 0],
        massTable=[1.0, 5.0, 0.0, 0.0, 0.0, 0.0],  # Fixed mass=5.0 for type 1
    )
    # No file needed since mass is fixed
    mock_file = BytesIO(b"")  # Empty
    result = gadget_readmass(mock_file, h, 1)
    assert np.allclose(result, np.array([5.0, 5.0]))


# Test gadget_read() driver for 'pos'
def test_gadget_read_pos(mock_gadget_file):
    h = MockHeader(
        npartThisFile=[0, 2, 0, 0, 0, 0],
        reading="pos",
        BLOCKORDER={"pos": [1]},  # Simulate block order
    )
    pos_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    mock_file = mock_gadget_file(pos_data)

    result = gadget_read(mock_file, h, 1, "pos")
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.allclose(result, expected)


def test_gadget_readgasprop_basic(mock_gadget_file):
    h = MockHeader(npartThisFile=[2, 0, 0, 0, 0, 0], reading="u")
    gasprop_data = np.array([1.0, 2.0], dtype=np.float32)

    with mock_gadget_file(gasprop_data) as f:
        result = gadget_readgasprop(f, h)

    assert np.allclose(result, np.array([1.0, 2.0]))


def test_gadget_readgasprop_with_temp_calc(mock_gadget_file):
    # Mock common.getTfactor to return a fixed value
    # monkeypatch.setattr("readgadget.modules.common.getTfactor", lambda ne, h: 2.0)

    h = MockHeader(
        npartThisFile=[2, 0, 0, 0, 0, 0],
        reading="u",
        units=True,
        dataType=np.float32,
        convert=1.0,
    )

    # Create data with three blocks: gasprop + rho + ne
    blocks = [
        np.array([1.0, 2.0], dtype=np.float32),  # gasprop (u)
        np.array([3.0, 4.0], dtype=np.float32),  # rho (will be skipped)
        np.array([0.5, 0.6], dtype=np.float32),  # ne
    ]

    # Gasprop block
    with mock_gadget_file(blocks) as f:
        result = gadget_readgasprop(f, h)

    # Should be (gasprop * convert) = [1.0, 2.0] * 2.0
    assert np.allclose(result, np.array([2.0, 4.0]))


def test_gadget_readgasstarprop_gas(mock_gadget_file):
    h = MockHeader(npartThisFile=[2, 0, 0, 0, 1, 0])
    # Block contains: 2 gas props + 1 star prop
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with mock_gadget_file(data) as f:
        # Read gas props (type 0)
        result = gadget_readgasstarprop(f, h, 0)

    assert np.allclose(result, np.array([1.0, 2.0]))


def test_gadget_readgasstarprop_star(mock_gadget_file):
    h = MockHeader(npartThisFile=[2, 0, 0, 0, 1, 0])
    # Block contains: 2 gas props + 1 star prop
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with mock_gadget_file(data) as f:
        # Read star props (type 4)
        result = gadget_readgasstarprop(f, h, 4)

    assert np.allclose(result, np.array([3.0]))


@pytest.mark.parametrize(
    "ptype, flag_metals, single, expected",
    [
        # Single metallicity, gas particles
        (0, 1, 1, [0.5, 0.6]),
        # Single metallicity, star particles
        (4, 1, 1, [0.7]),
        # Multiple metals, return as array
        (0, 3, 0, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        # Multiple metals, return total
        (0, 3, 1, [0.6, 1.5]),  # Assuming METALFACTOR=1.0
    ],
)
def test_gadget_readmetals(mock_gadget_file, ptype, flag_metals, single, expected):
    # Mock METALFACTOR == 1
    h = MockHeader(
        npartThisFile=[2, 0, 0, 0, 1, 0], flag_metals=flag_metals, dataType=np.float32
    )

    # Create data based on flag_metals
    if flag_metals == 1:
        data = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    else:  # flag_metals=3
        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)

    with mock_gadget_file(data) as f:
        result = gadget_readmetals(f, h, ptype, single=single)

    assert np.allclose(result, np.array(expected))


def test_gadget_readpotentials(mock_gadget_file):
    h = MockHeader(npartThisFile=[1, 2, 0, 0, 1, 0])
    # Block contains potentials for all particles: [type0, type1, type1, type4]
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    with mock_gadget_file(data) as f:
        # Read potentials for type1
        result = gadget_readpotentials(f, h, 1)

    assert np.allclose(result, np.array([2.0, 3.0]))


def test_gadget_readage(mock_gadget_file):
    h = MockHeader(npartThisFile=[0, 0, 0, 0, 2, 0])
    data = np.array([1.0, 2.0], dtype=np.float32)

    with mock_gadget_file(data) as f:
        result = gadget_readage(f, h)

    assert np.allclose(result, np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    "reading, ptype, expected",
    [
        ("vel", 1, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # Velocity
        ("pos", 1, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),  # Position
        ("pid", 1, [10002, 10003]),  # Particle IDs
        ("mass", 4, [500.0, 600.0]),  # Particle masses
        ("u", 0, [1.0, 2.0]),  # Gas internal energy
        ("tmax", 4, [3.0, 4.0]),  # Star property
        ("metallicity", 0, [4.0, 8.0]),  # Metallicity
        ("metalarray", 0, [[1.5, 2.5], [3.5, 4.5]]),  # Metal array
        ("pot", 1, [5.0, 6.0]),  # Potentials
        ("age", 4, [17.0, 42.0]),  # Star age
    ],
)
def test_gadget_read(mock_gadget_file, reading, ptype, expected):
    # Mock header and data based on reading type
    h = MockHeader(
        npartThisFile=np.array([2, 2, 0, 0, 2, 0]),
        reading=reading,
        flag_metals=2,
        BLOCKORDER={
            "pos": [-1],
            "vel": [-1],
            "pid": [-1],
            "mass": [-1],
            "u": [0],
            "tmax": [[0, 4]],
            "metallicity": [[0, 4]],
            "pot": [-1],
            "age": [4],
        },  # Simulate block order
    )

    # Create data with blocks
    pos_data = np.ones(np.sum(h.npartThisFile) * 3, dtype=np.float32)  # 6 particles

    vel_data = np.zeros(np.sum(h.npartThisFile) * 3, dtype=np.float32)  # 6 particles
    vel_data[6:12] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # Type1 particles

    pids = np.arange(np.sum(h.npartThisFile), dtype=np.uint32) + 10000
    mass = np.arange(1, np.sum(h.npartThisFile) + 1, dtype=np.uint32) * 100.0
    u = np.array([1.0, 2.0], dtype=np.float32)
    tmax = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    metallicity = (
        np.arange(np.sum(h.npartThisFile[[0, 4]]) * h.flag_metals, dtype=np.float32)
        + 1.5
    )

    pot_data = np.arange(np.sum(h.npartThisFile), dtype=np.float32) + 3  # 6 particles
    age = np.array([17.0, 42.0], dtype=np.float32)

    data_list = [pos_data, vel_data, pids, mass, u, tmax, metallicity, pot_data, age]
    data_dtypes = [
        np.float32,
        np.float32,
        np.uint32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ]

    blocks = [
        np.array(b_data, dtype=d_dtype)
        for (b_data, d_dtype) in zip(data_list, data_dtypes)
    ]

    with mock_gadget_file(blocks, dtypes=data_dtypes) as f:
        result = gadget_read(f, h, ptype, reading)

    assert np.allclose(result, np.array(expected))


def test_gadget_read_warning(capsys, mock_gadget_file):
    """Test warning when requesting gas property for non-gas particle"""
    h = MockHeader(npartThisFile=[2, 2, 0, 0, 0, 0], reading="u", BLOCKORDER={"u": [0]})
    data = np.array([1.0, 2.0], dtype=np.float32)

    with mock_gadget_file(data) as f:
        result = gadget_read(f, h, 1, "u")

    captured = capsys.readouterr()
    assert (
        "WARNING!! you requested ParticleType1 for u, returning GAS instead"
        in captured.out
    )
    assert np.allclose(result, np.array([1.0, 2.0]))


def test_gadget_read_unknown(capsys):
    """Test handling of unknown reading type"""
    h = MockHeader(npartThisFile=[0, 0, 0, 0, 0, 0], reading="unknown", BLOCKORDER={})
    # Empty file
    with tempfile.NamedTemporaryFile() as f:
        result = gadget_read(f, h, 0, "unknown")

    captured = capsys.readouterr()
    assert "no clue what to read =(" in captured.out
    assert result.size == 0
