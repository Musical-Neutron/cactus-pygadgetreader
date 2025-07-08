#!/usr/bin/env python3

import sys

import numpy as np
import pytest

from readgadget.modules.common import (
    BOLTZMANN,
    GAMMA,
    H_MASSFRAC,
    METALFACTOR,
    PROTONMASS,
    RecognizedOptions,
    gadgetPrinter,
    getTfactor,
    getTfactorNoNe,
    initUnits,
    pollHeaderOptions,
    pollOptions,
)

# Mock constants and dictionaries required by the module
headerTypes = {
    "npartThisFile": "npartThisFile",
    "npartTotal": "npartTotal",
    "ngas": "ngas",
    "ndm": "ndm",
}
dataTypes = {"velocity": "vel", "position": "pos"}
pTypes = {"gas": 0, "stars": 1}
dataNames = {"rho": "density", "vel": "velocity"}
dataUnits = {"rho": "g/cm³", "vel": "cm/s"}
dataDefaultUnits = {"rho": "code units", "vel": "code units"}
pNames = {0: "Gas", 1: "Stars"}


# Mock header class for testing
class MockHeader:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# Fixture to patch module-level dictionaries
@pytest.fixture(autouse=True)
def patch_names(monkeypatch):
    monkeypatch.setattr("readgadget.modules.common.dataTypes", dataTypes)
    monkeypatch.setattr("readgadget.modules.common.pTypes", pTypes)
    monkeypatch.setattr("readgadget.modules.common.dataNames", dataNames)
    monkeypatch.setattr("readgadget.modules.common.dataUnits", dataUnits)
    monkeypatch.setattr("readgadget.modules.common.dataDefaultUnits", dataDefaultUnits)
    monkeypatch.setattr("readgadget.modules.common.pNames", pNames)


# Test for pollOptions
@pytest.mark.parametrize(
    "dtypekey, dtypeval",
    [(k, v) for k, v in dataTypes.items()],
)
def test_pollOptions_valid_dtypes(capsys, dtypekey, dtypeval):
    h = MockHeader(fileType="hdf5")
    KWARGS = {"units": True, "hdf5": True}
    d, p = pollOptions(h, KWARGS, dtypekey, "gas")
    assert d == dtypeval
    assert p == 0
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out


@pytest.mark.parametrize(
    "ptypekey, ptypeval",
    [(k, v) for k, v in pTypes.items()],
)
def test_pollOptions_valid_ptypes(capsys, ptypekey, ptypeval):
    h = MockHeader(fileType="hdf5")
    KWARGS = {"units": True, "hdf5": True}
    d, p = pollOptions(h, KWARGS, "vel", ptypekey)
    assert d == "vel"
    assert p == ptypeval
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out


def test_pollOptions_unrecognized_option(capsys):
    h = MockHeader(fileType="hdf5")
    KWARGS = {"invalid_opt": True}
    d, p = pollOptions(h, KWARGS, "vel", "gas")
    captured = capsys.readouterr()
    assert "WARNING!! option not recognized: invalid_opt" in captured.out


@pytest.mark.parametrize(
    "ftype, kwargs, data, ptype",
    [
        ("invalid_ftype", {"units": True, "hdf5": True}, "vel", "gas"),
        ("hdf5", {"units": True, "hdf5": True}, "vel", "invalid_ptype"),
    ],
)
def test_pollOptions_invalid_sysexit(ftype, kwargs, data, ptype):
    h = MockHeader(fileType=ftype)
    with pytest.raises(SystemExit):
        pollOptions(h, kwargs, data, ptype)


# Test for pollHeaderOptions
@pytest.mark.parametrize(
    "htypename",
    [(k) for k in headerTypes.keys()],
)
def test_pollHeaderOptions_valid(htypename):
    # Should not exit if data is valid
    pollHeaderOptions(None, htypename)


def test_pollHeaderOptions_invalid(capsys):
    with pytest.raises(SystemExit):
        pollHeaderOptions(None, "invalid_data")
    captured = capsys.readouterr()
    assert "ERROR! invalid_data not a recognized header value" in captured.out


# Test for initUnits
def test_initUnits_rho_cosmological():
    h = MockHeader(
        units=True,
        fileType="gadget2",
        reading="rho",
        boxsize=100.0,
        OmegaLambda=0.7,
        redshift=1.0,
        UnitMass_in_g=1.989e43,
        UnitLength_in_cm=3.085678e21,
    )
    initUnits(h)
    expected_convert = (1.0 + 1.0) ** 3 * 1.989e43 / (3.085678e21) ** 3
    assert h.convert == pytest.approx(expected_convert)


def test_initUnits_vel_cosmological():
    h = MockHeader(
        units=True, fileType="gadget2", reading="vel", boxsize=100.0, Ol=0.7, time=0.5
    )
    initUnits(h)
    assert h.convert == pytest.approx(np.sqrt(0.5))


def test_initUnits_u_sets_energy():
    h = MockHeader(
        units=True,
        fileType="gadget2",
        reading="u",
        UnitLength_in_cm=3.085678e21,
        UnitVelocity_in_cm_per_s=1.0e5,
        UnitMass_in_g=1.989e43,
    )
    initUnits(h)
    assert hasattr(h, "UnitTime_in_s")
    assert hasattr(h, "UnitEnergy_in_cgs")
    expected_time = 3.085678e21 / 1.0e5
    expected_energy = 1.989e43 * (3.085678e21) ** 2 / expected_time**2
    assert h.UnitTime_in_s == pytest.approx(expected_time)
    assert h.UnitEnergy_in_cgs == pytest.approx(expected_energy)


# Test for getTfactor
def test_getTfactor():
    h = MockHeader(UnitEnergy_in_cgs=1.0, UnitMass_in_g=1.0)
    Ne = 0.0
    t_factor = getTfactor(Ne, h)
    mean_weight = 4.0 / (3.0 * H_MASSFRAC + 1.0 + 4.0 * H_MASSFRAC * Ne) * PROTONMASS
    expected = mean_weight / BOLTZMANN * (GAMMA - 1.0) * 1.0 / 1.0
    assert t_factor == pytest.approx(expected)


# Test for getTfactorNoNe
def test_getTfactorNoNe():
    t_factor = getTfactorNoNe()
    expected = (GAMMA - 1.0) * (PROTONMASS / BOLTZMANN) * 1.0e5**2
    assert t_factor == pytest.approx(expected)


# Test for gadgetPrinter
def test_gadgetPrinter_with_units(capsys):
    h = MockHeader(fileType="gadget2", units=True, suppress=False)
    gadgetPrinter(h, "rho", 0)
    captured = capsys.readouterr()
    assert "Returning Gas density g/cm³" in captured.out


def test_gadgetPrinter_suppressed(capsys):
    h = MockHeader(fileType="gadget2", units=False, suppress=True)
    gadgetPrinter(h, "vel", 0)
    captured = capsys.readouterr()
    assert captured.out == ""
