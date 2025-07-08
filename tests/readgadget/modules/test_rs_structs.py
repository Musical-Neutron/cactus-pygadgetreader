#!/usr/bin/env python3

import pytest

from readgadget.modules.rs_structs import (
    getRSformat,
    halogalaxystruct1,
    halostruct1,
    halostruct2,
)


# Mock object class to simulate input
class MockObj:
    def __init__(self, galaxies, format_revision, debug=False):
        self.galaxies = galaxies
        self.format_revision = format_revision
        self.debug = debug


# Test cases for halo structures
def test_halo_revision1():
    obj = MockObj(galaxies=0, format_revision=1)
    assert getRSformat(obj) == halostruct1


def test_halo_revision2():
    obj = MockObj(galaxies=0, format_revision=2)
    assert getRSformat(obj) == halostruct2


# Test cases for galaxy structures
def test_galaxy_revision1():
    obj = MockObj(galaxies=1, format_revision=1)
    assert getRSformat(obj) == halogalaxystruct1


# Test cases for outdated revisions
def test_halo_outdated_revision(capsys):
    obj = MockObj(galaxies=0, format_revision=0)
    with pytest.raises(SystemExit):
        getRSformat(obj)
    captured = capsys.readouterr()
    assert "OUTDATED ROCKSTAR" in captured.out


def test_galaxy_outdated_revision(capsys):
    obj = MockObj(galaxies=1, format_revision=0)
    with pytest.raises(SystemExit):
        getRSformat(obj)
    captured = capsys.readouterr()
    assert "OUTDATED ROCKSTAR-GALAXIES" in captured.out


# Test cases for unsupported future revisions
def test_halo_unsupported_revision(capsys):
    obj = MockObj(galaxies=0, format_revision=3)
    with pytest.raises(SystemExit):
        getRSformat(obj)
    captured = capsys.readouterr()
    assert "found HALO_FORMAT_REVISION=3" in captured.out


def test_galaxy_unsupported_revision(capsys):
    obj = MockObj(galaxies=1, format_revision=2)
    with pytest.raises(SystemExit):
        getRSformat(obj)
    captured = capsys.readouterr()
    assert "found HALO_FORMAT_REVISION=2" in captured.out
