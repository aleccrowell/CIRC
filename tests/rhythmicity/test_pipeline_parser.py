"""Tests for the standalone `circ rhythm-calcp` pipeline argument parser."""

import os

from circ.rhythmicity import pipeline


class TestParserRefFileDefaults:
    """#41: default ref-file paths must resolve to files that actually exist
    in circ/rhythmicity/ref_files/, regardless of the current directory."""

    def test_period_default_exists(self):
        ns = pipeline.__create_parser__().parse_args([])
        assert os.path.isfile(ns.period)

    def test_phase_default_exists(self):
        ns = pipeline.__create_parser__().parse_args([])
        assert os.path.isfile(ns.phase)

    def test_width_default_exists(self):
        ns = pipeline.__create_parser__().parse_args([])
        assert os.path.isfile(ns.width)

    def test_defaults_live_in_ref_dir(self):
        ns = pipeline.__create_parser__().parse_args([])
        for path in (ns.period, ns.phase, ns.width):
            assert os.path.dirname(path) == pipeline._REF_DIR
