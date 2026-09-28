import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
hubert = types.ModuleType("infer.hubert")
hubert.load_hubert_model = lambda *args, **kwargs: None
sys.modules["infer.hubert"] = hubert
from infer.vc.utils import get_index_path_from_model


class IndexPathTests(unittest.TestCase):
    def setUp(self):
        self._old_index = os.environ.get("index_root")
        self._old_outside = os.environ.get("outside_index_root")

    def tearDown(self):
        self._restore("index_root", self._old_index)
        self._restore("outside_index_root", self._old_outside)

    def _restore(self, key, value):
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    def test_short_name_does_not_take_a_longer_model(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "miku.index").write_bytes(b"x")
            os.environ["index_root"] = td
            os.environ["outside_index_root"] = ""
            self.assertEqual(get_index_path_from_model("mi.pth"), "")

    def test_version_suffix_still_matches(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "miku_v2.index").write_bytes(b"x")
            os.environ["index_root"] = td
            os.environ["outside_index_root"] = ""
            found = get_index_path_from_model("miku.pth")
            self.assertEqual(os.path.basename(found), "miku_v2.index")


if __name__ == "__main__":
    unittest.main()
