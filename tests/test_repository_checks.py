import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import check_repository as checks


class NotebookLinkTests(unittest.TestCase):
    def check_link(self, target):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notebook = root / 'notebooks/demo.ipynb'
            notebook.parent.mkdir()
            (root / 'evidence').mkdir()
            (root / 'evidence/result.json').write_text('{}', encoding='utf-8')
            notebook.write_text(json.dumps({'cells': [{'cell_type': 'markdown', 'source': [f'[result]({target})']}]}), encoding='utf-8')
            with patch.object(checks, 'ROOT', root), patch.object(checks, 'NOTEBOOK', notebook):
                return checks.check_notebook_links(False)

    def test_rejects_path_relative_to_wrong_directory(self):
        self.assertTrue(self.check_link('evidence/result.json').errors)

    def test_accepts_path_relative_to_notebook(self):
        self.assertFalse(self.check_link('../evidence/result.json').errors)

    def test_accepts_absolute_public_reference(self):
        self.assertFalse(self.check_link('https://github.com/example/lab/blob/main/evidence/result.json').errors)


if __name__ == '__main__':
    unittest.main()
