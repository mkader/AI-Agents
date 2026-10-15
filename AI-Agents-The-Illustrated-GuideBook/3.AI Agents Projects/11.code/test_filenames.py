import tempfile
import unittest
from pathlib import Path

from VibeStudio.filenames import documentation_path


class DocumentationPathTests(unittest.TestCase):
    def test_generated_titles_can_be_written_on_windows(self):
        titles = [
            "Configuration-driven patterns: YAML Agent\\Task Setup",
            "Agent/Task Setup",
            'Invalid < > : " / \\ | ? * characters',
            "CON",
            "nul.txt",
            "COM1",
            "Trailing dots... ",
            "Unicode \u2192 documentation",
            "A" * 400,
        ]
        with tempfile.TemporaryDirectory() as directory:
            docs_dir = Path(directory)
            for title in titles:
                with self.subTest(title=title):
                    path = documentation_path(docs_dir, title)
                    self.assertEqual(path.parent, docs_dir)
                    self.assertEqual(path.suffix, ".mdx")
                    path.write_text("Content \u2192", encoding="utf-8")
                    self.assertEqual(path.read_text(encoding="utf-8"), "Content \u2192")

    def test_reported_title(self):
        path = documentation_path(
            Path("docs"), "Configuration-driven patterns: YAML Agent\\Task Setup"
        )
        self.assertEqual(
            path.name, "configuration-driven_patterns_yaml_agent_task_setup.mdx"
        )

    def test_collisions_do_not_overwrite_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            docs_dir = Path(directory)
            original = documentation_path(docs_dir, "Agent/Task")
            original.write_text("Existing content", encoding="utf-8")
            duplicate = documentation_path(docs_dir, "Agent\\Task")
            self.assertEqual(duplicate.name, "agent_task_2.mdx")
            self.assertEqual(original.read_text(encoding="utf-8"), "Existing content")

    def test_unusable_title_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no usable filename"):
            documentation_path(Path("docs"), ' .:/\\?* ')


if __name__ == "__main__":
    unittest.main()
