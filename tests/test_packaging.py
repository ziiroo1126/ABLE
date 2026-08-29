import subprocess
import sys
import unittest
from pathlib import Path


class PackagingTests(unittest.TestCase):
    def test_src_package_defers_model_stack_until_requested(self):
        code = """
import sys
import src
assert "src.calculator.able" not in sys.modules
assert "captum" not in sys.modules
from src import RunnerABLE
assert RunnerABLE.__name__ == "RunnerABLE"
"""

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_english_installation_requires_a_cloned_editable_checkout(self):
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertIn(
            "git clone https://github.com/ziiroo1126/ABLE.git", readme
        )
        self.assertIn("python -m pip install -e .", readme)
        self.assertIn("`python -m pip install .` is not supported", readme)

    def test_chinese_readme_and_project_page_share_the_installation_boundary(self):
        chinese_readme = Path("README_zh-CN.md").read_text(encoding="utf-8")
        project_page = Path("docs/index.html").read_text(encoding="utf-8")

        self.assertIn(
            "git clone https://github.com/ziiroo1126/ABLE.git",
            chinese_readme,
        )
        self.assertIn("不支持 `python -m pip install .`", chinese_readme)
        self.assertIn(
            "git clone https://github.com/ziiroo1126/ABLE.git",
            project_page,
        )
        self.assertIn("editable installation", project_page)

    def test_distribution_exposes_the_three_public_commands(self):
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('[project.scripts]', pyproject)
        self.assertIn('able-calculate = "src.calculate_able:main"', pyproject)
        self.assertIn(
            'able-convert = "src.token_to_word_attribution:main"', pyproject
        )
        self.assertIn(
            'able-project = "src.process_able_features:main"', pyproject
        )


if __name__ == "__main__":
    unittest.main()
