from pathlib import Path

from askui.tools.playwright.agent_os import _to_unique_path


class TestToUniquePath:
    def test_returns_path_unchanged_when_free(self, tmp_path: Path) -> None:
        target = tmp_path / "report.pdf"
        assert _to_unique_path(target) == target

    def test_appends_counter_when_path_exists(self, tmp_path: Path) -> None:
        target = tmp_path / "report.pdf"
        target.write_text("first", encoding="utf-8")
        assert _to_unique_path(target) == tmp_path / "report (1).pdf"

    def test_increments_counter_until_free(self, tmp_path: Path) -> None:
        (tmp_path / "report.pdf").write_text("a", encoding="utf-8")
        (tmp_path / "report (1).pdf").write_text("b", encoding="utf-8")
        (tmp_path / "report (2).pdf").write_text("c", encoding="utf-8")
        assert _to_unique_path(tmp_path / "report.pdf") == tmp_path / "report (3).pdf"

    def test_handles_name_without_suffix(self, tmp_path: Path) -> None:
        target = tmp_path / "archive"
        target.write_text("x", encoding="utf-8")
        assert _to_unique_path(target) == tmp_path / "archive (1)"
