from types import ModuleType

import pytest

from serena.tools import external_tool_loader
from serena.tools.tools_base import tool_packages


class FakeEntryPoint:
    def __init__(self, name: str, loaded_object: object):
        self.name = name
        self._loaded_object = loaded_object

    def load(self) -> object:
        return self._loaded_object


def test_load_external_tool_packages_registers_module_once(monkeypatch: pytest.MonkeyPatch) -> None:
    original_tool_packages = list(tool_packages)
    external_module = ModuleType("example_serena_tools")
    entry_point = FakeEntryPoint("example", external_module)
    requested_groups: list[str] = []

    def fake_entry_points(*, group: str) -> list[FakeEntryPoint]:
        requested_groups.append(group)
        return [entry_point]

    monkeypatch.setattr(external_tool_loader, "entry_points", fake_entry_points)

    try:
        external_tool_loader.load_external_tool_packages()
        external_tool_loader.load_external_tool_packages()

        assert requested_groups == ["serena.tools", "serena.tools"]
        assert tool_packages.count("example_serena_tools") == 1
    finally:
        tool_packages[:] = original_tool_packages


def test_load_external_tool_packages_rejects_non_module_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_point = FakeEntryPoint("invalid", object())
    monkeypatch.setattr(external_tool_loader, "entry_points", lambda *, group: [entry_point])

    with pytest.raises(TypeError, match="must resolve to a Python module"):
        external_tool_loader.load_external_tool_packages()
