"""Discovery of externally installed Serena tool packages."""

from importlib.metadata import entry_points
from types import ModuleType

from sensai.util import logging

from serena.tools.tools_base import tool_packages

log = logging.getLogger(__name__)

EXTERNAL_TOOL_ENTRY_POINT_GROUP = "serena.tools"


def load_external_tool_packages() -> None:
    """Import tool packages registered through the ``serena.tools`` entry-point group.

    Entry points must resolve to Python modules containing concrete ``Tool`` subclasses.
    Loading the module makes those subclasses discoverable through Serena's existing
    subclass scan; adding the module name to ``tool_packages`` allows the normal
    ``ToolRegistry`` validation and registration path to accept them.
    """
    for entry_point in entry_points(group=EXTERNAL_TOOL_ENTRY_POINT_GROUP):
        tool_package = entry_point.load()
        if not isinstance(tool_package, ModuleType):
            raise TypeError(
                f"Serena tool entry point '{entry_point.name}' must resolve to a Python module, "
                f"got {type(tool_package).__name__}."
            )

        package_name = tool_package.__name__
        if package_name not in tool_packages:
            tool_packages.append(package_name)
            log.info("Loaded external Serena tool package '%s' from entry point '%s'.", package_name, entry_point.name)
