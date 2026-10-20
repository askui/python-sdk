from importlib.metadata import version

import askui


def test_version_consistency() -> None:
    package_version = version("askui")
    module_version = askui.__version__
    assert package_version == module_version, (
        f"Version mismatch: package={package_version}, module={module_version}; "
        "Please run: pdm sync"
    )
