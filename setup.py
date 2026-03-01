from setuptools import setup, find_packages
from pathlib import Path
import re

def get_version():
    """Read version from pyproject.toml"""
    pyproject_path = Path(__file__).parent / "pyproject.toml"
    with open(pyproject_path) as f:
        content = f.read()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not match:
        raise RuntimeError("Could not find version in pyproject.toml")
    return match.group(1)

setup(
    name="models",
    version=get_version(),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "typer>=0.24.1",
        "fireducks>=1.3.0",
        "rapidfuzz>=3.9.0",
        "requests>=2.32.4",
        "rich>=14.0.0",
        "textual>=0.58.0",
        "tomli>=2.0.0; python_version < '3.11'"
    ],
    entry_points={
        'console_scripts': [
            'models = models.main:app',
        ],
    },
    python_requires='>=3.11',
    include_package_data=True,
    zip_safe=False,
)
