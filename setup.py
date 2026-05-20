# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the hyperleg_rl package."""

import os
import re

from setuptools import find_packages, setup

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML = os.path.join(PROJECT_ROOT, "source", "hyperleg_rl", "config", "extension.toml")
with open(EXTENSION_TOML) as f:
    content = f.read()


def _extract(field: str, default: str) -> str:
    match = re.search(rf'{field}\s*=\s*"([^"]*)"', content)
    return match.group(1) if match and match.group(1) else default


_version = _extract("version", "0.1.0")
_description = _extract("description", "HyperLeg biped locomotion RL with PhysX physics")
_repo = _extract("repository", "")

INSTALL_REQUIRES = [
    "psutil",
]

setup(
    name="hyperleg_rl",
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    author="IRIM",
    maintainer="IRIM",
    url=_repo,
    version=_version,
    description=_description,
    keywords=["extension", "isaaclab", "physx", "hyperleg", "locomotion"],
    install_requires=INSTALL_REQUIRES,
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,
)
