# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Task implementations for hyperleg_rl."""

from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = [".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
