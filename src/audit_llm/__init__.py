# SPDX-FileCopyrightText: 2024 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

import importlib.metadata

try:
    import vllm  # type: ignore[import-not-found]

    __version__ = importlib.metadata.version("audit-llm")
except ImportError:
    # vllm not installed (post-processing environment) — use generic package name
    try:
        __version__ = importlib.metadata.version("audit-llm")
    except importlib.metadata.PackageNotFoundError:
        __version__ = "0.0.0-dev"
