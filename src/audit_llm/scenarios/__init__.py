# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Declarative scenario layer for FLIPS.

A *scenario* (``config/scenarios/<name>.yaml``) selects which model × variation
instances the analysis layer materialises, over four axes — ``temperature``,
``system_prompt_idx``, ``quantization``, ``abliteration`` — with
cartesian-within-group / union-across-groups semantics. See
``config/scenarios/README.md`` for the YAML contract and the CSV invariant.
"""

from audit_llm.scenarios.schema import AXES, Scenario, VariationGroup
from audit_llm.scenarios.loader import load_scenario

__all__ = ["AXES", "Scenario", "VariationGroup", "load_scenario"]
