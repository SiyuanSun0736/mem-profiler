#!/usr/bin/env python3

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from category_schema import build_category_feature_map


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    train_variant: bool
    category_feature_map: OrderedDict[str, list[str]] | None = None
    reference_model_path: str | None = None
    reference_eval_path: str | None = None


def build_ablation_variants(non_time_cols: list[str]) -> list[AblationVariant]:
    full_map = build_category_feature_map(non_time_cols)

    coarse_2way = OrderedDict(
        [
            (
                "execution",
                list(full_map["core"]) + list(full_map["phase"]),
            ),
            (
                "memory",
                list(full_map["cache"]) + list(full_map["tlb"]) + list(full_map["fault"]) + list(full_map["mm"]),
            ),
        ]
    )

    no_mm_phase_4way = OrderedDict(
        [
            ("core", list(full_map["core"])),
            ("cache", list(full_map["cache"])),
            ("tlb", list(full_map["tlb"])),
            ("fault", list(full_map["fault"])),
        ]
    )

    summary_only: OrderedDict[str, list[str]] = OrderedDict()

    return [
        AblationVariant(
            name="summary_only",
            description="Only keep the global summary token; no semantic category tokens.",
            train_variant=True,
            category_feature_map=summary_only,
        ),
        AblationVariant(
            name="coarse_2way",
            description="Two extra tokens: execution (core+phase) and memory (cache+tlb+fault+mm).",
            train_variant=True,
            category_feature_map=coarse_2way,
        ),
        AblationVariant(
            name="no_mm_phase_4way",
            description="Keep core/cache/tlb/fault tokens and drop mm/phase tokens.",
            train_variant=True,
            category_feature_map=no_mm_phase_4way,
        ),
        AblationVariant(
            name="semantic_full_reference",
            description="Reference to the existing 6-way semantic category-token experiment.",
            train_variant=False,
            reference_model_path="train_set/category_token_transformer/model_category_token_transformer.pt",
            reference_eval_path="train_set/category_token_transformer/model_category_token_eval.json",
        ),
    ]


def build_schema_metadata(
    non_time_cols: list[str],
    category_feature_map: OrderedDict[str, list[str]],
) -> dict[str, object]:
    tokens: list[dict[str, object]] = [
        {
            "token": "summary",
            "description": "Global run-level summary projection over all non-time features.",
            "n_features": len(non_time_cols),
            "features": list(non_time_cols),
        }
    ]

    for name, cols in category_feature_map.items():
        tokens.append(
            {
                "token": name,
                "description": f"Ablation token '{name}'.",
                "n_features": len(cols),
                "features": list(cols),
            }
        )

    return {
        "include_summary_token": True,
        "n_total_features": len(non_time_cols),
        "n_category_tokens": len(category_feature_map),
        "n_tokens_per_program": len(tokens),
        "category_order": list(category_feature_map.keys()),
        "token_order": [token["token"] for token in tokens],
        "tokens": tokens,
    }