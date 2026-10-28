"""Rule-based candidate issue tags for Amazon Beauty review text."""

from __future__ import annotations

import re

import numpy as np


ISSUE_PATTERNS = {
    "allergic_reaction": r"\b(allergic reaction|allergy|allergic|hives)\b",
    "breakouts": r"\b(breakouts?|acne|pimples?)\b",
    "broken_packaging": r"\b(broken|cracked|shattered)\b.{0,30}\b(container|bottle|jar|package|packaging)\b|\b(broken|cracked|shattered)\b",
    "color_mismatch": r"\b(color|colour)\b.{0,25}\b(mismatch|different|wrong|off)\b",
    "counterfeit_suspected": r"\b(fake|counterfeit|imitation|not authentic|not genuine)\b",
    "damaged_packaging": r"\b(damaged|dented|crushed|torn)\b.{0,30}\b(package|packaging|box|container)\b",
    "dryness": r"\b(dry|drying|dried out|dried my skin)\b",
    "expired_product": r"\b(expired|past its expiration|out of date|old stock)\b",
    "formula_change": r"\b(formula|formulation|ingredient)\b.{0,35}\b(changed|different|reformulated)\b|\b(new formula|reformulated)\b",
    "greasy_texture": r"\b(greasy|oily residue|too oily|oil[y]? feel)\b",
    "hair_loss": r"\b(hair loss|hair fell out|hair falling out|caused shedding)\b",
    "hard_to_open": r"\b(hard|difficult|impossible)\b.{0,20}\b(open|remove the cap|unscrew)\b",
    "ineffective": r"\b(did not work|does not work|doesn't work|no effect|not effective|ineffective|did nothing)\b",
    "leakage": r"\b(leak|leaking|leaked|spilled|spill during shipping)\b",
    "messy_application": r"\b(messy|hard to apply|difficult to apply|drips everywhere)\b",
    "missing_item": r"\b(missing|did not receive|didn't receive|not included)\b.{0,30}\b(item|product|part|piece|order)\b|\b(empty box|nothing inside)\b",
    "poor_absorption": r"\b(does not absorb|doesn't absorb|won't absorb|sits on my skin|does not sink in)\b",
    "poor_quality": r"\b(poor quality|low quality|cheaply made|fell apart|broke after|stopped working)\b",
    "poor_value": r"\b(not worth|poor value|waste of money|overpriced|too expensive|pricey for)\b",
    "pump_dispenser_issue": r"\b(pump|dispenser|spray nozzle)\b.{0,35}\b(broken|stuck|jammed|not working|doesn't work|does not work|clogged)\b|\b(broken|stuck|jammed|clogged)\b.{0,25}\b(pump|dispenser|nozzle)\b",
    "rash_or_irritation": r"\b(irritat(?:ion|ed|ing)|rash|burn(?:ing|ed)?|stinging|itch(?:y|ing))\b",
    "shade_mismatch": r"\b(shade|foundation|concealer)\b.{0,30}\b(mismatch|wrong|too (?:light|dark)|not match)\b",
    "shipping_delay": r"\b(late|delayed|delay|took too long)\b.{0,25}\b(delivery|shipping|arriv(?:e|al)|package)\b|\b(delivery|shipping)\b.{0,25}\b(late|delayed|delay)\b",
    "short_lasting": r"\b(did not last|doesn't last|does not last|wore off|faded quickly|short lasting|short-lived)\b",
    "small_quantity": r"\b(too little|very small|tiny amount|small amount|less than expected|not much product)\b",
    "staining": r"\b(stain(?:ed|ing)?|left a mark|discolored my)\b",
    "sticky_texture": r"\b(sticky|tacky|too thick|thick and sticky)\b",
    "strong_fragrance": r"\b(too strong|overpowering|strong smell|strong scent|heavily fragranced)\b",
    "unpleasant_fragrance": r"\b(bad smell|unpleasant smell|unpleasant scent|smells awful|smells bad|off-putting smell)\b",
    "unwanted_side_effect": r"\b(side effect|adverse reaction|made my skin worse)\b",
    "wrong_item": r"\b(wrong item|wrong product|incorrect item|received the wrong|not what i ordered)\b",
    "wrong_product_description": r"\b(not as described|description is misleading|misleading description|different from the listing)\b",
}

TAG_DESCRIPTIONS = {
    "allergic_reaction": "Allergic response or hives",
    "breakouts": "Acne or breakouts",
    "broken_packaging": "Broken product container or package",
    "color_mismatch": "Product color differs from expectation",
    "counterfeit_suspected": "Reviewer suspects an inauthentic product",
    "damaged_packaging": "Package arrived damaged",
    "dryness": "Product caused or worsened dryness",
    "expired_product": "Product was expired or out of date",
    "formula_change": "Reviewer reports a formula or ingredient change",
    "greasy_texture": "Greasy or oily residue",
    "hair_loss": "Reviewer reports hair loss or shedding",
    "hard_to_open": "Packaging is difficult to open",
    "ineffective": "Product did not produce the expected effect",
    "leakage": "Product leaked or spilled",
    "messy_application": "Application is messy or difficult",
    "missing_item": "Item or component was missing",
    "poor_absorption": "Product does not absorb well",
    "poor_quality": "General quality or durability complaint",
    "poor_value": "Price or value complaint",
    "pump_dispenser_issue": "Pump, dispenser, or nozzle problem",
    "rash_or_irritation": "Skin irritation, rash, itching, or burning",
    "shade_mismatch": "Makeup shade does not match",
    "shipping_delay": "Delivery arrived late",
    "short_lasting": "Product effect, wear, or scent does not last",
    "small_quantity": "Amount of product is smaller than expected",
    "staining": "Product caused unwanted staining",
    "sticky_texture": "Sticky or tacky texture",
    "strong_fragrance": "Fragrance is too strong",
    "unpleasant_fragrance": "Reviewer dislikes or reports an unpleasant smell",
    "unwanted_side_effect": "Other reported unwanted side effect",
    "wrong_item": "Incorrect item was delivered",
    "wrong_product_description": "Product differs from its description",
}

_COMBINED_PATTERN = re.compile(
    "|".join(f"(?P<{tag}>{pattern})" for tag, pattern in ISSUE_PATTERNS.items()),
    re.IGNORECASE,
)


def normalize_tags(value: object) -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [tag for tag in value if isinstance(tag, str)]
    return []


def has_issue_tags(value: object) -> bool:
    return bool(normalize_tags(value))


def tag_review(text: object) -> list[str]:
    """Return all issue-tag candidates found in review text."""
    if not isinstance(text, str) or not text.strip():
        return []

    tags = {
        tag
        for match in _COMBINED_PATTERN.finditer(text)
        for tag, value in match.groupdict().items()
        if value is not None
    }
    return sorted(tags)


def parse_tags(value: object) -> set[str]:
    """Parse pipe-separated annotation/tag values; 'none' means no issue."""
    if not isinstance(value, str) or not value.strip() or value.strip().lower() == "none":
        return set()
    tags = {tag.strip() for tag in value.split("|") if tag.strip()}
    unknown = tags.difference(ISSUE_PATTERNS)
    if unknown:
        raise ValueError(f"Unknown issue tags: {', '.join(sorted(unknown))}.")
    return tags
