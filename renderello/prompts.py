"""Prompt assembly. Every generative call gets the same hard product-fidelity
guard plus the template's environment fragments plus an optional owner
instruction. The guard is non-negotiable per the problem statement: change only
scene/lighting/context, never the product.

The full owner instruction is retained. Model-specific encoders must handle
long text explicitly; including text does not guarantee model obedience.
"""
from __future__ import annotations

from .schemas import EditRequest

FIDELITY_GUARD = (
    "Keep the exact product unchanged: identical shape, packaging, color, logos, "
    "label text and typography, proportions and material. Do not redraw, restyle, "
    "retext or relabel the product. Change only the background, surface, lighting "
    "and scene context around it."
)

NEGATIVE_PROMPT = (
    "altered logo, changed label text, invented text, misspelled brand, deformed "
    "packaging, warped bottle, wrong product, extra products, duplicated product, "
    "melted edges, plastic-looking material change, watermark, frame, border"
)

MAX_INSTRUCTION_CHARS = 16000  # explicit application budget, never silent truncation


def build_prompt(request: EditRequest) -> str:
    """Assemble the final edit prompt for a generative route."""
    t = request.template
    parts = [
        FIDELITY_GUARD,
        f"Place the product in this scene: {t.scene}.",
        f"Lighting: {t.lighting}.",
        f"Camera: {t.camera}.",
        f"Mood: {t.mood}.",
        f"Background: {t.background_hint}.",
        f"Shadow and contact: {t.shadow}.",
    ]
    instruction = validate_instruction(request.custom_instruction).strip()
    if instruction:
        parts.append(f"Additional scene instruction (environment only): "
                     f"{instruction}.")
    parts.append("Photorealistic commercial product photograph.")
    return " ".join(parts)


def build_negative_prompt() -> str:
    return NEGATIVE_PROMPT


def build_harmonize_prompt(request: EditRequest) -> str:
    """Route 3: gentler pass over an already-correct composite. The product is
    already pixel-correct; ask only for light/shadow blending."""
    t = request.template
    instruction = validate_instruction(request.custom_instruction).strip()
    extra = f" Additional scene instruction (environment only): {instruction}." if instruction else ""
    return (
        f"{FIDELITY_GUARD} This is already the correct product cutout on a new "
        f"background. Only harmonize lighting, shadow and color grading so the "
        f"product sits naturally in the scene ({t.lighting}; {t.scene}). "
        "Do not move, resize, rotate or repaint the product." + extra
    )


def validate_instruction(text: str) -> str:
    """Reject instructions that ask to change the product itself - the problem
    statement forbids it, so fail loudly instead of quietly corrupting labels."""
    if not isinstance(text, str):
        raise ValueError("custom instruction must be text")
    if len(text) > MAX_INSTRUCTION_CHARS:
        raise ValueError(f"custom instruction exceeds {MAX_INSTRUCTION_CHARS} characters; no truncation performed")
    banned = ("change the label", "edit the logo", "remove the text", "rebrand",
              "different product", "swap the product", "redesign the packaging")
    low = text.lower()
    for b in banned:
        if b in low:
            raise ValueError(f"custom instruction '{b}' would alter the product; "
                             "only scene/lighting/context instructions are allowed")
    return text
