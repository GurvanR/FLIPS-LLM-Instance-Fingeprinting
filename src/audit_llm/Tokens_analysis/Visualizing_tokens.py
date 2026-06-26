"""Token sequence visualization using PIL/ImageDraw."""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont


def tokens_highlighter_image(
    token_ids_seqs: List[List[int]],
    token_vocab: Dict[str, int],
    saving_folder_path: Path = Path(),
    output_prefix: str = "tokens_visualization",
    font_path: Optional[str] = None,
    font_size: int = 20,
    padding: int = 10,
    max_image_width: int = 1200,
    max_image_height: int = 1000,
) -> None:
    """Plot token sequences with a color per token, saving multi-page PDFs."""
    id_to_token = {v: k for k, v in token_vocab.items()}

    def id_to_color(tok_id: int) -> tuple:
        random.seed(tok_id)
        return tuple(random.randint(100, 255) for _ in range(3))

    try:
        font = ImageFont.truetype(font_path or "arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    current_img_index = 1
    y = padding
    img = Image.new("RGB", (max_image_width, max_image_height), "white")
    draw = ImageDraw.Draw(img)

    def save_image():
        nonlocal current_img_index, img, draw, y
        filename = os.path.join(saving_folder_path, f"{output_prefix}_{current_img_index}.pdf")
        img.save(filename)
        current_img_index += 1
        y = padding
        img = Image.new("RGB", (max_image_width, max_image_height), "white")
        draw = ImageDraw.Draw(img)

    for seq_idx, seq in enumerate(token_ids_seqs):
        label = f"Sequence {seq_idx + 1}"
        draw.text((padding, y), label, fill="gray", font=font)
        y += font_size + padding // 2
        draw.line([(padding, y), (max_image_width - padding, y)], fill="gray", width=1)
        y += padding

        x = padding
        line_height = 0

        for tok_id in seq:
            token = id_to_token.get(tok_id, f"<UNK:{tok_id}>")
            bbox = font.getbbox(token)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            box_width = w + padding * 2
            box_height = h + padding * 2

            if x + box_width > max_image_width:
                x = padding
                y += line_height + padding
                line_height = 0
                if y + box_height > max_image_height:
                    save_image()

            color = id_to_color(tok_id)
            draw.rectangle([x, y, x + box_width, y + box_height], fill=color)
            draw.text((x + padding, y + padding), token, fill="black", font=font)
            x += box_width + padding
            line_height = max(line_height, box_height)

        y += line_height + padding * 2
        if y + font_size + padding * 3 > max_image_height:
            save_image()

    if y > padding:
        save_image()
