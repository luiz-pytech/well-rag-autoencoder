"""Renderização e OCR do cabeçalho de PDFs de perfil composto.

Os PDFs de perfil composto (ANP/Petrobras) não têm texto extraível diretamente:
o "texto" é desenhado como formas vetoriais (sem fontes reais embutidas), então
`page.get_text()` sempre retorna vazio. A extração exige renderizar a página como
imagem e rodar OCR sobre ela.

A grande maioria dos poços usa um template **portrait** (página alta e estreita),
com o cabeçalho técnico (nome do poço, bacia, localização, profundidades, datas,
perfis corridos) ocupando a faixa superior. Uma minoria usa um template
**landscape** (página larga e baixa) em que o mesmo cabeçalho aparece desenhado
girado 90°, comprimido numa faixa vertical estreita no canto superior esquerdo —
detectamos isso pela proporção largura/altura da página e ajustamos o recorte e a
rotação de acordo.

O zoom de renderização (`RENDER_ZOOM`) importa bastante para a qualidade do OCR:
em zoom baixo (2x), algumas letras iniciais de rótulos como "BACIA" saem
ilegíveis; 3x-4x resolve a maioria dos casos observados.
"""

from __future__ import annotations

import os
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_DIR = Path(__file__).resolve().parents[2] / ".tessdata"

HEADER_HEIGHT_POINTS = 1200  # faixa superior da página (template portrait) que contém o cabeçalho
HEADER_WIDTH_POINTS = 1200  # faixa esquerda da página (template landscape) que contém o cabeçalho
RENDER_ZOOM = 4.0  # fator de ampliação antes do OCR
LANDSCAPE_ASPECT_THRESHOLD = 2.0  # largura/altura acima disso => template landscape rotacionado


def _configure_tesseract() -> None:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)


def render_header(
    pdf_path: str | Path,
    height_points: float = HEADER_HEIGHT_POINTS,
    width_points: float = HEADER_WIDTH_POINTS,
    zoom: float = RENDER_ZOOM,
) -> Image.Image:
    """Renderiza a região do cabeçalho da primeira página como imagem PIL.

    Detecta automaticamente o template (portrait: recorta faixa superior;
    landscape: recorta faixa esquerda e rotaciona 90° para a orientação de leitura).
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[0]
        rect = page.rect
        is_landscape = rect.width / rect.height > LANDSCAPE_ASPECT_THRESHOLD

        mat = fitz.Matrix(zoom, zoom)
        if is_landscape:
            clip = fitz.Rect(0, 0, min(width_points, rect.width), rect.height)
            pix = page.get_pixmap(matrix=mat, clip=clip)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            return image.rotate(-90, expand=True)

        clip = fitz.Rect(0, 0, rect.width, min(height_points, rect.height))
        pix = page.get_pixmap(matrix=mat, clip=clip)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def ocr_header(pdf_path: str | Path, lang: str = "por") -> str:
    """Renderiza o cabeçalho do PDF e roda OCR, retornando o texto extraído."""
    _configure_tesseract()
    image = render_header(pdf_path)
    return pytesseract.image_to_string(image, lang=lang)
