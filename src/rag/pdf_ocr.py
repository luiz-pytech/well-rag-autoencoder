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

Motor de OCR: **GLM-OCR** (modelo multimodal 0.9B, via Ollama local) é o padrão —
ele entende a estrutura de formulário do cabeçalho (rótulo + valor lado a lado) e
supera claramente o Tesseract em campos como nome do poço, bloco e área/campo, que
o Tesseract (OCR caráctere-a-caráctere, sem noção de layout) praticamente não
conseguia capturar. Tesseract continua disponível como `ocr_header_tesseract` para
comparação ou como alternativa se o Ollama não estiver disponível.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import fitz
import requests
from PIL import Image

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_DIR = Path(__file__).resolve().parents[2] / ".tessdata"

HEADER_HEIGHT_POINTS = 1200  # faixa superior da página (template portrait) que contém o cabeçalho
HEADER_WIDTH_POINTS = 1200  # faixa esquerda da página (template landscape) que contém o cabeçalho
RENDER_ZOOM = 4.0  # fator de ampliação antes do OCR
LANDSCAPE_ASPECT_THRESHOLD = 2.0  # largura/altura acima disso => template landscape rotacionado

OLLAMA_URL = "http://localhost:11434/api/generate"
GLM_OCR_MODEL = "glm-ocr"
GLM_MAX_IMAGE_WIDTH = 1400  # reduz a imagem antes de enviar: acelera bastante a inferência em CPU
# 800 tokens cortava a resposta antes da seção "PERFIS CORRIDOS" (que vem depois de
# uma seção intermediária extensa de códigos de litologia) — 1600 dá margem para
# chegar até lá sem estourar o tempo de inferência em CPU.
GLM_NUM_PREDICT = 1600
GLM_NUM_CTX = 8192

GLM_PROMPT = (
    "Este é o cabeçalho de um perfil composto de poço de petróleo (documento técnico "
    "ANP/Petrobras). Extraia, na ordem em que aparecem, os campos e valores visíveis no "
    "formulário: nome do poço, designação, bacia, bloco, área ou campo, profundidade de "
    "perfuração, profundidade de perfilagem, formação, operador, datum. Depois, ignore a "
    "legenda de códigos de litologias (não a transcreva) e vá direto para a seção "
    "'PERFIS CORRIDOS', listando cada sigla de curva com sua faixa de profundidade entre "
    "parênteses (ex.: GR (15.0 - 1362.4)). Liste cada campo no formato 'CAMPO: valor', um por linha."
)


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


def _image_to_base64(image: Image.Image, max_width: int = GLM_MAX_IMAGE_WIDTH) -> str:
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, int(image.height * ratio)))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _looks_like_prompt_echo(response_text: str) -> bool:
    """Detecta o modelo devolvendo o próprio prompt em vez de processar a imagem.

    Observado ocasionalmente com GLM-OCR: em vez de descrever o conteúdo da
    imagem, a resposta repete literalmente (às vezes várias vezes) o texto do
    prompt de entrada — um sinal de que o modelo não "olhou" a imagem. Detectamos
    isso checando se um trecho característico do prompt aparece na resposta.
    """
    marker = "Extraia, na ordem em que aparecem"
    return marker in response_text


def ocr_header(pdf_path: str | Path, timeout: float = 180.0, max_retries: int = 2) -> str:
    """Renderiza o cabeçalho do PDF e roda OCR via GLM-OCR (Ollama), retornando o texto.

    Requer o servidor Ollama local rodando com o modelo `glm-ocr` baixado
    (`ollama pull glm-ocr`). Se o modelo ecoar o próprio prompt (falha de geração
    observada ocasionalmente, não relacionada à qualidade da imagem), tenta de
    novo até `max_retries` vezes antes de desistir.
    """
    image = render_header(pdf_path)
    image_b64 = _image_to_base64(image)

    payload = {
        "model": GLM_OCR_MODEL,
        "prompt": GLM_PROMPT,
        "images": [image_b64],
        "stream": False,
        "options": {"num_predict": GLM_NUM_PREDICT, "num_ctx": GLM_NUM_CTX},
    }

    last_text = ""
    for attempt in range(max_retries + 1):
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        text = response.json().get("response", "")
        if not _looks_like_prompt_echo(text):
            return text
        last_text = text

    raise ValueError(f"GLM-OCR retornou eco do prompt após {max_retries + 1} tentativas: {pdf_path}")


def ocr_header_tesseract(pdf_path: str | Path, lang: str = "por") -> str:
    """Alternativa via Tesseract (mais rápido, menos preciso em campos de layout complexo)."""
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)
    image = render_header(pdf_path)
    return pytesseract.image_to_string(image, lang=lang)
