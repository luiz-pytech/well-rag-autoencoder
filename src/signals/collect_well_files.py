"""Coleta arquivos de perfil composto (PDF) e perfil convencional (DLIS) do NAS da ANP.

Estrutura esperada no NAS (uma pasta por poço, com várias subpastas por tipo de dado):

    <nas_root>/<well_id>/perfil_composto/*.pdf
    <nas_root>/<well_id>/perfil_convencional/*.dlis
    <nas_root>/<well_id>/<outras pastas: agp, amostradores_laterais, teste_de_formacao...>

Só nos interessam os arquivos dentro de `perfil_composto` e `perfil_convencional`
(identificados pelo NOME DA PASTA ANCESTRAL, não por regex no nome do arquivo — isso
evita pegar .dlis de teste_de_formacao_a_cabo, perfil_durante_a_perfuracao etc., que
não são o perfil convencional). O well_id é sempre o nome da subpasta imediatamente
abaixo de --nas-root, que já corresponde exatamente ao identificador do poço.

Destino:
    <out_dir>/gamma_ray/<well_id>/dlis/*.dlis        (perfil_convencional)
    <out_dir>/profiles_pdf/<well_id>/*.pdf           (perfil_composto)

Dentro de perfil_convencional pode haver múltiplos .dlis (um por ferramenta: AIT, CMR,
MSIP, FMI...); todos são copiados. A escolha de qual arquivo contém a curva Gamma Ray
é feita depois, na leitura (ver src/signals/gamma_ray_loader.py), que varre os frames
de cada .dlis à procura do canal GR.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PERFIL_COMPOSTO_DIR = "perfil_composto"
PERFIL_CONVENCIONAL_DIR = "perfil_convencional"


def find_well_id(path: Path, nas_root: Path) -> str | None:
    """well_id = nome da subpasta imediatamente abaixo de nas_root."""
    try:
        relative = path.relative_to(nas_root)
    except ValueError:
        return None

    if len(relative.parts) < 2:
        return None
    return relative.parts[0]


def find_ancestor_dir(path: Path, nas_root: Path, dir_name: str) -> bool:
    """Verifica se algum diretório entre nas_root e o arquivo se chama dir_name (case-insensitive)."""
    try:
        relative = path.relative_to(nas_root)
    except ValueError:
        return False

    return any(part.lower() == dir_name.lower() for part in relative.parts[:-1])


def classify_file(path: Path, nas_root: Path) -> str | None:
    """Retorna 'dlis', 'perfil_composto' ou None (arquivo fora do escopo do projeto)."""
    if path.suffix.lower() == ".dlis" and find_ancestor_dir(path, nas_root, PERFIL_CONVENCIONAL_DIR):
        return "dlis"

    if path.suffix.lower() == ".pdf" and find_ancestor_dir(path, nas_root, PERFIL_COMPOSTO_DIR):
        return "perfil_composto"

    return None


def target_path(file_type: str, well_id: str, filename: str, out_dir: Path) -> Path:
    if file_type == "dlis":
        return out_dir / "gamma_ray" / well_id / "dlis" / filename
    return out_dir / "profiles_pdf" / well_id / filename


def safe_copy(src: Path, dst: Path, dry_run: bool = False, skip_existing: bool = True) -> tuple[Path, bool]:
    """Copia arquivo sem sobrescrever.

    Se o destino exato já existir com o mesmo tamanho do arquivo de origem, considera
    que já foi copiado em uma execução anterior e pula (skip_existing=True) — permite
    retomar uma coleta interrompida sem duplicar nem recopiar tudo. Caso contrário
    (tamanho diferente ou skip_existing=False), copia com sufixo _1, _2... para não
    sobrescrever.

    Retorna (caminho_final, foi_copiado).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        if skip_existing and dst.stat().st_size == src.stat().st_size:
            return dst, False

        stem, suffix = dst.stem, dst.suffix
        i = 1
        while True:
            candidate = dst.parent / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                dst = candidate
                break
            i += 1

    if not dry_run:
        shutil.copy2(src, dst)

    return dst, True


def _enumerate_targets(nas_root: Path, out_dir: Path) -> tuple[list[tuple[Path, str, str, Path]], int]:
    """Varre o NAS (rápido, só metadados) e monta a lista de (origem, tipo, well_id, destino)."""
    targets = []
    skipped_no_well_id = 0

    for path in nas_root.rglob("*"):
        if not path.is_file():
            continue

        file_type = classify_file(path, nas_root)
        if file_type is None:
            continue

        well_id = find_well_id(path, nas_root)
        if well_id is None:
            skipped_no_well_id += 1
            print(f"[SKIP] não foi possível determinar well_id: {path}")
            continue

        dst = target_path(file_type, well_id, path.name, out_dir)
        targets.append((path, file_type, well_id, dst))

    return targets, skipped_no_well_id


def collect_files(
    nas_root: Path,
    out_dir: Path,
    dry_run: bool = False,
    manifest_path: Path | None = None,
    max_workers: int = 8,
):
    """Enumera os arquivos alvo e copia em paralelo (I/O de rede se beneficia de concorrência).

    A varredura do NAS é sequencial (rápida, só lê metadados de diretório), mas a
    cópia de fato roda em um ThreadPoolExecutor — copiar vários arquivos ao mesmo
    tempo tende a usar melhor a banda disponível do que um arquivo por vez.
    """
    manifest = []
    summary = defaultdict(lambda: {"dlis": 0, "perfil_composto": 0})
    n_copied = 0
    n_skipped_existing = 0

    targets, skipped_no_well_id = _enumerate_targets(nas_root, out_dir)
    print(f"{len(targets)} arquivo(s) no escopo (perfil_composto + perfil_convencional).\n")

    manifest_fh = None
    manifest_writer = None
    write_lock = threading.Lock()
    if manifest_path is not None and not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not manifest_path.exists()
        manifest_fh = manifest_path.open("a", newline="", encoding="utf-8")
        manifest_writer = csv.DictWriter(
            manifest_fh,
            fieldnames=["well_id", "file_type", "source_path", "destination_path", "size_bytes"],
        )
        if is_new:
            manifest_writer.writeheader()

    def _copy_one(item: tuple[Path, str, str, Path]):
        src, file_type, well_id, dst = item
        copied_to, was_copied = safe_copy(src, dst, dry_run=dry_run)
        row = {
            "well_id": well_id,
            "file_type": file_type,
            "source_path": str(src),
            "destination_path": str(copied_to),
            "size_bytes": src.stat().st_size,
        }
        return row, was_copied

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_copy_one, item): item for item in targets}
            for future in as_completed(futures):
                row, was_copied = future.result()

                summary[row["well_id"]][row["file_type"]] += 1
                if was_copied:
                    n_copied += 1
                else:
                    n_skipped_existing += 1

                with write_lock:
                    manifest.append(row)
                    if manifest_writer is not None and was_copied:
                        manifest_writer.writerow(row)
                        manifest_fh.flush()

                action = ("DRY-RUN" if dry_run else "COPIED") if was_copied else "SKIP (já existe)"
                print(f"[{action}] {row['file_type']}: {row['source_path']} -> {row['destination_path']}")
    finally:
        if manifest_fh is not None:
            manifest_fh.close()

    if skipped_no_well_id:
        print(f"\n{skipped_no_well_id} arquivo(s) ignorado(s) por falta de well_id detectável.")
    print(f"{n_copied} arquivo(s) copiado(s) nesta execução | {n_skipped_existing} já existiam e foram pulados.")

    return manifest, summary


def print_summary(summary):
    print("\nResumo por poço:")
    print("-" * 60)
    for well_id, counts in sorted(summary.items()):
        print(f"{well_id}: {counts['dlis']} DLIS | {counts['perfil_composto']} perfil composto")

    total_dlis = sum(c["dlis"] for c in summary.values())
    total_pdf = sum(c["perfil_composto"] for c in summary.values())
    print("-" * 60)
    print(f"Total: {len(summary)} poços | {total_dlis} DLIS | {total_pdf} perfil composto")


def main():
    parser = argparse.ArgumentParser(
        description="Coleta arquivos .dlis (perfil_convencional) e PDFs (perfil_composto) do NAS da ANP."
    )
    parser.add_argument("--nas-root", required=True, help="Caminho raiz do NAS (ex.: Z:\\BOF_12128-st\\POCO).")
    parser.add_argument("--out-dir", default="data/raw", help="Diretório de saída (data/raw por padrão).")
    parser.add_argument("--dry-run", action="store_true", help="Mostra o que seria copiado, mas não copia nada.")
    parser.add_argument(
        "--max-workers", type=int, default=8, help="Cópias simultâneas (padrão: 8). Ajuste conforme a rede."
    )
    args = parser.parse_args()

    nas_root = Path(args.nas_root)
    out_dir = Path(args.out_dir)

    if not nas_root.exists():
        raise FileNotFoundError(f"Caminho do NAS não encontrado: {nas_root}")

    manifest_path = out_dir / "manifest.csv"
    _, summary = collect_files(
        nas_root=nas_root,
        out_dir=out_dir,
        dry_run=args.dry_run,
        manifest_path=manifest_path,
        max_workers=args.max_workers,
    )

    if not args.dry_run:
        print(f"\nManifest atualizado em: {manifest_path}")
    print_summary(summary)
    print("\nFinalizado.")


if __name__ == "__main__":
    main()
