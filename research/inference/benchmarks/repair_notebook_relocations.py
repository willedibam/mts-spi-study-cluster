"""Repair presentation paths after the research-layout move, preserving scientific outputs."""
from pathlib import Path
import hashlib
import json
import os
import re
import nbformat

ROOT=Path(__file__).resolve().parents[3]
FOLDER=ROOT/'research/inference/benchmarks/notebooks'
TARGETS=[FOLDER/name for name in [
    'order-parameter-benchmarks-lean.ipynb','order-parameter-benchmarks-lean-extended.ipynb',
    'order-parameter-benchmarks-lean-2.ipynb','order-parameter-benchmark-comparison.ipynb']]
TARGETS+=[ROOT/'research/methods/pair_sampling/notebooks/channel-pair-sampling.ipynb']
LINK=re.compile(r'(?<=\]\()([^\s)]+)(?=\))')


def relocated_link(reference,notebook):
    path,sep,fragment=reference.partition('#')
    if not path or '://' in path or path.startswith('mailto:'):return reference
    if (notebook.parent/path).exists():return reference
    destinations={
        'kuramoto-order-parameter-confirmation.ipynb':ROOT/'research/inference/kuramoto',
        'stuart-landau-order-coordinate.ipynb':ROOT/'research/inference/stuart_landau',
        'miller-huse-order-coordinate.ipynb':ROOT/'research/inference/miller_huse',
        'spi-pair-sampling-assessment.md':ROOT/'research/methods/pair_sampling/assessment.md',
        'cml_order_parameter.py':ROOT/'src/mtsspi/cml_order_parameter.py'}
    name=Path(path).name
    if name in destinations:destination=destinations[name]
    elif path.startswith('../../results/'):
        destination=ROOT/path.removeprefix('../../')
    else:
        matches=list((ROOT/'research').rglob(name))
        if len(matches)!=1:raise ValueError(f'Unresolved/ambiguous link in {notebook}: {reference}')
        destination=matches[0]
    assert destination.exists(),destination
    return os.path.relpath(destination,notebook.parent)+(sep+fragment if sep else '')


def repair():
    receipts=[]
    for path in TARGETS:
        before=path.read_bytes();nb=nbformat.read(path,4);outputs=[c.get('outputs') for c in nb.cells]
        for cell in nb.cells:
            if cell.cell_type=='markdown':
                cell.source=LINK.sub(lambda m:relocated_link(m[0],path),cell.source)
                cell.source=cell.source.replace('The detailed system notebooks remain the provenance and diagnostic records,',
                    'The system study folders and retained notebooks provide provenance and diagnostic records,')
            else:
                cell.source=cell.source.replace('src/spi_spi_contract.py','src/mtsspi/spi_spi_contract.py')
                cell.source=cell.source.replace('from mtsspi.figures import figure_style, style_axis, X_LABEL',
                    'from mtsspi.figures import figure_style\nfrom research.methods.pair_sampling.plot_large_m_pair_sampling import style_axis, X_LABEL')
                cell.source=cell.source.replace('FIGURE_DIR / "desai-zwanzig"', 'ROOT / "research/inference/benchmarks/notebooks"')
                cell.source=cell.source.replace('notebooks/inference/figures','results/order-parameter-inference/benchmark-figures')
                cell.source=cell.source.replace("'font.serif': ['Computer Modern Roman']","'font.serif': ['Computer Modern Roman', 'DejaVu Serif']")
                cell.source=cell.source.replace('"font.serif": ["Computer Modern Roman"]','"font.serif": ["Computer Modern Roman", "DejaVu Serif"]')
        assert outputs==[c.get('outputs') for c in nb.cells]
        if 'typography_refresh' in nb.metadata:
            nb.metadata.typography_refresh.preserved_prefix_sha256=hashlib.sha256(json.dumps(nb.cells[:37],sort_keys=True).encode()).hexdigest()
        nbformat.validate(nb);assert path.read_bytes()==before
        nbformat.write(nb,path)
        receipts.append(dict(notebook=str(path.relative_to(ROOT)),before_sha256=hashlib.sha256(before).hexdigest(),
                             after_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),outputs_preserved=True))
    # Keep the appendix generator's Markdown references correct on future builds.
    builder=FOLDER.parent/'build_lean_interpretation_appendix.py'
    source=builder.read_text();repaired=LINK.sub(lambda m:relocated_link(m[0],FOLDER/'order-parameter-benchmarks-lean-extended.ipynb'),source)
    if repaired!=source:
        # This is a mechanical link migration, not regeneration of notebook content.
        builder.write_text(repaired)
    return receipts


if __name__=='__main__':print(json.dumps(repair(),indent=2))
