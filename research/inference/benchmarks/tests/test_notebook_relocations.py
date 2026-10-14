"""Presentation paths must resolve from the relocated notebook folders."""
import ast
import nbformat
from research.inference.benchmarks.repair_notebook_relocations import TARGETS,ROOT,LINK


def test_relocated_notebook_sources_and_links():
    for path in TARGETS:
        notebook=nbformat.read(path,4);nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type=='code':
                ast.parse(cell.source)
                assert 'src/spi_spi_contract.py' not in cell.source
                assert 'notebooks/inference/figures' not in cell.source
                assert 'from mtsspi.figures import figure_style, style_axis' not in cell.source
            else:
                for match in LINK.finditer(cell.source):
                    reference=match[0].split('#',1)[0]
                    if reference and '://' not in reference:
                        assert (path.parent/reference).exists(),(path,reference)


def test_extended_notebook_assets_and_loader():
    path=next(p for p in TARGETS if p.name.endswith('lean-extended.ipynb'))
    notebook=nbformat.read(path,4)
    loader=next(c for c in notebook.cells if 'def _appendix_show(' in c.source)
    assert 'src/mtsspi/spi_spi_contract.py' in loader.source
    for cell in notebook.cells:
        if cell.cell_type=='code' and cell.source.startswith('_appendix_show('):
            relative=ast.literal_eval(ast.parse(cell.source).body[0].value.args[0])
            assert (ROOT/'results/order-parameter-inference'/relative).is_file()
