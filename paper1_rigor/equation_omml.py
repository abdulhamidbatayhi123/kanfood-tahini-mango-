"""Turn a fitted model's LaTeX into an editable Word equation, with no hand-typing anywhere.

Why this exists
---------------
The equations in the previous draft were transcribed by hand into a rendering script. That is the
exact failure mode the playbook forbids: a coefficient can be mistyped and nothing will catch it,
because the printed equation and the model that produced it are no longer connected. Here the
chain is closed end to end:

    fitted model -> sympy expression -> LaTeX -> MathML -> OMML -> .docx

and `verify()` reads the OMML back, strips it to plain text, and checks that every coefficient in
it appears in the source expression. An equation that does not survive that check is not written.

The MathML-to-OMML step uses `MML2OMML.XSL`, the transform Microsoft ships with Office, so the
result is a real Word equation object the authors can edit, not a picture of one.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission_foods" / "equations"
OUT.mkdir(parents=True, exist_ok=True)

XSL_CANDIDATES = [
    Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"),
    Path(r"C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL"),
]


def find_xsl():
    for p in XSL_CANDIDATES:
        if p.exists():
            return p
    hits = list(Path(r"C:\Program Files").rglob("MML2OMML.XSL"))
    if hits:
        return hits[0]
    raise FileNotFoundError("MML2OMML.XSL not found; Word's MathML transform is required")


def latex_to_omml(latex):
    """LaTeX -> MathML -> OMML, returning the `<m:oMath>` fragment as a string."""
    import latex2mathml.converter
    from lxml import etree
    mathml = latex2mathml.converter.convert(latex)
    xslt = etree.parse(str(find_xsl()))
    transform = etree.XSLT(xslt)
    omml = transform(etree.fromstring(mathml.encode("utf-8")))
    xml = etree.tostring(omml, encoding="unicode")
    xml = re.sub(r"<\?xml[^>]*\?>\s*", "", xml)
    return xml


NUM = re.compile(r"\d+\.?\d*")


def verify(omml_xml, latex, tol_missing=0):
    """Every numeric coefficient in the LaTeX must survive into the OMML.

    Word splits a run at almost any character, so the check is on the concatenated text of the
    `<m:t>` elements rather than on element structure.
    """
    from lxml import etree
    root = etree.fromstring(f"<w xmlns:m='http://schemas.openxmlformats.org/officeDocument/2006/math'>"
                            f"{omml_xml}</w>")
    text = "".join(t.text or "" for t in root.iter(
        "{http://schemas.openxmlformats.org/officeDocument/2006/math}t"))
    flat = text.replace(" ", "")
    want = NUM.findall(re.sub(r"\\[a-zA-Z]+", " ", latex).replace(" ", ""))
    # Word emits each symbol as its own run, so adjacent numbers end up concatenated in the
    # flattened text ("0.001" then "0.2" reads as "0.0010.2"). Splitting that back into tokens is
    # ambiguous, so instead every wanted coefficient is matched IN ORDER, consuming the string as
    # it goes. That checks presence and sequence together, which is what can actually go wrong.
    pos, missing = 0, []
    for w in want:
        i = flat.find(w, pos)
        if i < 0:
            missing.append(w)
        else:
            pos = i + len(w)
    return (len(missing) <= tol_missing), missing, text


def write(latex, name, label=None):
    """Write `<name>.xml` (the OMML fragment) and `<name>.tex` (its source), after checking."""
    xml = latex_to_omml(latex)
    ok, missing, text = verify(xml, latex)
    if not ok:
        raise AssertionError(f"{name}: coefficients lost in conversion: {missing}")
    (OUT / f"{name}.xml").write_text(xml, encoding="utf-8")
    (OUT / f"{name}.tex").write_text(latex, encoding="utf-8")
    print(f"  -> equations/{name}.xml  ({len(text)} characters of maths, all coefficients checked)")
    return xml


def sympy_to_latex(expr, lhs=r"\hat{y}", round_to=3):
    """A sympy expression as display LaTeX, linear terms first so the equation reads sensibly."""
    import sympy
    e = sympy.nsimplify(expr, rational=False) if False else expr
    terms = e.args if e.is_Add else (e,)
    linear, other = [], []
    for t in terms:
        (linear if t.is_polynomial() and sympy.total_degree(t) <= 1 else other).append(t)
    ordered = sympy.Add(*linear, evaluate=False) if linear else None
    parts = []
    if ordered is not None:
        parts.append(sympy.latex(sympy.Add(*linear)))
    for t in other:
        s = sympy.latex(t)
        parts.append(s if s.startswith("-") else "+ " + s)
    body = " ".join(parts) if parts else sympy.latex(e)
    return f"{lhs} \\approx {body}"


if __name__ == "__main__":
    demo = (r"\hat{y}_{\mathrm{tahini}} \approx 1.034 x_{1} - 0.700 x_{2} + 0.001 "
            r"\left(0.2 - 10 x_{8}\right)^{2}")
    ok, missing, text = verify(latex_to_omml(demo), demo)
    print("self-test:", "OK" if ok else f"FAILED, missing {missing}")
    print("           round-tripped text:", text)


# --------------------------------------------------------------------------- line wrapping
def split_display_latex(latex, max_chars=88):
    r"""Break a display equation into lines at TOP-LEVEL + and - signs only.

    A single-line equation of this length runs past the right margin of a two-column-width MDPI
    text block and is silently clipped in the PDF -- which is how this was found. Splitting has to
    respect grouping: a `-` inside `\left( ... \right)` or inside `{ ... }` is part of a term, not
    a break point, so cutting there would corrupt the maths. Depth is tracked for both.

    Returns a list of LaTeX strings; the first carries the `lhs \approx` prefix and each later one
    begins with the operator that joins it to the previous.
    """
    head, _, body = latex.partition(r"\approx")
    head = head.strip()
    body = body.strip()

    # find top-level break points
    depth, i, breaks = 0, 0, []
    while i < len(body):
        if body.startswith(r"\left", i):
            depth += 1; i += 5; continue
        if body.startswith(r"\right", i):
            depth -= 1; i += 6; continue
        c = body[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c in "+-" and depth == 0 and i > 0 and body[i - 1] == " ":
            breaks.append(i)
        i += 1

    # greedily fill lines up to max_chars, breaking only at those points
    lines, start, last_ok = [], 0, None
    for b in breaks + [len(body)]:
        if b - start > max_chars and last_ok is not None:
            lines.append(body[start:last_ok].strip())
            start, last_ok = last_ok, b
        else:
            last_ok = b
    lines.append(body[start:].strip())
    lines = [l for l in lines if l]

    out = [f"{head} " + r"\approx" + f" {lines[0]}"]
    out += [l if l[0] in "+-" else "+ " + l for l in lines[1:]]
    return out


def write_wrapped(latex, name, max_chars=88):
    """Write `<name>.xml` holding one `<m:oMath>` per display line, each verified separately.

    `build_docx.add_equation` places each `<m:oMath>` in its own paragraph, so the equation reads
    as a normal multi-line display and stays a real, editable Word equation on every line.
    """
    lines = split_display_latex(latex, max_chars)
    frags, total = [], []
    for k, line in enumerate(lines, 1):
        xml = latex_to_omml(line)
        ok, missing, text = verify(xml, line)
        if not ok:
            raise AssertionError(f"{name} line {k}: coefficients lost in conversion: {missing}")
        frags.append(xml)
        total.append(text)
    # every coefficient of the whole equation must still be present, in order, across the lines
    ok, missing, _ = verify("".join(frags), latex)
    if not ok:
        raise AssertionError(f"{name}: coefficients lost across the split: {missing}")
    (OUT / f"{name}.xml").write_text("".join(frags), encoding="utf-8")
    (OUT / f"{name}.tex").write_text(latex, encoding="utf-8")
    print(f"  -> equations/{name}.xml  ({len(lines)} lines, "
          f"{sum(len(t) for t in total)} characters of maths, all coefficients checked)")
    return frags
