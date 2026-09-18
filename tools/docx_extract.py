"""Stdlib-only DOCX text/table extractor used to read the frozen spec (no pandoc/python-docx needed).

Usage: python3 tools/docx_extract.py <file.docx> > out.txt
Paragraphs are printed as "[Style] text"; tables as "| cell | cell |" rows between <TABLE> markers.
"""
import sys
import zipfile
import xml.etree.ElementTree as ET

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def ptext(p):
    out = []
    for n in p.iter():
        if n.tag == W + 't':
            out.append(n.text or '')
        elif n.tag == W + 'tab':
            out.append('\t')
        elif n.tag in (W + 'br', W + 'cr'):
            out.append('\n')
    return ''.join(out)


def style(p):
    s = p.find(f'{W}pPr/{W}pStyle')
    return s.get(W + 'val') if s is not None else ''


def main(path):
    z = zipfile.ZipFile(path)
    body = ET.fromstring(z.read('word/document.xml')).find(W + 'body')
    for el in body:
        if el.tag == W + 'p':
            t = ptext(el).strip()
            st = style(el)
            if t:
                print((f'[{st}] ' if st else '') + t)
        elif el.tag == W + 'tbl':
            print('<TABLE>')
            for tr in el.findall(W + 'tr'):
                cells = [' '.join(ptext(p).strip() for p in tc.iter(W + 'p')).strip() for tc in tr.findall(W + 'tc')]
                print('| ' + ' | '.join(cells) + ' |')
            print('</TABLE>')
    print('=== core.xml ===')
    print(z.read('docProps/core.xml').decode())
    print('=== app.xml ===')
    print(z.read('docProps/app.xml').decode())


if __name__ == '__main__':
    main(sys.argv[1])
