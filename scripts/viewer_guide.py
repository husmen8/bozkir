"""Write docs/viewer-guide.md from web/info.json.

The viewer's info drawer and this page are the same text, so they cannot
drift apart; tests/test_pipeline.py fails when the page is out of date.

    python scripts/viewer_guide.py
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['about', 'tileset', 'tiles', 'terrain', 'world', 'relief', 'detail',
         'rule', 'share', 'blend', 'decision', 'presets', 'camera', 'walk',
         'look', 'lighting', 'macro', 'renderer', 'farfield', 'quality',
         'warp', 'lod', 'ordering', 'merging', 'measure', 'repetition']


def render(topics):
    """The guide as markdown: topics in reading order, then any others."""
    ids = [k for k in ORDER if k in topics] + sorted(set(topics) - set(ORDER))
    out = ['# The viewer, control by control', '',
           'Generated from `web/info.json` by `scripts/viewer_guide.py`; the',
           "same text is the viewer's info drawer (the ⓘ marks, or I over a",
           'control). Edit the JSON, not this page.', '']
    out += [f'- [{topics[k]["title"]}](#{k})' for k in ids]
    for k in ids:
        t = topics[k]
        out += ['', f'<a id="{k}"></a>', f'## {t["title"]}', '']
        if t.get('short'):
            out += [f'*{t["short"]}*', '']
        # A picture only once it exists (web/img is filled by hand).
        if t.get('image') and (ROOT / 'web' / t['image']).exists():
            out += [f'![{t["title"]}](../web/{t["image"]})', '']
        for p in t.get('body', []):
            out += [p, '']
        for s in t.get('sources', []):
            out.append(f'- {s}')
        if t.get('code'):
            out.append('- In the code: ' + ', '.join(f'`{c}`' for c in t['code']))
        if t.get('related'):
            out.append('- See also: ' + ', '.join(
                f'[{topics[r]["title"]}](#{r})' for r in t['related'] if r in topics))
    text = '\n'.join(out).rstrip() + '\n'
    return re.sub(r'\n{3,}', '\n\n', text)


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    topics = json.loads((ROOT / 'web/info.json').read_text(encoding='utf-8'))
    dest = ROOT / 'docs/viewer-guide.md'
    dest.write_text(render(topics), encoding='utf-8', newline='\r\n')
    print(f'wrote {dest.relative_to(ROOT)} ({len(topics)} topics)')


if __name__ == '__main__':
    main()
