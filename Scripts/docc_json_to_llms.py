#!/usr/bin/env python3
"""
Render every DocC JSON page under docs/data/documentation/ into a single
Markdown concat at docs/llms.txt.

Bridges the gap until Xcode bundles a docc with
--enable-experimental-markdown-output. Output format mirrors the
intent of that flag: one section per symbol, prefixed with `---` and a
path header, suitable for grep / paste-into-LLM use.

Usage (invoked from Scripts/build_docs.sh when EMIT_LLMS_TXT=1):
    python3 Scripts/docc_json_to_llms.py <output_dir> <target_name>
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def render_inline(items, refs):
    """Render a list of inline content items to a plain-text Markdown string."""
    if not items:
        return ""
    out = []
    for it in items:
        t = it.get("type")
        if t == "text":
            out.append(it.get("text", ""))
        elif t == "codeVoice":
            out.append(f"`{it.get('code', '')}`")
        elif t == "emphasis":
            out.append(f"*{render_inline(it.get('inlineContent', []), refs)}*")
        elif t == "strong":
            out.append(f"**{render_inline(it.get('inlineContent', []), refs)}**")
        elif t == "reference":
            ident = it.get("identifier", "")
            ref = refs.get(ident, {}) if isinstance(refs, dict) else {}
            title = ref.get("title") or ref.get("name") or ident.split("/")[-1]
            out.append(f"`{title}`")
        elif t == "link":
            title = render_inline(it.get("inlineContent", []), refs) or it.get("destination", "")
            out.append(f"[{title}]({it.get('destination', '')})")
        elif t == "image":
            out.append(f"![{it.get('identifier', '')}]")
        elif t == "inlineHead":
            out.append(f"**{render_inline(it.get('inlineContent', []), refs)}**")
        else:
            # Fallback: try the most common nested field.
            nested = it.get("inlineContent") or it.get("content")
            if isinstance(nested, list):
                out.append(render_inline(nested, refs))
    return "".join(out)


def render_block(block, refs, depth=0):
    """Render a block content item to Markdown lines."""
    kind = block.get("type") or block.get("kind")
    if kind == "paragraph":
        return [render_inline(block.get("inlineContent", []), refs), ""]
    if kind == "codeListing":
        lang = block.get("syntax") or ""
        lines = block.get("code", [])
        return [f"```{lang}", *lines, "```", ""]
    if kind == "aside":
        style = (block.get("style") or "Note").capitalize()
        rendered = []
        for inner in block.get("content", []):
            rendered.extend(render_block(inner, refs, depth + 1))
        body = "\n".join(rendered).rstrip()
        return [f"> **{style}:** " + body.replace("\n", "\n> "), ""]
    if kind in ("orderedList", "unorderedList"):
        bullet = lambda i: f"{i + 1}." if kind == "orderedList" else "-"
        out = []
        for i, item in enumerate(block.get("items", [])):
            inner_lines = []
            for inner in item.get("content", []):
                inner_lines.extend(render_block(inner, refs, depth + 1))
            text = "\n".join(inner_lines).rstrip()
            text = text.replace("\n", "\n  ")
            out.append(f"{bullet(i)} {text}")
        out.append("")
        return out
    if kind == "heading":
        level = block.get("level", 3)
        return [f"{'#' * level} {block.get('text', '')}", ""]
    if kind == "termList":
        out = []
        for it in block.get("items", []):
            term = render_inline(it.get("term", {}).get("inlineContent", []), refs)
            defn_lines = []
            for d in it.get("definition", {}).get("content", []):
                defn_lines.extend(render_block(d, refs, depth + 1))
            out.append(f"- **{term}** — {' '.join(l.strip() for l in defn_lines if l.strip())}")
        out.append("")
        return out
    if kind == "content":
        out = []
        for inner in block.get("content", []):
            out.extend(render_block(inner, refs, depth))
        return out
    if kind == "table":
        # Minimal — DocC tables are rare in our codebase
        return ["(table content)", ""]
    return []


def render_parameters(d, refs):
    """Find any parameters section and render it."""
    out = []
    for sec in d.get("primaryContentSections", []):
        if sec.get("kind") == "parameters":
            out.append("### Parameters")
            out.append("")
            for p in sec.get("parameters", []):
                name = p.get("name", "")
                lines = []
                for inner in p.get("content", []):
                    lines.extend(render_block(inner, refs))
                body = "\n".join(lines).strip()
                out.append(f"- **{name}**: {body}")
            out.append("")
            break
    return out


def render_returns(d, refs):
    """Find any 'Return Value' content section and render."""
    out = []
    for sec in d.get("primaryContentSections", []):
        if sec.get("kind") == "content":
            for blk in sec.get("content", []):
                if blk.get("type") == "heading" and "Return" in (blk.get("text", "")):
                    # Found a return-value heading; render following content
                    # (DocC usually wraps return docs in this pattern.)
                    out.append("### Returns")
                    out.append("")
    return out


def render_declaration(d):
    """Pull the Swift declaration tokens and stitch them."""
    for sec in d.get("primaryContentSections", []):
        if sec.get("kind") == "declarations":
            for decl in sec.get("declarations", []):
                tokens = decl.get("tokens", [])
                code = "".join(t.get("text", "") for t in tokens)
                return ["```swift", code, "```", ""]
    return []


def render_page(path: Path, refs_root=None):
    """Render one DocC JSON page to a list of Markdown lines."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    meta = d.get("metadata", {})
    refs = d.get("references", {}) or {}
    title = meta.get("title") or path.stem
    kind = meta.get("symbolKind") or meta.get("role") or ""

    lines = []
    header = f"## {title}"
    if kind:
        header += f"  *({kind})*"
    lines.append(header)
    lines.append("")

    decl = render_declaration(d)
    if decl:
        lines.extend(decl)

    abstract = d.get("abstract")
    if abstract:
        lines.append(render_inline(abstract, refs))
        lines.append("")

    # Body content
    for sec in d.get("primaryContentSections", []):
        if sec.get("kind") == "content":
            for blk in sec.get("content", []):
                lines.extend(render_block(blk, refs))

    lines.extend(render_parameters(d, refs))

    return lines


def main():
    output_dir = Path(sys.argv[1])
    target = sys.argv[2]
    docs_root = output_dir / "data" / "documentation"
    if not docs_root.is_dir():
        sys.stderr.write(f"no DocC data dir at {docs_root}\n")
        sys.exit(1)

    pages = sorted(docs_root.rglob("*.json"))
    if not pages:
        sys.stderr.write(f"no JSON pages under {docs_root}\n")
        sys.exit(1)

    out_lines = [
        f"# {target} — DocC export for LLM consumption",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} from swift-docc JSON.",
        "",
        "Each section is prefixed with `---` and the page's data path so",
        "the file is splittable. Sections are ordered by path; deeper",
        "paths are nested below their parents.",
        "",
    ]

    for page in pages:
        rel = page.relative_to(output_dir)
        out_lines.append("---")
        out_lines.append(f"### Source: `{rel}`")
        out_lines.append("")
        out_lines.extend(render_page(page))

    llms = output_dir / "llms.txt"
    llms.write_text("\n".join(out_lines))
    with open(llms) as f:
        n = sum(1 for _ in f)
    print(f"Wrote {llms} ({n} lines, {len(pages)} pages).")


if __name__ == "__main__":
    main()
