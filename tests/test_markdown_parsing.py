"""Tests for parse_markdown_to_blocks in create_report.py."""

import wandb_workspaces.reports.v2 as wr

from wandb_mcp_server.mcp_tools.create_report import parse_markdown_to_blocks


class TestMarkdownHeaders:
    def test_h1(self):
        blocks = parse_markdown_to_blocks("# Title")
        assert len(blocks) == 1
        assert isinstance(blocks[0], wr.H1)

    def test_h2(self):
        blocks = parse_markdown_to_blocks("## Section")
        assert len(blocks) == 1
        assert isinstance(blocks[0], wr.H2)

    def test_h3(self):
        blocks = parse_markdown_to_blocks("### Subsection")
        assert len(blocks) == 1
        assert isinstance(blocks[0], wr.H3)

    def test_all_header_levels(self):
        md = "# H1\n## H2\n### H3"
        blocks = parse_markdown_to_blocks(md)
        assert len(blocks) == 3
        assert isinstance(blocks[0], wr.H1)
        assert isinstance(blocks[1], wr.H2)
        assert isinstance(blocks[2], wr.H3)


class TestMarkdownTOC:
    def test_toc_marker(self):
        blocks = parse_markdown_to_blocks("[TOC]")
        assert len(blocks) == 1
        assert isinstance(blocks[0], wr.TableOfContents)

    def test_toc_case_insensitive(self):
        blocks = parse_markdown_to_blocks("[toc]")
        assert len(blocks) == 1
        assert isinstance(blocks[0], wr.TableOfContents)


class TestMarkdownCodeBlocks:
    def test_code_block_with_language(self):
        md = "```python\nprint('hello')\n```"
        blocks = parse_markdown_to_blocks(md)
        assert len(blocks) == 1
        assert isinstance(blocks[0], wr.CodeBlock)

    def test_code_block_bash_maps_to_valid_language(self):
        md = "```bash\necho hello\n```"
        blocks = parse_markdown_to_blocks(md)
        assert len(blocks) == 1
        assert isinstance(blocks[0], wr.CodeBlock)

    def test_code_block_unknown_language(self):
        md = "```rust\nfn main() {}\n```"
        blocks = parse_markdown_to_blocks(md)
        assert len(blocks) == 1
        assert isinstance(blocks[0], wr.CodeBlock)

    def test_code_block_no_language(self):
        md = "```\nsome code\n```"
        blocks = parse_markdown_to_blocks(md)
        assert len(blocks) == 1
        assert isinstance(blocks[0], wr.CodeBlock)

    def test_header_inside_code_block_not_extracted(self):
        md = "```python\n# This is a comment, not a header\nx = 1\n```"
        blocks = parse_markdown_to_blocks(md)
        assert len(blocks) == 1
        assert isinstance(blocks[0], wr.CodeBlock)


class TestMarkdownRichContent:
    def test_table_produces_markdown_block(self):
        md = "| col1 | col2 |\n| --- | --- |\n| a | b |"
        blocks = parse_markdown_to_blocks(md)
        assert len(blocks) == 1
        assert isinstance(blocks[0], wr.MarkdownBlock)

    def test_bullet_list_produces_markdown_block(self):
        md = "- item 1\n- item 2\n- item 3"
        blocks = parse_markdown_to_blocks(md)
        assert len(blocks) == 1
        assert isinstance(blocks[0], wr.MarkdownBlock)

    def test_ordered_list_produces_markdown_block(self):
        md = "1. first\n2. second\n3. third"
        blocks = parse_markdown_to_blocks(md)
        assert len(blocks) == 1
        assert isinstance(blocks[0], wr.MarkdownBlock)


class TestMarkdownParagraphs:
    def test_simple_paragraph(self):
        blocks = parse_markdown_to_blocks("Hello world")
        assert len(blocks) == 1
        assert isinstance(blocks[0], wr.P)

    def test_empty_input(self):
        blocks = parse_markdown_to_blocks("")
        assert len(blocks) == 1
        assert isinstance(blocks[0], wr.P)

    def test_formatted_text_uses_markdown_block(self):
        md = "This has **bold** and `code`"
        blocks = parse_markdown_to_blocks(md)
        assert len(blocks) == 1
        assert isinstance(blocks[0], wr.MarkdownBlock)


class TestMarkdownMixed:
    def test_mixed_content(self):
        md = "\n".join(
            [
                "# Report Title",
                "[TOC]",
                "## Overview",
                "This is a summary paragraph.",
                "- bullet one",
                "- bullet two",
                "### Code Example",
                "```python",
                "x = 42",
                "```",
                "## Conclusion",
                "Done.",
            ]
        )
        blocks = parse_markdown_to_blocks(md)

        types = [type(b).__name__ for b in blocks]
        assert "H1" in types
        assert "TableOfContents" in types
        assert "H2" in types
        assert "H3" in types
        assert "CodeBlock" in types
        assert len(blocks) >= 7
