import pypandoc
from pathlib import Path

def word_to_markdown(docx_path: str, output_md: str = None):
    docx_path = Path(docx_path)

    if output_md is None:
        output_md = docx_path.with_suffix(".md")

    pypandoc.convert_file(
        source_file=str(docx_path),
        to="md",
        format="docx",
        outputfile=str(output_md),
        extra_args=[
            "--wrap=none",          # 不自动换行
            "--markdown-headings=atx"  # 使用 ### 风格标题
        ]
    )

    print(f"转换完成: {output_md}")

if __name__ == "__main__":
    word_to_markdown("新一代交易系统非功能技术需求说明书_20251219.docx")
