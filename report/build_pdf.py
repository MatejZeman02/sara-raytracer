import base64
import re

# read the html
html_file = "report_print_ready.html"
with open(html_file, "r", encoding="utf-8") as f:
    html_data = f.read()


# find images and encode as base64
def encode_image(match):
    img_tag = match.group(0)
    src_match = re.search(r'src="([^"]+)"', img_tag)
    if not src_match:
        return img_tag

    img_path = src_match.group(1)

    if img_path.startswith("data:") or img_path.startswith("http"):
        return img_tag

    try:
        with open(img_path, "rb") as img_f:
            encoded = base64.b64encode(img_f.read()).decode("utf-8")
            ext = img_path.split(".")[-1].lower()
            mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
            new_src = f"data:{mime};base64,{encoded}"
            return img_tag.replace(f'src="{img_path}"', f'src="{new_src}"')
    except Exception as e:
        print(f"warning: could not embed {img_path} - {e}")
        return img_tag


# replace
embedded_html = re.sub(r"<img[^>]+>", encode_image, html_data)

# save standalone html
with open("report_standalone.html", "w", encoding="utf-8") as f:
    f.write(embedded_html)
print("created 'report_standalone.html' with embedded images.")

# attempt pdf generation
try:
    from weasyprint import HTML

    print("generating pdf with weasyprint...")
    HTML(string=embedded_html).write_pdf("final_report.pdf")
    print("successfully created 'final_report.pdf'.")
except ImportError:
    print("\nweasyprint is not installed. to build the pdf directly from python, run:")
    print("or open 'report_standalone.html' in firefox and use print to pdf!")
