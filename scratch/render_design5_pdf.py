import os
import pypdfium2 as pdfium

pdf_path = r"C:\Users\saipu\Downloads\camera-design5.pdf"
out_dir = r"c:\Users\saipu\OneDrive\Desktop\fiber polytopes-2\scratch\pdf_pages_design5"
os.makedirs(out_dir, exist_ok=True)

pdf = pdfium.PdfDocument(pdf_path)
print(f"Total pages: {len(pdf)}")

for i in range(len(pdf)):
    page = pdf[i]
    image = page.render(scale=2.0).to_pil()
    out_path = os.path.join(out_dir, f"page_{i+1}.png")
    image.save(out_path)
    print(f"Saved page {i+1} -> {out_path}")
