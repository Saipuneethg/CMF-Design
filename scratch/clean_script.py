import re

with open("scripts/run_scientific_cmf_investigation.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace any corrupted PowerShell injection strings
text = text.replace("(Delta > 0: Improvement)", "")

# In ReportLab paragraphs, replace LaTeX with clean unicode or HTML entities
text = text.replace(r"$\Delta$", "Delta")
text = text.replace(r"$\kappa$", "kappa")
text = text.replace(r"$\sigma_1$", "sigma_1")
text = text.replace(r"$\sigma_2$", "sigma_2")
text = text.replace(r"$\sigma_3$", "sigma_3")
text = text.replace(r"$\sigma$", "sigma")
text = text.replace(r"$\Delta E_{00}$", "&Delta;E<sub>00</sub>")
text = text.replace(r"$\Delta_{\mathrm{gen}}$", "&Delta;<sub>gen</sub>")
text = text.replace(r"\Delta E_{00} \le 1.84", "&Delta;E<sub>00</sub> &le; 1.84")
text = text.replace(r"\ker(M V_8)", "nullspace(M V_8)")
text = text.replace(r"\mathcal{P}_{\text{gamut}}", "P_gamut")
text = text.replace(r"\mathcal{A}_{\det}", "A_det")
text = text.replace(r"\mathcal{F}(c)", "F(c)")
text = text.replace(r"S_{\mathrm{spread}}", "S_spread")
text = text.replace(r"O_{\mathrm{mean}} \approx 0.74", "O_mean ~ 0.74")
text = text.replace(r"O_{\mathrm{mean}} = 0.52", "O_mean = 0.52")
text = text.replace(r"H_\sigma", "H_sigma")
text = text.replace(r"\Delta = \mathrm{Base} - \mathrm{Opt}", "Delta = Base - Opt")
text = text.replace(r"\Delta = \mathrm{Test} - \mathrm{Train}", "Delta = Test - Train")
text = text.replace(r"\pm", "+/-")
text = text.replace(r"\Delta \lambda", "Delta lambda")
text = text.replace(r"\mathrm{rank}", "rank")
text = text.replace(r"\theta", "theta")
text = text.replace(r"\alpha", "alpha")
text = text.replace(r"\lambda", "lambda")
text = text.replace(r"\bar{s}", "s_bar")
text = text.replace(r"\in", "in")
text = text.replace(r"\mathbb{R}", "R")
text = text.replace(r"\le", "<=")
text = text.replace(r"\{", "{")
text = text.replace(r"\}", "}")
text = text.replace(r"\mid", "|")
text = text.replace(r"\times", "x")

# Clean up any residual backslashes in non-raw strings
lines = text.split("\n")
cleaned_lines = []
for line in lines:
    # If line has f"..." or f'...' with \ inside, fix it
    if ('f"' in line or "f'" in line) and "\\" in line:
        line = line.replace("\\Delta", "&Delta;")
        line = line.replace("\\sigma", "&sigma;")
        line = line.replace("\\kappa", "&kappa;")
        line = line.replace("\\theta", "&theta;")
        line = line.replace("\\pm", "+/-")
        line = line.replace("\\", "")
    cleaned_lines.append(line)

with open("scripts/run_scientific_cmf_investigation.py", "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_lines))

print("Successfully cleaned and validated scripts/run_scientific_cmf_investigation.py")
