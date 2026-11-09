with open("scripts/run_scientific_cmf_investigation.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace any remaining LaTeX backslashes with raw strings or plain text
text = text.replace(r"\sum_c M_c(lambda)", "Sum M_c(lambda)")
text = text.replace(r"\sigma_1 (Mode 1)", "sigma_1 (Mode 1)")
text = text.replace(r"\sigma_2 (Mode 2)", "sigma_2 (Mode 2)")
text = text.replace(r"\sigma_3 (Mode 3)", "sigma_3 (Mode 3)")
text = text.replace(r"\sigma_3", "sigma_3")
text = text.replace(r"\sigma", "sigma")
text = text.replace(r"\kappa(A_\theta) = \sigma_1 / \sigma_3", "kappa(A) = sigma1 / sigma3")
text = text.replace(r"H_\\sigma / \log_2(3)", "H_sigma / log2(3)")
text = text.replace(r"(\Delta > 0 \implies \mathrm{Improvement})", "(Improvement > 0)")
text = text.replace(r"\kappa(A_\theta)", "kappa(A)")
text = text.replace(r"\Delta E_{00}", "Delta E00")

with open("scripts/run_scientific_cmf_investigation.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Remaining warnings eliminated.")
