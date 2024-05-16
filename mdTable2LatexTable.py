# JXu
# Convert a Markdown table to Latex.

PATH_INPUT_MD = './inpt_md.txt'
PATH_OPT_LATEX = './opt_latex.txt'

inlines = []
cline = ''

# read from inpt
inpt = '|    | backbone    | classifier   | data type   | acc              | f1               |\n\
|---:|:------------|:-------------|:------------|:-----------------|:-----------------|\n\
|  8 | bolt        | hcpa-ukb     | statfcBOLD  | 0.86291+-0.03434 | 0.86514+-0.03386 |\n\
|  9 | braingnn    | hcpa-ukb     | dynfcBOLD   | 0.67600+-0.02723 | 0.64733+-0.05787 |\n\
| 13 | nagphormer  | hcpa-ukb     | statfcBOLD  | 0.62125+-0.06573 | 0.74492+-0.04008 |\n\
| 11 | neurodetour | hcpa-ukb     | dynfcBOLD   | 0.71214+-0.01374 | 0.72079+-0.02147 |\n\
| 12 | graphormer  | hcpa-ukb     | statfcBOLD  | 0.44062+-0.01101 | 0.39091+-0.28145 |\n\
| 13 | nagphormer  | hcpa-ukb     | dynfcBOLD   | 0.71374+-0.01174 | 0.70174+-0.01315 |\n\
| 14 | graphormer  | hcpa-ukb     | dynfcBOLD   | 0.38550+-0.06149 | 0.50967+-0.04008 |\n\
| 15 | braingnn    | hcpa-ukb     | statfcBOLD  | 0.75689+-0.06640 | 0.78586+-0.04322 |\n\
| 16 | bnt         | hcpa-ukb     | statfcBOLD  | 0.92301+-0.03929 | 0.92923+-0.02825 |\n\
| 17 | neurodetour | hcpa-ukb     | statfcBOLD  | 0.90607+-0.02702 | 0.91291+-0.02097 |\n\
| 18 | bnt         | hcpa-ukb     | dynfcBOLD   | 0.82948+-0.10066 | 0.78714+-0.20357 |\n\
| 35 | bnt         | ukb-hcpa     | dynfcBOLD   | 0.79109+-0.06155 | 0.74557+-0.15500 |\n\
| 26 | nagphormer  | ukb-hcpa     | statfcBOLD  | 0.93029+-0.01088 | 0.93226+-0.01076 |\n\
| 27 | nagphormer  | ukb-hcpa     | dynfcBOLD   | 0.74548+-0.00647 | 0.73441+-0.00701 |\n\
| 28 | bnt         | ukb-hcpa     | dynfcBOLD   | 0.79109+-0.06155 | 0.74557+-0.15500 |\n\
| 29 | graphormer  | ukb-hcpa     | dynfcBOLD   | 0.51669+-0.16678 | 0.64364+-0.07898 |\n\
| 30 | neurodetour | ukb-hcpa     | dynfcBOLD   | 0.75905+-0.02472 | 0.75617+-0.02981 |\n\
| 31 | neurodetour | ukb-hcpa     | statfcBOLD  | 0.88726+-0.04995 | 0.90615+-0.03650 |\n\
| 32 | graphormer  | ukb-hcpa     | statfcBOLD  | 0.51466+-0.08892 | 0.57780+-0.14501 |\n\
| 33 | bnt         | ukb-hcpa     | statfcBOLD  | 0.86078+-0.03176 | 0.87040+-0.02899 |\n\
| 34 | braingnn    | ukb-hcpa     | statfcBOLD  | 0.81792+-0.03667 | 0.82497+-0.03273 |\n\
| 35 | braingnn    | ukb-hcpa     | dynfcBOLD   | 0.68784+-0.01474 | 0.63613+-0.03837 |'
lines = inpt.split('\n')
cline = lines[1].strip().replace('---', 'c')
cline = cline.replace(' ', '')
del lines[1]
for line in lines:
    line = line.strip()
    line = line[1:]
    line = line[:-1]
    line = line.replace('|', '&')
    line = line.split(' & ')
    line = [f"{float(it.split('+-')[0])*100:.2f}"+"$_{\\pm"+f"{float(it.split('+-')[1])*100:.2f}"+"}$" if '0.' in it else it for it in line]
    line = ' & '.join(line)
    line = line+'\\\\'+'\n'
    inlines.append(line)

# write to opt
out = ''
with open(PATH_OPT_LATEX, 'w') as opt:
    out += '\\begin{center}\n'
    out += '\\begin{tabular}'+'{' +cline+ '}'+'\n'
    for inline in inlines:
        # out += '\hline'+'\n'
        out += inline
    # out += '\hline'+'\n'
    out += '\\end{tabular}\n'
    out += '\\end{center}\n'
print(out)