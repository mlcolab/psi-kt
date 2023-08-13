import subprocess
import re
command = 'nvidia-smi'
while True:
    p = subprocess.check_output(command)
    ram_using = re.findall(r'\b\d+MiB+ /', str(p))[0][:-5]
    ram_total = re.findall(r'/  \b\d+MiB', str(p))[0][3:-3]
    ram_percent = int(ram_using) / int(ram_total)