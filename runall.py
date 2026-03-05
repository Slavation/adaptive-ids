# Script to run all 91 pairwise attack experiments for the Reptile IDS model
# @Author: Vladislav Zagidulin
import subprocess
import itertools
import sys

# All attacks in EdgeIIoTset dataset
attacks = [
    "Backdoor", "DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP", "DDoS_UDP",
    "Fingerprinting", "MITM", "Password", "Port_Scanning",
    "Ransomware", "SQL_injection", "Uploading",
    "Vulnerability_scanner", "XSS"
]

# Generate all possible unique pairs
combinations = list(itertools.combinations(attacks, 2))

print(f"Starting 91 experiments...")

# Execute experiments for each pair
for a1, a2 in combinations:
    subprocess.run([sys.executable, "adaptive-ids.py", a1, a2])

print("All 91 experiments finished!")