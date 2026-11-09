import os, sys, subprocess

my_pid = os.getpid()
print(f"Current PID: {my_pid}")

# Get all python PIDs via tasklist
output = subprocess.check_output('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH', shell=True).decode()
pids = []
for line in output.strip().splitlines():
    parts = line.split(',')
    if len(parts) >= 2:
        try:
            pid = int(parts[1].strip('"'))
            if pid != my_pid:
                pids.append(pid)
        except ValueError:
            pass

print(f"Found {len(pids)} lingering python processes to terminate: {pids}")
for pid in pids:
    try:
        subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
        print(f"Terminated PID {pid}")
    except Exception as e:
        print(f"Failed to terminate PID {pid}: {e}")

print("Clean-up complete!")
