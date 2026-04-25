import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def remove_lines_with_string(filename, target_string):
    # Start with the base name 'log_.txt' and find an available number to avoid overwriting
    logfile = "Logs0.txt"
    count = 0
    while logfile in os.listdir():
        count += 1
        logfile = f"Logs{count}.txt"
    
    # Open the input file for reading and the new log file for writing
    with open(filename, 'r') as input_file, open(logfile, 'w') as output_file:
        for line in input_file:
            # Only write lines that do NOT contain the target string
            if target_string not in line:
                output_file.write(line)
    
    # Log file creation is done, and it's safe to proceed

# Usage
remove_lines_with_string('output.txt', 'Working')