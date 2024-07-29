import os

def swap_numbers(file_path):
    """Swaps the numbers 15 for 0 and 16 for 1 in the specified file."""
    with open(file_path, 'r') as file:
        content = file.read()

    content = content.replace("15", "0").replace("16", "1")

    with open(file_path, 'w') as file:
        file.write(content)

# Get the current directory path
current_directory = os.getcwd()

# Iterate over files in the directory
for filename in os.listdir(current_directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(current_directory, filename)
        swap_numbers(file_path)
        print(f"Processed: {filename}")