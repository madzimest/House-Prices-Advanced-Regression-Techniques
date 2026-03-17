import os

# Root directory you want to scan
root_dir = "."  # Change this to your directory
output_file = "all_files.txt"

# File extensions to look for
extensions = (".py", ".css")

with open(output_file, "w", encoding="utf-8") as out_f:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(extensions):
                full_path = os.path.join(dirpath, file)
                print(full_path)
                out_f.write(f"Filename: {file}\n")
                out_f.write(f"Path: {full_path}\n")
                out_f.write("Contents:\n")
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        out_f.write(f.read())
                except Exception as e:
                    out_f.write(f"[Error reading file: {e}]")
                out_f.write("\n\n---\n\n")

print(f"All {extensions} files have been combined into {output_file}")
