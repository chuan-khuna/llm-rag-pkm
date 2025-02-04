import os


def list_all_files(directory: str, ext: str):
    """
    list all files in a directory with a specific extension
    """
    filtered_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                filtered_files.append(os.path.join(root, file))
    return filtered_files
