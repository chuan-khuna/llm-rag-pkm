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


def split_readable_file_path(split_by: str, file_path: str):
    """
    split a file path by a specific character and return the last element
    """
    return file_path.split(split_by)[-1][1:]
