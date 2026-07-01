import psutil


def get_ram_usage():
    """
    Returns RAM usage percentage.
    """
    return psutil.virtual_memory().percent