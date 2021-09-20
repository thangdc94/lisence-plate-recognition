import os, psutil

def convert_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.1f %s" % (size, x)
        size /= 1024.0
    return size

def get_process_mem():
    process = psutil.Process(os.getpid())
    return convert_bytes(process.memory_info().rss)  # in bytes 