def memory_usage():
    import psutil
    current_process = psutil.Process()
    memory = current_process.memory_info().rss
    return "%.1f %s" % (int(memory / (1024 * 1024))*0.001, "GB")