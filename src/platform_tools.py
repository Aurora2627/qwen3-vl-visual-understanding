"""Resolve the native curl executable on Windows, macOS or Linux."""
import shutil

def curl_executable():
    executable=shutil.which('curl')
    if not executable:raise RuntimeError('Native curl executable is required for verified HTTP Range downloads')
    return executable
