import ctypes
import time
import os

# Full path to the DLL
dll_path = os.path.abspath("print_numbers.dll")
lib = ctypes.CDLL(dll_path)

start_time = time.time()
lib.print_numbers()
end_time = time.time()

print(f"Time taken: {end_time - start_time:.2f} seconds")
