import time

start_time = time.time()

for i in range(1, 1_000_001):
    print(i)

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")
