```python
def run_toy():
    # Five-step search in a 16x16 grid
    for x in range(-8, 9):
        for y in range(-8, 9):
            # Check the squared distance to (3, -2)
            if (x - 3) ** 2 + (y + 2) ** 2 == 0:
                return [x, y]

# Final program
run_toy()
```