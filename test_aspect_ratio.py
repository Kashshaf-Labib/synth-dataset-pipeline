"""Quick test for compute_best_aspect_ratio."""
from pipeline.image_generator import compute_best_aspect_ratio

tests = [
    (1024, 1024, "1:1"),
    (2048, 2500, "4:5"),
    (3000, 2000, "3:2"),
    (1024, 1792, "9:16"),
    (1792, 1024, "16:9"),
    (800, 1000, "4:5"),
    (600, 900, "2:3"),
    (400, 300, "4:3"),
]

all_pass = True
for w, h, expected in tests:
    result = compute_best_aspect_ratio(w, h)
    status = "PASS" if result == expected else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  {w}x{h} -> {result:5s}  (expected {expected}, {status})")

print()
print("=== ALL PASSED ===" if all_pass else "=== SOME TESTS FAILED ===")
