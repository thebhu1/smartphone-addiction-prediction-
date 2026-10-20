"""
Test Script: Verify Prompt 1 (Screen Time → Addiction Probability) Logic
=========================================================================
Tests the calculate_deterministic_probabilities() method against ALL
stated constraints from Prompt 1.
"""

import sys
sys.path.insert(0, '.')
from src.prediction import AddictionPredictor

predictor = AddictionPredictor()

def fmt(probs):
    """Format probabilities as percentages."""
    return f"Low={probs[0]*100:.1f}%  Mod={probs[1]*100:.1f}%  High={probs[2]*100:.1f}%"

def check_sum(probs, label):
    """Check probabilities sum to 100%."""
    total = sum(probs)
    ok = abs(total - 1.0) < 0.001
    status = "✅" if ok else "❌"
    if not ok:
        print(f"  {status} [{label}] Sum = {total*100:.2f}% (MUST be 100%)")
    return ok

all_pass = True

def test(name, condition, detail=""):
    global all_pass
    status = "✅ PASS" if condition else "❌ FAIL"
    if not condition:
        all_pass = False
    print(f"  {status}: {name}  {detail}")

print("=" * 70)
print("PROMPT 1 CONSTRAINT VERIFICATION")
print("=" * 70)

# -----------------------------------------------------------------------
# 1. Zero Usage (0 hrs): Low Risk = 100%
# -----------------------------------------------------------------------
print("\n--- Zero Usage (0 hrs) ---")
p = predictor.calculate_deterministic_probabilities(0.0)
print(f"  Result: {fmt(p)}")
check_sum(p, "0h")
test("Low Risk == 100%", abs(p[0] - 1.00) < 0.001)
test("Moderate Risk == 0%", abs(p[1] - 0.00) < 0.001)
test("High Risk == 0%", abs(p[2] - 0.00) < 0.001)

# -----------------------------------------------------------------------
# 2. Healthy User (1-3 hrs): Low Risk 80-95%
# -----------------------------------------------------------------------
print("\n--- Healthy User (1-3 hrs) ---")
for st in [1.0, 1.5, 2.0, 2.5, 3.0]:
    p = predictor.calculate_deterministic_probabilities(st)
    print(f"  Screen Time={st}h → {fmt(p)}")
    check_sum(p, f"{st}h")
    test(f"ST={st}h: Low Risk in [80%, 95%]",
         0.80 <= p[0] <= 0.95,
         f"(got {p[0]*100:.1f}%)")

# -----------------------------------------------------------------------
# 3. Average User (4-5 hrs): Moderate ~50%, Low ~45%, High ~5%
# -----------------------------------------------------------------------
print("\n--- Average User (4-5 hrs) ---")
for st in [4.0, 4.5, 5.0]:
    p = predictor.calculate_deterministic_probabilities(st)
    print(f"  Screen Time={st}h → {fmt(p)}")
    check_sum(p, f"{st}h")
    test(f"ST={st}h: Moderate around 50%",
         0.45 <= p[1] <= 0.55,
         f"(got {p[1]*100:.1f}%)")
    test(f"ST={st}h: Low around 45%",
         0.40 <= p[0] <= 0.50,
         f"(got {p[0]*100:.1f}%)")
    test(f"ST={st}h: High around 5%",
         0.02 <= p[2] <= 0.08,
         f"(got {p[2]*100:.1f}%)")

# -----------------------------------------------------------------------
# 4. Transition Zone (6-8 hrs): Moderate ~70%, High ~25%, Low ~5%
# -----------------------------------------------------------------------
print("\n--- Transition Zone (6-8 hrs) ---")
for st in [6.0, 7.0, 8.0]:
    p = predictor.calculate_deterministic_probabilities(st)
    print(f"  Screen Time={st}h → {fmt(p)}")
    check_sum(p, f"{st}h")
    test(f"ST={st}h: Moderate around 70%",
         0.65 <= p[1] <= 0.75,
         f"(got {p[1]*100:.1f}%)")
    test(f"ST={st}h: High around 25%",
         0.20 <= p[2] <= 0.30,
         f"(got {p[2]*100:.1f}%)")
    test(f"ST={st}h: Low around 5%",
         0.02 <= p[0] <= 0.08,
         f"(got {p[0]*100:.1f}%)")

# -----------------------------------------------------------------------
# 5. Heavy User (9-14 hrs): High 85-90%, Moderate 10-15%
# -----------------------------------------------------------------------
print("\n--- Heavy User (9-14 hrs) ---")
for st in [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]:
    p = predictor.calculate_deterministic_probabilities(st)
    print(f"  Screen Time={st}h → {fmt(p)}")
    check_sum(p, f"{st}h")
    test(f"ST={st}h: High Risk in [85%, 90%]",
         0.85 <= p[2] <= 0.90,
         f"(got {p[2]*100:.1f}%)")
    test(f"ST={st}h: Moderate in [10%, 15%]",
         0.10 <= p[1] <= 0.15,
         f"(got {p[1]*100:.1f}%)")

# -----------------------------------------------------------------------
# 6. Extreme High (15+ hrs): High > 95%, Mod < 4%, Low < 1%
# -----------------------------------------------------------------------
print("\n--- Extreme High (15+ hrs) ---")
for st in [15.0, 18.0, 20.0, 24.0]:
    p = predictor.calculate_deterministic_probabilities(st)
    print(f"  Screen Time={st}h → {fmt(p)}")
    check_sum(p, f"{st}h")
    test(f"ST={st}h: High Risk > 95%",
         p[2] > 0.95,
         f"(got {p[2]*100:.1f}%)")
    test(f"ST={st}h: Moderate < 4%",
         p[1] < 0.04,
         f"(got {p[1]*100:.1f}%)")
    test(f"ST={st}h: Low < 1%",
         p[0] < 0.01,
         f"(got {p[0]*100:.1f}%)")

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
if all_pass:
    print("🎉  ALL CONSTRAINTS PASSED!")
else:
    print("⚠️   SOME CONSTRAINTS FAILED — fixes needed.")
print("=" * 70)
