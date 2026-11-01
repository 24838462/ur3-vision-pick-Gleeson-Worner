import xmlrpc.client

s = xmlrpc.client.ServerProxy("http://192.168.0.197:41414/")

print("Testing with root path:")
for fn in ["rg_get_status", "rg_get_width", "rg_get_busy", "rg_get_grip_detected"]:
    try:
        res = getattr(s, fn)(0)
        print(f"  {fn}(0): {res}")
    except Exception as e:
        print(f"  {fn}(0) error: {e}")

try:
    print("rg_get_all_variables(0):", s.rg_get_all_variables(0))
except Exception as e:
    print("rg_get_all_variables error:", e)

# Test TwoFG and VG as well just in case
print("\nTesting other grippers:")
for fn in ["twofg_get_status", "twofg_get_external_width"]:
    try:
        print(f"  {fn}(0):", getattr(s, fn)(0))
    except Exception as e:
        print(f"  {fn}(0) error: {e}")
