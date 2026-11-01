import xmlrpc.client

s = xmlrpc.client.ServerProxy("http://192.168.0.197:41414")

print("Calling rg_get_status for id 0:")
try:
    print("rg_get_status(0):", s.rg_get_status(0))
except Exception as e:
    print("Error:", e)

try:
    print("rg_get_status(1):", s.rg_get_status(1))
except Exception as e:
    print("Error:", e)

try:
    print("rg_get_width(0):", s.rg_get_width(0))
except Exception as e:
    print("Error:", e)

try:
    print("rg_get_busy(0):", s.rg_get_busy(0))
except Exception as e:
    print("Error:", e)

try:
    print("rg_get_all_variables(0):", s.rg_get_all_variables(0))
except Exception as e:
    print("Error:", e)
