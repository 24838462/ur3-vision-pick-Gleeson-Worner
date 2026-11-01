import urllib.request
import xmlrpc.client

url = "http://192.168.0.197:41414"

def test_call(method, params):
    xml_request = f"""<?xml version="1.0"?>
<methodCall>
<methodName>{method}</methodName>
<params>
"""
    for p in params:
        if isinstance(p, int):
            xml_request += f"<param><value><int>{p}</int></value></param>\n"
        elif isinstance(p, float):
            xml_request += f"<param><value><double>{p}</double></value></param>\n"
        elif isinstance(p, bool):
            xml_request += f"<param><value><boolean>{1 if p else 0}</boolean></value></param>\n"
        else:
            xml_request += f"<param><value><string>{p}</string></value></param>\n"
    xml_request += """</params>
</methodCall>"""

    req = urllib.request.Request(
        url,
        data=xml_request.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = resp.read().decode("utf-8")
            print(f"[{method}] Response: {data.strip()}")
            parsed, _ = xmlrpc.client.loads(data)
            return parsed
    except Exception as exc:
        print(f"[{method}] Error: {exc}")
        return None

print("Testing gripper calls on 192.168.0.197:41414 ...")
# Test RG gripper with ID 0 and ID 1
for gid in [0, 1]:
    print(f"\n--- Testing Gripper ID {gid} ---")
    test_call("rg_get_width", [gid])
    test_call("rg_get_status", [gid])
    test_call("twofg_get_external_width", [gid])

# Check system methods
print("\n--- Testing system introspection ---")
test_call("system.listMethods", [])
