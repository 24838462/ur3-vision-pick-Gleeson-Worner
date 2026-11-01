import urllib.request
import xmlrpc.client

req = """<?xml version="1.0"?>
<methodCall>
<methodName>system.listMethods</methodName>
<params></params>
</methodCall>"""

r = urllib.request.Request(
    "http://192.168.0.197:41414",
    data=req.encode("utf-8"),
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)
with urllib.request.urlopen(r, timeout=3.0) as resp:
    data = resp.read().decode("utf-8")
    methods, _ = xmlrpc.client.loads(data)
    methods = methods[0]
    print(f"Total methods: {len(methods)}")
    rg_methods = [m for m in methods if "rg" in m.lower()]
    print("Methods matching 'rg':", rg_methods)
    grip_methods = [m for m in methods if "grip" in m.lower()]
    print("Methods matching 'grip':", grip_methods)
