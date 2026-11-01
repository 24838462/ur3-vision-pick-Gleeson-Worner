import xmlrpc.client

s = xmlrpc.client.ServerProxy("http://192.168.0.197:41414/")

for gid in range(6):
    try:
        status = s.rg_get_status(gid)
        width = s.rg_get_width(gid)
        print(f"ID {gid} -> status: {status}, width: {width}")
    except Exception as e:
        print(f"ID {gid} -> error: {e}")
