#!/usr/bin/env bash
# Verify PyPI openpi-client 0.1.2 image_tools matches the validated local checkout.
set -e
source ~/webvla-env.sh
cd /tmp/opc
python3 - <<'EOF'
import json, urllib.request
d = json.load(urllib.request.urlopen("https://pypi.org/pypi/openpi-client/0.1.2/json"))
url = [u["url"] for u in d["urls"] if u["filename"].endswith(".whl")][0]
data = urllib.request.urlopen(url).read()
import zipfile, io
z = zipfile.ZipFile(io.BytesIO(data))
open("pypi_image_tools.py", "w").write(z.read("openpi_client/image_tools.py").decode())
print("downloaded", len(data), "bytes")
EOF
diff pypi_image_tools.py /root/webvla/openpi/packages/openpi-client/src/openpi_client/image_tools.py && echo IDENTICAL
