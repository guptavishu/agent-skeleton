"""Demo of all three sandbox modes."""

import tempfile
import os

from agentos.executor import LocalSandbox, RestrictedSandbox

print("=" * 60)
print("1. LocalSandbox — full access (default)")
print("=" * 60)
sb = LocalSandbox()
r = sb.execute("print('hello from local sandbox')")
print(f"  stdout: {r.stdout.strip()}")
print(f"  rc: {r.returncode}")

r = sb.execute("import os; print('cwd:', os.getcwd())")
print(f"  os access: {r.stdout.strip()}")

print()
print("=" * 60)
print("2. LocalSandbox — restricted to one directory")
print("=" * 60)
d = tempfile.mkdtemp()
open(os.path.join(d, "ok.txt"), "w").write("this file is allowed")

sb = LocalSandbox(allowed_dirs=[d])

r = sb.execute(f"print(open('{os.path.join(d, 'ok.txt')}').read())")
print(f"  allowed read: {r.stdout.strip()} (rc={r.returncode})")

r = sb.execute("print(open('/etc/hosts').read()[:30])")
print(f"  blocked read: rc={r.returncode}")
print(f"  error: {r.stderr.strip().splitlines()[-1]}")

print()
print("=" * 60)
print("3. RestrictedSandbox — in-process, blocks dangerous builtins")
print("=" * 60)
sb = RestrictedSandbox()

tests = [
    ("math (allowed)",      "import math; print(math.factorial(10))"),
    ("json (allowed)",      "import json; print(json.dumps({'a': 1}))"),
    ("os (blocked)",        "import os"),
    ("subprocess (blocked)","import subprocess"),
    ("open (blocked)",      "open('/etc/passwd')"),
    ("eval (blocked)",      "eval('1+1')"),
    ("pure compute",        "print(2 ** 100)"),
]

for label, code in tests:
    r = sb.execute(code)
    if r.returncode == 0:
        print(f"  {label}: OK — {r.stdout.strip()}")
    else:
        print(f"  {label}: BLOCKED — {r.stderr.strip()}")

print()
print("=" * 60)
print("4. RestrictedSandbox — custom allowed modules")
print("=" * 60)
sb = RestrictedSandbox(allowed_modules=["math", "statistics"])

r = sb.execute("import statistics; print(statistics.mean([1,2,3,4,5]))")
print(f"  statistics (allowed): {r.stdout.strip()}")

r = sb.execute("import json")
print(f"  json (blocked): {r.stderr.strip()}")
