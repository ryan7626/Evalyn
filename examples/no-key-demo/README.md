# No-key differential demo

This demo executes the trusted pytest benchmark fixture without contacting a
model or container runtime. It is deliberately small enough to inspect:

1. the base integer-cent implementation passes one probe three times;
2. a float-rounding candidate fails that same probe three times;
3. the fixed implementation passes again;
4. Evalyn’s classifier records stable differential evidence.

From the repository root:

~~~bash
pip install -e '.[dev]'
python examples/no-key-demo/run_demo.py
~~~

The demo copies snapshots into operating-system temporary directories. It does
not initialize Git, change the current checkout, install dependencies, read API
keys, or access the network.

