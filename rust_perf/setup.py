from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="rust_perf",
    version="0.1.0",
    rust_extensions=[RustExtension("rust_perf.rust_perf", binding=Binding.PyO3)],
    packages=["rust_perf"],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)
