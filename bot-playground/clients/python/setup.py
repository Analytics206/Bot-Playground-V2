from setuptools import setup, find_packages

setup(
    name="bot-playground-client",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # This tells setuptools packages are under src
    install_requires=[
        "websocket-client>=1.6.0",
        "PyJWT>=2.8.0",
    ],
    author="Bot Playground Team",
    description="Client SDK for Bot Playground platform",
)