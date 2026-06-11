"""Store backends for the store-comparison suite (test_stores.py).

One module per backend, each exposing `ensure(src)` — build the store
from an extxyz file once, cached next to it in benchmarks/.cache/ — and
the read paths the benchmarks exercise. Ingest is setup, never a
benchmark: conversion is not a routine ETL step, and the comparison we
care about is reading.
"""
