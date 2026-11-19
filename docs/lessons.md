# Development lessons

- When a Python script is run by path from a subdirectory, explicitly add the repository root to `sys.path` (or invoke it as a module). Python's script path is the subdirectory, so imports from the project root can otherwise fail.
- When testing a CLI `main()` from a test runner, pass an explicit argument list such as `[]`; otherwise it may parse the test runner's command-line flags.
