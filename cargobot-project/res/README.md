# Important note on model files
We could not find a way to use local paths in model directives. So, if you want to refer to another file in a directive, do it with absolute paths like this:

```
- add_directives:
    file: file:///usr/cargobot/cargobot-project/res/warehouse.dmd.yaml
```

Make sure the directives you are writing also work correctly in our Docker build.
