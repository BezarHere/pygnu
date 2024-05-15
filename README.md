# pygnu
Project manager &amp; builder for GCC

*please star pygnu, it really help*

## arguments

### new

creates a new project is the working directory if not other directory path is given

### build

builds the project in the working or directory (`pygnu.json`)  if not other directory path is given

### help

the `help` initiates a video call with *alex & teather*, also shows the descriptions and uses of other commands/flags

## multiple build configurations

you can specify the build configuration ('mode') by passing -M\<build config name> (a single argument, can use quots for spaces)
not specifying the build mode will default to `debug`; if there is no build mode of name `debug`, an error message will be displayed

## pygnu.json?

the project metadata as a json file (subject to change)
self documented, easy to understand:
```json
{
    "output_dir": ".",
    "output_cache_dir": ".",
    "output_name": "output",
    "source_selectors": [
        "**/*.c",
        "**/*.cpp",
        "**/*.cc",
        "**/*.cxx"
    ],
    "build_configurations": {
        "debug": {
            "predefines": {
                "_DEBUG": null
            },
            "optimization_lvl": 2,
            "optimization_type": 2,
            "standard": "c17",
            "warning_level": 2,
            "warning_pedantic": false,
            "print_includes": false,
            "catch_typos": true,
            "exit_on_errors": true,
            "dynamicly_linkable": true,
            "print_stats": true,
            "simd_type": "sse",
            "include_dirs": [],
            "lib_dirs": [],
            "lib_names": [],
            "assempler_args": [],
            "linker_args": [],
            "preprocessor_args": []
        },
        "release": {
            "predefines": {
                "NDEBUG": null,
                "_RELEASE": null
            },
            "optimization_lvl": 4,
            "optimization_type": 4,
            "standard": "c17",
            "warning_level": 2,
            "warning_pedantic": false,
            "print_includes": false,
            "catch_typos": true,
            "exit_on_errors": true,
            "dynamicly_linkable": true,
            "print_stats": true,
            "simd_type": "sse",
            "include_dirs": [],
            "lib_dirs": [],
            "lib_names": [],
            "assempler_args": [],
            "linker_args": [],
            "preprocessor_args": []
        }
    }
}
```

### NOTES

1. pygnu is in early development, so you might see a bug here and there
2. C/C++ are the most supported, others can build but it might be hassle (btw c++ is still broken)

---

*masha'a alah*
