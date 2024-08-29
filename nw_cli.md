# New CLI `fdbiomed` command


## Usage

Configuration file is the main asset to have to be able to execute node or researcher component the path of the config file is always considered as root of fedbiomed. The folder structure

```
fediomed-node-1
|__ config.ini
|__ var
    |__ ....
|__db.json

```

### Executing fedbiomed command without config file

If Fed-BioMed commands (e.g. `fedbiomed node`, `fedbiomed researcher`) are executed without providing a configuration path, the command will look into the directory where it is executed. If there is the root directory of any fedbiomed component it will use the `config.ini` defined in the directory. Otherwise, it will create `fedbiomed` component initial file and folders.

### Running `fedbiomed` command from anywhere else in the filesystem


The `-p`, `--path` or `-c`, `--config` is the path to declare the component that command will be executed for. If there is no coomponent instantiated in the given directory/path a new one with default configuration will be generated.

`fedbiomed node -c <path-to-component> start`


