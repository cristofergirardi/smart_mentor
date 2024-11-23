# SMART_MENTOR

To run the config_helper alone or independently add 
```shell
if __name__ == "__main__":
    config = ConfigHelper()
    print(config.settings)
```

It goes to terminal and run
```shell
python -m smart_mentor.config.config_helper
```

### Running observability
#### Creating a vectordb
```shell
python -m smart_mentor.observability.vectordb_creator
```