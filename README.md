# SMART_MENTOR

## Prepare the environment
```shell
pip install -r requirements.txt
```

### How to run config 
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

### Running tutor to execute a single scenario and a question.
```shell
python -m smart_mentor.tutor
```

### Running observability
#### Creating a vectordb
```shell
python -m smart_mentor.observability.vectordb_creator
```

#### Running Evaluation each scenarios and all questions selected
```shell
python -m smart_mentor.evaluation.evaluation_models
```

#### Running Evaluation harmonic meean
```shell
python -m smart_mentor.evaluation.evaluation_graphs
```

### Structure of folder
```
smart_mentor                            # Root directory
├── smart_mentor/                       # Source code directory
|   ├── config/                         # Configuration files
|   |   ├── config_helper.py            # Configuration helper
|   |   └── logging_config.py           # Logging configuration
|   ├── evaluation/                     # Evaluation-related modules
|   |   ├── evaluation_models.py        # Evaluate models for scenarios/questions
|   |   ├── evaluation_graphs.py        # Generate evaluation graphs
|   ├── file/                           # File handling modules
|   |   ├── smart_reader.py             # Smart reader for file handling
|   |   └── smart_writer.py             # Smart writer for file handling
|   ├── images/                         # Directory for images
|   ├── lib/                            # Library modules
|   |   └── core.py                     # Core abstract classes
|   ├── model_objects/                  # Model objects of LLMs config
|   |   └── model_objects.py            # Model objects for LLMs
|   ├── models_api/                     # Model API modules
|   |   ├── llama/                      # LLaMA API modules
|   |   |   └── client_llamaAi.py       # LLaMA client
|   |   ├── openai/                     # OpenAI API modules
|   |   |   ├── client_openAi.py        # OpenAI client
|   |   |   └── embedding_openAi.py     # OpenAI embedding
|   ├── observability/                  # Observability modules
|   |   ├── bert_similarity.py          # BERT similarity
|   |   ├── codet5_similarity.py        # CodeT5 similarity
|   |   ├── rouge_eval.py               # ROUGE evaluation
|   |   └── vectordb_creator.py         # VectorDB creator
|   ├── prompts/                        # Prompt templates
|   |   ├── prompt_rar.py               # RAR prompt template
|   |   ├── prompt_response.py          # Response prompt template
|   |   ├── prompt_role.py              # Role prompt template
|   |   ├── prompt_self_verification.py # Self-verification prompt template
|   |   ├── prompt_skeleton_thought.py  # Skeleton thought prompt template
|   |   ├── prompt_zero_shot.py         # Zero-shot prompt
|   |   └── promptHandler.py            # Prompt handler
|   ├── rag/                            # RAG modules
|   |   └── retriever.py                # Retriever for RAG
|   ├── resources/                      # Resource files
|   |   ├── ground_truth_data.csv       # Ground truth data
|   |   └── random_numbers.csv          # Random numbers
|   ├── jupiter_notebook/               # Tools for Jupyter notebooks
|   |   ├── statistics_test.ipynb       # Jupyter notebook for statistics evaluation
|   |   ├── graphs_scenarios.ipynb      # Jupyter notebook for generating graphs
|   |   └── check_database_ifpb.ipynb   # Jupyter notebook for creating dataset
|   ├── vectordb/                       # VectorDB modules
|   |   ├── vectorDatabase.py           # VectorDB class
|   ├── tutor.py                        # Main script to execute scenarios/questions
|   └── orchestrator.py                 # Orchestrator for managing scenarios
├── .gitignore                          # Specifies ignored files for Git
├── README.md                           # Project overview
└── requirements.txt                    # Production dependencies
```

