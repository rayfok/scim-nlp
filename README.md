## Data processing and NLP scripts for powering Scim: a skim reading interface

### Download datasets
- Download IUI pdfs

    ```bash
    ./data_scripts/fetch_iui_data.sh
    ```

- Download scisummnet papers

    ```bash
    ./data_scripts/fetch_scisumm_data.sh
    ```

### Run scienceparseplus to detect PDF tokens and their bounding boxes

1. Follow the instructions in the README.md of the `scienceparseplus` module. Specfically, make sure you:
    - download the model weights
    - create the docker image
    - start the service in a docker container.
2. Process your PDFs with scienceparseplus

    ```bash
    python src/run_spp.py --input_dir=<PATH TO PDFS>
    ```
### Run sequential sentence classifier to classify sentences into 5 broad rhetorical categories

1. Download the pretrained model

    ```
    cd sequential_sentence_classification
    gdown https://drive.google.com/uc?id=1bx9hl6AhQdQ6hId4-N4ENM-jt7cwpqi6
    ```

2. Altnertively, follow the README.md in the `sequential_sentence_classification` module to re-train the model.
3. Run the classifier

    ```bash
    python src/run_ssc.py \
        --path_to_model sequential_sentence_classification/model.tar.gz \
        --test_jsonl_file=<path to input JSONL file> \
        --output_file=<path to output JSON file>
    ```
