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

1. Follow the instructions in the README.md of the scienceparseplus module. Specfically, make sure you:
    - download the model weights
    - create the docker image
    - start the service in a docker container.
2. Process your PDFs with scienceparseplus

    ```bash
    python src/run_spp.py --input_dir=<PATH TO PDFS>
    ```
