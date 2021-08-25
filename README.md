<div align="center">
  <img src=".github/SPP_logo.png" alt="Layout Parser Logo" width="35%">

*Better PDF Document Information Extraction with Layout-Awareness*

</div>

---

## Install 

1. Download the model weights:
    ```bash
    cd tools
    bash download_model_weights.sh
    ```

### Setting up using Docker:

1. Create the docker image:
    ```
    docker build -t scienceparseplus .
    ```

2. Start the docker container and it will start a layout detection service for PDF files:
    ```bash
    docker run -p 8080:8080 -ti scienceparseplus   
    ```
    Note: the detached mode might not be preferred in the case because it usually takes long 
    for running layout detection models for a whole PDF, and it might be helpful to watch the progress
    bar in the command line.
 
3. Send a post request with PDF files and get the layout data (possibly after a while). Here is an exemplar script:
    ```python
    import requests
    f = open("<path-to-your-pdf.pdf", 'rb')
    files = {"pdf_file": (f.name, f, "multipart/form-data")}
    r = requests.post('http://localhost:8080/detect', files=files)
    layout = r.json()['layout']
    ```
    And you can directly detect PDF layouts via its URL:
    ```python
    import requests
    r = requests.get("http://127.0.0.1:8080/detect/", 
                      params={"pdf_url":"https://arxiv.org/pdf/1601.00978.pdf"})
    layout = r.json()['layout']
    ```
    Please see below for a detailed explanation of the returned layout data. 
4. [NEW] We create a new service for using the CERMINE engine for parsing PDF layouts: it will return PDF regions/textlines/token annotations. 
    ```python
    import requests
    r = requests.get("http://127.0.0.1:8080/cermine/", 
                      params={"pdf_url":"https://arxiv.org/pdf/1601.00978.pdf"})
    layout = r.json()['layout']
    ```
    The output data structure is a bit different from the PDF detection service, please see below for a detailed explanation.

### Setup using pip

1. Install PDF utility tool: 
    - Mac users will have to install [poppler](http://macappstore.org/poppler/) for Mac.

2. Install the dependency
    ```bash
    pip install -e .
    pip install 'git+https://github.com/facebookresearch/detectron2.git#egg=detectron2' 
    ```


## Usage

1. Run inference

    ```bash
    cd tools 
    python predict_pdf_layout.py --filename example.pdf
    ```
    It will generate a JSON file with the following structure:
    ```
    [
        {
            "width": 612.0, # width of this page,
            "height": 792.0, # width of this page
            "index": 0, # page index
            "layout": [...] # layout data
        }, # information for a single page
        ...
    ]
    ```
    For layout data, it looks like: 
    ```
    {
        "bundles": [
            {
                "bbox": [90.0316162109375, 569.6625366210938, 524.3961791992188, 682.0439453125], #left, top, right, bottom coordinates
                "type": "paragraph", # Predicted type for the bundle
                "confidence": 0.9985978007316589,  # Type prediction confidence
                "tokens: [...] # A list of tokens contained inside this bundle 
            },
            ...
        ],
        "remaining_tokens": [...] # A list of tokens that are not inside any bundles  
    }
    ```
    Bundles are predicted region blocks + all the contained tokens inside the blocks. For tokens, they are stored in dictionaries like this: 
    ```
    {
        "bbox": [90.0, 570.125, 114.084, 581.034], #left, top, right, bottom coordinates
        "text": "Deep", # Corresponding text
        "type": "paragraph", # Predicted type for the token
        "confidence": 8.500739097595215 # Type prediction confidence
    }
    ```
2. When using CERMINE Engine to parsing the PDFs, it will return a similar JSON structure with few modifications:
    ```
    [
        {
            "width": 612.0, # width of this page,
            "height": 792.0, # width of this page
            "index": 0, # page index
            "layout": {...} # **layout data - changed**
        }, # information for a single page
        ...
    ]
    ```
    For layout data, it is now a dict with the following structure:
    ```
    {
        "layout": {
            "blocks": [...] # A list of region blocks, 
            "lines": [...] # A list of line blocks,
            "tokens": [...] # A list of token blocks,
        }
    }
    ```
    And for each of region/line/token block, it is saved in the following format: 
    ```
    {
        "id": 1, # The id of the region/line/token, sorted based on reading order
        "bbox": [90.0, 570.125, 114.084, 581.034], #left, top, right, bottom coordinates
        "text": "Deep", # Corresponding text. for region/line, it will the ' '.join(all tokens within this region/line)
        "type": "paragraph", # Predicted type for the region/line/token (from CERMINE)
        "parent_block_id": 2, # The id of the parent block (it will be -1 if the element itself is a region)
        "parent_line_id": 9, # The id of the parent line (it will be -1 if the element itself is a region/line)
    }
    ```
    So the content hierarchy is stored via the `parent_block_id` and `parent_line_id` field. And you can fetch 
    tokens within a specific line (e.g., 24) using the following (pseudo)code:
    ```python
    [ele for ele in layout['tokens'] if ele.parent_line_id == 24]
    ```