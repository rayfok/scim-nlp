# The dataset schema

We convert the original dataset into the target formats to make it more efficient to train the models.

## Token Table

A csv table for each page of document, acted as an intermediate level representation and condensing all representing from different datasets. There are several columns: 

| Feature  | Description                                                                            |
| -------- | -------------------------------------------------------------------------------------- |
| id       | The block/line/token ids (they are separately indexed)                                 |
| x_1      | The right coordinate of the given element                                              |
| y_1      | The top coordinate of the given element                                                |
| x_2      | The left coordinate of the given element                                               |
| y_2      | The bottom coordinate of the given element                                             |
| text     | The containing text within each element, whether a block/line of word or a single word |
| category | The category id                                                                        |
| block_id | The id of the belonging block for the element, -1 for blocks                           |
| line_id  | The id of the belonging line for the element, -1 for blocks/lines                      |
| is_block | Whether the element is a block element                                                 |
| is_line  | Whether the element is a line element                                                  |

## Token Level information JSON 

Specified in `./schema-token.json`, with only three files (for training, testing, dev) for each dataset. 