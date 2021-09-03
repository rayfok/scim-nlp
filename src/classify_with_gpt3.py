import os
import openai
import json
from paper import SPPPaper


def upload_labeled_data(training_file):
    res = openai.File.create(file=open(training_file), purpose="classifications")
    print(f"Uploaded {training_file}. Accessible with id {res.get('id', None)}")
    return res.id


def query_gpt3(query, gpt3_training_file):
    response = openai.Classification.create(
        file=gpt3_training_file, query=query, search_model="ada", model="curie", max_examples=5
    )
    return response


def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Uploading the training file only needs to happen once
    # upload_labeled_data(training_file="data/gpt3/contribution_train.jsonl")
    # return

    p = SPPPaper("data/spp-output/weisz_perfection-not-required.json")

    # Select sentences from the introduction only
    intro_sentences = p.sentences[:24]

    preds = []
    for sentence in intro_sentences:
        try:
            pred = query_gpt3(query=sentence, gpt3_training_file="file-ItTYR0mmPqonb4ViuKNk89Dz",)
        except openai.error.InvalidRequestError:
            pred = {}
        pred["input"] = sentence
        preds.append(pred)
    with open("pred.json", "w") as out:
        json.dump(preds, out, indent=2)


if __name__ == "__main__":
    main()
