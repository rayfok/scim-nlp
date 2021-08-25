import sys

sys.path.append("../../src")
from scienceparseplus.datasets import JSONTokenClassficationDataset
from scienceparseplus.modeling.tokenizer import JSONDatasetWithFontTokenizer, FontTokenizer
# Load the class specific models 
from scienceparseplus.modeling.layoutlm_font import LayoutLMForTokenClassification
from scienceparseplus.modeling.bert_font import BertForTokenClassification

from base_trainer import BaseTokenClassificationTrainer


class TokenClassificationWithFontEmbTrainer(BaseTokenClassificationTrainer):

    task_name = "token_classification" + "-" + "font_emb"
    # We use "-" to join two terms and "_" to split word 

    def create_argparser(self):

        parser = super().create_argparser()
        parser.add_argument("--font_vocab_path", type=str, default="../font/font-vocab.txt")

        return parser

    def load_model_tokenizer(self):
        font_tokenizer = FontTokenizer(self.args.font_vocab_path)
        tokenizer = JSONDatasetWithFontTokenizer(
            self.args.model_name, font_tokenizer, use_layout="layout" in self.args.model_name
        )

        model_path = self.determine_training_model_path()

        # This is the same as the original load_model_tokenizer in BaseTokenClassificationTrainer
        # But we reinitialize the models here to use the newly imported modules 

        if "layoutlm" in self.args.model_name.lower():
            model = LayoutLMForTokenClassification.from_pretrained(
                model_path, num_labels=self.num_labels
            )
        else:
            model = BertForTokenClassification.from_pretrained(
                model_path, num_labels=self.num_labels
            )

        self.model = model
        self.tokenizer = tokenizer

if __name__ == "__main__":
    trainer = TokenClassificationWithFontEmbTrainer() 
    # Remember to change this 
    
    args = trainer.create_argparser().parse_args()

    trainer.parse_args_and_load_config(args)
    
    trainer.run()