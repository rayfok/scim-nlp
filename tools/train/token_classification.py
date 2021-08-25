from base_trainer import BaseTokenClassificationTrainer

if __name__ == "__main__":
    trainer = BaseTokenClassificationTrainer()
    
    args = trainer.create_argparser().parse_args()

    trainer.parse_args_and_load_config(args)
    
    trainer.run()