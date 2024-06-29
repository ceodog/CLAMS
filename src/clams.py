from functools import partial
import logging
import torch
from transformers.trainer_callback import EarlyStoppingCallback
from transformers import (AutoTokenizer, VisionEncoderDecoderModel,
                    Seq2SeqTrainer, Seq2SeqTrainingArguments,
                    default_data_collator)


def create_clams_model(config, pretrained_vit_dir, pretrained_decodert_dir,
                tokenizer, device=torch.device('cpu')):

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        pretrained_vit_dir, pretrained_decodert_dir,
        tie_encoder_decoder=True).to(device)

    
    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # set beam search parameters
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.max_length = config['max_length']
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 0
    model.config.length_penalty = 1.0
    model.config.num_beams = config['num_beams']
    
    return model


def train_clams_model(model, config, ic_train_set, ic_val_set):
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['model_dir'],
        logging_dir=config['model_dir'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        predict_with_generate=True,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        do_train=True,
        do_eval=True,
        num_train_epochs = config['num_train_epochs'],
        overwrite_output_dir=True,
        save_total_limit=config['save_total_limit'],
        load_best_model_at_end=True,
    )
    
    early_stopping = EarlyStoppingCallback(
                early_stopping_patience=config['early_stopping_patience'])
    
    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ic_train_set,
        eval_dataset=ic_val_set,
        data_collator=default_data_collator,
        callbacks=[early_stopping],
    )
    
    trainer.train()
    trainer.save_model(config['model_dir'])