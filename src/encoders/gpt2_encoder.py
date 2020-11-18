import os
from typing import Dict, Any
from transformers import GPT2Tokenizer, TFGPT2Model, GPT2Config
import tensorflow as tf
# from .utils.bert_self_attention import BertConfig, BertModel
from .masked_seq_encoder import MaskedSeqEncoder
from utils.tfutils import pool_sequence_embedding


# from transformers import GPT2Config

class GPT2Encoder(MaskedSeqEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:

        encoder_hypers = {'self_attention_pool_mode': 'weighted_mean',
                          'use_subtokens': False,
                          'mark_subtoken_end': False,

                          'use_bpe': True,
                          'pct_bpe': 0.5
                          }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    @property
    def output_representation_size(self):
        return self.get_hyper('self_attention_hidden_size')

    def make_model(self, is_train: bool = False, name="default"):
        # with tf.compat.v1.variable_scope("gpt2_encoder_" + name):
        self._make_placeholders()
        """
        GPT-2 uses Transformer's decoder as a building block, excluding the encoder-decoder attention module.
        Thus, the only difference with Bert's building blocks(Transformer's encoder) is the masked attention.
        However, in this implementation the masked attention is used for the BertEncoder.
        Therefore the BertModel will be used and adjust the hyper-parameters to be the same of those of the
        pretrained GPT-2 models.
        """
        # print(self.placeholders['tokens'])
        # print(self.placeholders['tokens_mask'])
        cache_dir = "../resources/hugging_face/gpt2/"
        model = TFGPT2Model.from_pretrained('gpt2', cache_dir=cache_dir, return_dict=True)
        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # model = TFGPT2Model.from_pretrained('gpt2', return_dict=True)
        # model = tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # model = TFGPT2Model.from_pretrained('gpt2', return_dict=True)
        # outputs = model(input_ids=self.placeholders['tokens'],
        #                 attention_mask=self.placeholders['tokens_mask'])
        output = model(self.placeholders['tokens'], training=True)

        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        seq_token_embeddings = output.last_hidden_state

        seq_token_masks = self.placeholders['tokens_mask']
        seq_token_lengths = tf.reduce_sum(input_tensor=seq_token_masks, axis=1)  # B
        return pool_sequence_embedding("weighted_mean",
                                       sequence_token_embeddings=seq_token_embeddings,
                                       sequence_lengths=seq_token_lengths,
                                       sequence_token_masks=seq_token_masks)

    # def loss(self):
    #     loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #     per_sample_loss = per_sample_loss * self.placeholders['sample_loss_weights']
    #     tf.reduce_sum(input_tensor=per_sample_loss) / tf.reduce_sum(
    #         input_tensor=self.placeholders['sample_loss_weights'])
    #     return loss
    #
    # def train(self, train_dataset):
    #     model = self.make_model(is_train=True)
    #     loss = self.loss()
    #
    #     model.compile(loss=loss)
    #     model.fit(train_dataset, epochs=2, steps_per_epoch=115)
    #     model.save_pretrained('/gpt2_csnet/')

    # def train(self, model, train_dataset, test_dataset, data_collator):
    #     training_args = TrainingArguments(
    #         output_dir="./gpt2-csnet",  # The output directory
    #         overwrite_output_dir=True,  # overwrite the content of the output directory
    #         num_train_epochs=3,  # number of training epochs
    #         per_device_train_batch_size=32,  # batch size for training
    #         per_device_eval_batch_size=64,  # batch size for evaluation
    #         eval_steps=400,  # Number of update steps between two evaluations.
    #         save_steps=800,  # after # steps model is saved
    #         warmup_steps=500,  # number of warmup steps for learning rate scheduler
    #     )
    #     trainer = Trainer(
    #         model=model,
    #         args=training_args,
    #         data_collator=data_collator,
    #         train_dataset=train_dataset,
    #         eval_dataset=test_dataset,
    #         prediction_loss_only=True,
    #     )
    #     trainer.train()
    #     trainer.save_model()

