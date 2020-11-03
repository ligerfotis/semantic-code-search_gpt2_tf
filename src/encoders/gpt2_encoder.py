from typing import Dict, Any
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2Model, GPT2Config, TrainingArguments, Trainer, TFGPT2LMHeadModel

# from .utils.bert_self_attention import BertConfig, BertModel
from .masked_seq_encoder import MaskedSeqEncoder
from utils.tfutils import pool_sequence_embedding


# from transformers import GPT2Config

class GPT2Encoder(MaskedSeqEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        config = GPT2Config()

        encoder_hypers = {'self_attention_activation': config.activation_function,
                          'self_attention_hidden_size': config.hidden_size,
                          'self_attention_intermediate_size': config.hidden_size,
                          'self_attention_num_layers': config.n_layer,
                          'self_attention_num_heads': config.n_head,
                          'self_attention_pool_mode': 'weighted_mean',
                          'token_vocab_size': config.vocab_size,
                          'token_vocab_count_threshold': config.n_positions,
                          'token_embedding_size': config.hidden_size,

                          'use_subtokens': False,
                          'mark_subtoken_end': False,

                          'max_num_tokens': config.n_positions,

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

    def make_model(self, is_train: bool = False):
        with tf.compat.v1.variable_scope("gpt2_encoder"):
            self._make_placeholders()
            """
            GPT-2 uses Transformer's decoder as a building block, excluding the encoder-decoder attention module.
            Thus, the only difference with Bert's building blocks(Transformer's encoder) is the masked attention.
            However, in this implementation the masked attention is used for the BertEncoder.
            Therefore the BertModel will be used and adjust the hyper-parameters to be the same of those of the
            pretrained GPT-2 models.
            """
            config = GPT2Config()

            # print(self.placeholders['tokens'])
            # print(self.placeholders['tokens_mask'])

            model = TFGPT2Model.from_pretrained('gpt2', config=config)

            # outputs = model(input_ids=self.placeholders['tokens'],
            #                 attention_mask=self.placeholders['tokens_mask'])
            output = model.call(self.placeholders['tokens'])

            # tokenizer = GPT2Tokenizer.from_pretrained('bert-base-uncased')

            seq_token_embeddings = output[0]
            # Tensor("query_encoder/self_attention_encoder/bert/encoder/Reshape_4:0", shape=(?, 30, 128), dtype=float32)
            seq_token_masks = self.placeholders['tokens_mask']
            # Tensor("query_encoder/self_attention_encoder/tokens_mask:0", shape=(?, 30), dtype=float32)
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
