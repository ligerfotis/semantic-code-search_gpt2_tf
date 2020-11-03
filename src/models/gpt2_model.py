import os
from typing import Any, Dict, Optional, List, Tuple
import tensorflow as tf
from encoders import GPT2Encoder
from transformers import GPT2Config

from .model import Model
from dpu_utils.utils import RichPath
import numpy as np
import random
import wandb

LoadedSamples = Dict[str, List[Dict[str, Any]]]
SampleId = Tuple[str, int]


class GPT2Model(Model):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        # config = GPT2Config()
        hypers = {}
        for label in ["code", "query"]:
            hypers.update({f'{label}_{key}': value
                           for key, value in GPT2Encoder.get_default_hyperparameters().items()})
        model_hypers = {
            'learning_rate': 5e-4,
            'code_use_subtokens': False,
            'code_mark_subtoken_end': False,
            'batch_size': 450,
        }
        hypers.update(super().get_default_hyperparameters())
        hypers.update(model_hypers)
        return hypers

    def __init__(self,
                 hyperparameters: Dict[str, Any],
                 run_name: str = None,
                 model_save_dir: Optional[str] = None,
                 log_save_dir: Optional[str] = None):
        super().__init__(
            hyperparameters,
            code_encoder_type=GPT2Encoder,
            query_encoder_type=GPT2Encoder,
            run_name=run_name,
            model_save_dir=model_save_dir,
            log_save_dir=log_save_dir)

    # def make_model(self, is_train: bool):
    #     with self.__sess.graph.as_default():
    #         random.seed(self.hyperparameters['seed'])
    #         np.random.seed(self.hyperparameters['seed'])
    #         tf.compat.v1.set_random_seed(self.hyperparameters['seed'])
    #
    #         self._make_model(is_train=is_train)
    #         self._make_loss()
    #         if is_train:
    #             self._make_training_step()
    #             self.__summary_writer = tf.compat.v1.summary.FileWriter(self.__tensorboard_dir, self.__sess.graph)
    #
    #
    #
    # def train(self,
    #           train_data: LoadedSamples,
    #           valid_data: LoadedSamples,
    #           azure_info_path: Optional[str],
    #           quiet: bool = False,
    #           resume: bool = False) -> RichPath:
    #     model_path = RichPath.create(self.model_save_path, azure_info_path)
    #     with self.__sess.as_default():
    #         tf.compat.v1.set_random_seed(self.hyperparameters['seed'])
    #         train_data_per_lang_nums = {language: len(samples) for language, samples in train_data.items()}
    #         print('Training on %s samples.' % (
    #             ", ".join("%i %s" % (num, lang) for (lang, num) in train_data_per_lang_nums.items())))
    #         valid_data_per_lang_nums = {language: len(samples) for language, samples in valid_data.items()}
    #         print('Validating on %s samples.' % (
    #             ", ".join("%i %s" % (num, lang) for (lang, num) in valid_data_per_lang_nums.items())))
    #
    #         if resume:
    #             # Variables should have been restored.
    #             best_val_mrr_loss, best_val_mrr, _ = self.__run_epoch_in_batches(valid_data, "RESUME (valid)",
    #                                                                              is_train=False, quiet=quiet)
    #             self.train_log('Validation Loss on Resume: %.6f' % (best_val_mrr_loss,))
    #         else:
    #             init_op = tf.compat.v1.variables_initializer(
    #                 self.__sess.graph.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES))
    #             self.__sess.run(init_op)
    #             self.save(model_path)
    #             best_val_mrr = 0
    #         no_improvement_counter = 0
    #         epoch_number = 0
    #         while (epoch_number < self.hyperparameters['max_epochs']
    #                and no_improvement_counter < self.hyperparameters['patience']):
    #
    #             self.train_log('==== Epoch %i ====' % (epoch_number,))
    #
    #             # run training loop and log metrics
    #             train_loss, train_mrr, train_time = self.__run_epoch_in_batches(train_data,
    #                                                                             "%i (train)" % (epoch_number,),
    #                                                                             is_train=True,
    #                                                                             quiet=quiet)
    #             self.train_log(' Training Loss: %.6f' % (train_loss,))
    #
    #             # run validation calcs and log metrics
    #             val_loss, val_mrr, val_time = self.__run_epoch_in_batches(valid_data, "%i (valid)" % (epoch_number,),
    #                                                                       is_train=False,
    #                                                                       quiet=quiet)
    #             self.train_log(' Validation:  Loss: %.6f | MRR: %.6f' % (val_loss, val_mrr,))
    #
    #             log = {'epoch': epoch_number,
    #                    'train-loss': train_loss,
    #                    'train-mrr': train_mrr,
    #                    'train-time-sec': train_time,
    #                    'val-loss': val_loss,
    #                    'val-mrr': val_mrr,
    #                    'val-time-sec': val_time}
    #
    #             # log to wandb
    #             wandb.log(log)
    #
    #             # log to tensorboard
    #             for key in log:
    #                 if key != 'epoch':
    #                     self._log_tensorboard_scalar(tag=key,
    #                                                  value=log[key],
    #                                                  step=epoch_number)
    #
    #             #  log the final epoch number
    #             wandb.run.summary['epoch'] = epoch_number
    #
    #             if val_mrr > best_val_mrr:
    #                 # save the best val_mrr encountered
    #                 best_val_mrr_loss, best_val_mrr = val_loss, val_mrr
    #
    #                 wandb.run.summary['best_val_mrr_loss'] = best_val_mrr_loss
    #                 wandb.run.summary['best_val_mrr'] = val_mrr
    #                 wandb.run.summary['best_epoch'] = epoch_number
    #
    #                 no_improvement_counter = 0
    #                 self.save(model_path)
    #                 self.train_log("  Best result so far -- saved model as '%s'." % (model_path,))
    #             else:
    #                 # record epochs without improvement for early stopping
    #                 no_improvement_counter += 1
    #             epoch_number += 1
    #
    #         log_path = os.path.join(self.__log_save_dir,
    #                                 f'{self.run_name}.train_log')
    #         wandb.save(log_path)
    #         tf.io.write_graph(self.__sess.graph,
    #                           logdir=wandb.run.dir,
    #                           name=f'{self.run_name}-graph.pbtxt')
    #
    #     self.__summary_writer.close()
    #     return model_path
