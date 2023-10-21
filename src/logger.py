from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from collections import deque


class SummaryWriterCallback(BaseCallback):

    def __init__(
            self,
            exp_name,
            model_save_dir,
            eval_env,
            eval_fn,
            discriminator,
            args
    ):
        super(SummaryWriterCallback, self).__init__()
        self.exp_name = exp_name
        self.model_save_dir = model_save_dir
        self.check_start = False
        self.done_reward_queue = deque([0] * 10, maxlen=10)
        self.best_model = None
        self.best_metric = - float("inf")
        self.eval_env = eval_env
        self.eval_fn = eval_fn
        self.discriminator = discriminator
        self.args = args

    def _on_training_start(self):
        self._log_freq = 5000
        self.next_logging_point = 0
        self.done_num = 1
        output_formats = self.logger.output_formats
        self.tb_formatter = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self):
        # regular logging
        if self.num_timesteps >= self._log_freq and self.num_timesteps >= self.next_logging_point:
            ASR, completed, DO, TO = self.eval_fn(self.eval_env, self.model, 50, self.discriminator, self.args)
            if ASR > 90:
                self.model.save(self.model_save_dir + f"{self.exp_name}_checkpoint_{self.next_logging_point}.zip")
            self.tb_formatter.writer.add_scalar("test/completed", completed, self.next_logging_point)
            self.tb_formatter.writer.add_scalar("test/ASR", ASR, self.next_logging_point)
            self.tb_formatter.writer.add_scalar("test/DO", DO, self.next_logging_point)
            self.tb_formatter.writer.add_scalar("test/TO", TO, self.next_logging_point)
            self.tb_formatter.writer.flush()
            self.next_logging_point += self._log_freq
