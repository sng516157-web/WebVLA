import sys
from absl import app, flags
from ml_collections import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    "/mnt/c/Users/sng51/Desktop/WebVLA/setup/finetune_libero_config.py",
    "config",
    lock_config=False,
)


def main(_):
    c = FLAGS.config
    print("num_steps:", c.num_steps)
    print("lr dict:", dict(c.optimizer.learning_rate))
    print("optimizer:", dict(c.optimizer))


if __name__ == "__main__":
    app.run(main)
