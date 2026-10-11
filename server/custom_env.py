"""LiberoWrapper subclass that can override the task instruction per episode.

The policy is language-conditioned and binds the instruction once per episode
(run_episode_frames: policy.reset(obs["instruction"]) right after env.reset),
so swapping env.instruction + the reset observation's instruction after
super().reset() redirects the policy for the whole episode. The sim scene and
success checker stay those of the base task — which is why the UI must show
custom-instruction episodes as UNCHECKED rather than SUCCESS/FAILED.
"""

from sim.wrapper import LiberoWrapper


class CustomInstructionWrapper(LiberoWrapper):
    custom_instruction: str | None = None

    def reset(self, task_id, seed):
        obs = super().reset(task_id, seed)
        # _make_env refreshes self.instruction only when the task CHANGES, so a
        # custom job would leak its instruction into the next same-task job
        # unless we always restore the base task language here.
        instr = self.custom_instruction or self.task.language
        if instr != self.instruction:
            self.instruction = instr
            obs = dict(obs)
            obs["instruction"] = instr
        return obs
