import logging
import os
from typing import Any, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import abc
import logging
import os
import pickle
from collections import deque
from typing import Any

import einops
import torch
import torch.nn as nn
import wandb
from MambaVLA.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler

log = logging.getLogger(__name__)




from MambaVLA.utils.scaler import ActionScaler, MinMaxScaler, Scaler



class MambaVLA(nn.Module):
    def __init__(
        self,
        model: Any,
        obs_encoders: Any,
        language_encoders: Any,
        optimizer: Any,
        lr_scheduler: Any,
        action_dim: int,
        perception_seq_len: int,
        action_seq_len: int,
        cam_names: list[str],
        use_lr_scheduler: bool = True,
        consider_robot_states: bool = False,
        if_film_condition: bool = False,
        device: str = "cpu",
        state_dim: int = 7,
        latent_dim: int = 64,
        sampling_steps: int = 50,
        use_uncertainty: bool = False,
        lambda_u: float = 1.0,
        lambda_u_warmup_epochs: int = 5,
        lambda_u_target: Optional[float] = None,
        refinement_steps: int = 0,
        uncertainty_threshold: float = 0.0,
    ):
        super().__init__()

        self.device = device
        self.working_dir = os.getcwd()
        self.scaler = None

        # Initialize model and encoders
        self.img_encoder = obs_encoders.to(device)
        self.language_encoder = language_encoders.to(device)
        self.model = model.to(device)
        self.state_emb = nn.Linear(state_dim, latent_dim)

        self.cam_names = cam_names

        # for inference
        self.rollout_step_counter = 0
        self.action_seq_len = action_seq_len
        self.perception_seq_len = perception_seq_len

        self.obs_seq: dict[str, deque[torch.Tensor]] = {}
        self._last_uncertainty_seq: Optional[torch.Tensor] = None
        self._last_refinement_stats: Optional[dict[str, list[float]]] = None

        self.action_dim = action_dim

        self.consider_robot_states = consider_robot_states
        self.if_film_condition = if_film_condition


        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler

        self.use_lr_scheduler = use_lr_scheduler

        self.sampling_steps = sampling_steps
        self.use_uncertainty = use_uncertainty
        self.lambda_u = lambda_u
        self.lambda_u_warmup_epochs = lambda_u_warmup_epochs
        self.lambda_u_target = lambda_u if lambda_u_target is None else lambda_u_target
        self.refinement_steps = refinement_steps
        self.uncertainty_threshold = uncertainty_threshold

    def set_scaler(self, scaler):
        self.scaler = scaler

    def _input_embeddings(self, obs_dict):

        if "lang" in obs_dict:
            obs_dict["lang_emb"] = self.language_encoder(obs_dict["lang"]).float()

        lang_embed = obs_dict["lang_emb"]

        if self.cam_names is not None:

            first_cam_key = f"{self.cam_names[0]}_image"
            first_cam_tensor = obs_dict[first_cam_key]
            if first_cam_tensor.ndim == 4:
                first_cam_tensor = first_cam_tensor.unsqueeze(1)
                obs_dict[first_cam_key] = first_cam_tensor
            elif first_cam_tensor.ndim != 5:
                raise ValueError(
                    f"Expected {first_cam_key} to have 4 or 5 dimensions, "
                    f"got {first_cam_tensor.ndim}."
                )

            B, T, C, H, W = obs_dict[first_cam_key].shape

            for camera in self.cam_names:
                camera_key = f"{camera}_image"
                camera_tensor = obs_dict[camera_key]
                if camera_tensor.ndim == 4:
                    camera_tensor = camera_tensor.unsqueeze(1)
                elif camera_tensor.ndim != 5:
                    raise ValueError(
                        f"Expected {camera_key} to have 4 or 5 dimensions, "
                        f"got {camera_tensor.ndim}."
                    )
                obs_dict[camera_key] = camera_tensor.view(B * T, C, H, W)

            if self.if_film_condition:
                obs_embed = self.img_encoder(obs_dict, lang_embed)
            else:
                obs_embed = self.img_encoder(obs_dict)
        else:
            raise NotImplementedError("Either use point clouds or images as input.")

        if self.consider_robot_states and "robot_states" in obs_dict.keys():
            robot_states = obs_dict["robot_states"]
            robot_states = self.state_emb(robot_states)

            obs_embed = torch.cat([obs_embed, robot_states], dim=1)

        return obs_embed, lang_embed

    def reset(self):
        self.rollout_step_counter = 0
        self.obs_seq: dict[str, deque[torch.Tensor]] = {}

    @torch.no_grad()
    def predict(
        self,
        obs_dict: dict[str, torch.Tensor],
        refinement_steps: Optional[int] = None,
        return_diagnostics: bool = False,
        collect_refinement_stats: bool = False,
    ) -> torch.Tensor:
        if not self.obs_seq:
            for key in obs_dict.keys():
                self.obs_seq[key] = deque(maxlen=self.perception_seq_len)

        for key in obs_dict.keys():
            if key.endswith("_image") and obs_dict[key].ndim == 4:
                obs_dict[key] = obs_dict[key].unsqueeze(1)
            elif key == "robot_states" and obs_dict[key].ndim == 2:
                obs_dict[key] = obs_dict[key].unsqueeze(1)
            self.obs_seq[key].append(obs_dict[key])

            if key == "lang":
                continue
            obs_dict[key] = torch.concat(list(self.obs_seq[key]), dim=1)

            if obs_dict[key].shape[1] < self.perception_seq_len:
                pad = einops.repeat(
                    obs_dict[key][:, 0],
                    "b ... -> b t ...",
                    t=self.perception_seq_len - obs_dict[key].shape[1],
                )
                obs_dict[key] = torch.cat([pad, obs_dict[key]], dim=1)
                
        if self.rollout_step_counter == 0:
            self.eval()

            pred_action_seq = self(obs_dict)[:, :self.action_seq_len]
            if refinement_steps is None:
                refinement_steps = self.refinement_steps
            if refinement_steps and refinement_steps > 0:
                obs_embed, lang_embed = self._input_embeddings(obs_dict)
                pred_action_seq = self._refine_action_sequence(
                    pred_action_seq,
                    obs_embed,
                    lang_embed,
                    refinement_steps,
                    self.uncertainty_threshold,
                    collect_refinement_stats=collect_refinement_stats,
                )
                self._last_refinement_stats = getattr(self, "_last_refinement_stats", None)
            elif collect_refinement_stats:
                self._last_refinement_stats = None
            if return_diagnostics:
                obs_embed, lang_embed = self._input_embeddings(obs_dict)
                time_steps = torch.zeros((pred_action_seq.size(0),), device=pred_action_seq.device)
                _, s_hat = self.model.predict_velocity_and_uncertainty(
                    pred_action_seq, obs_embed, lang_embed, time_steps
                )
                self._last_uncertainty_seq = s_hat.detach()
            pred_action_seq = self.scaler.inverse_scale_output(pred_action_seq)
            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[0, self.rollout_step_counter]
        diagnostics = None
        if return_diagnostics and self._last_uncertainty_seq is not None:
            diagnostics = {
                "s_hat": self._last_uncertainty_seq[0, self.rollout_step_counter],
                "refinement_stats": self._last_refinement_stats,
            }

        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.action_seq_len:
            self.rollout_step_counter = 0

        if return_diagnostics:
            return current_action, diagnostics
        return current_action

    def _refine_action_sequence(
        self,
        action_seq: torch.Tensor,
        obs_embed: torch.Tensor,
        lang_embed: torch.Tensor,
        refinement_steps: int,
        uncertainty_threshold: float,
        *,
        collect_refinement_stats: bool = False,
    ) -> torch.Tensor:
        refined_actions = action_seq
        batch_size = refined_actions.size(0)
        step_size = 0.5 / refinement_steps
        time_steps = torch.zeros((batch_size,), device=refined_actions.device)
        refinement_stats = {"mean_s_hat": [], "mean_residual": []} if collect_refinement_stats else None
        for _ in range(refinement_steps):
            v_hat, s_hat = self.model.predict_velocity_and_uncertainty(
                refined_actions, obs_embed, lang_embed, time_steps
            )
            if refinement_stats is not None:
                refinement_stats["mean_s_hat"].append(float(s_hat.mean().detach()))
            mask = s_hat > uncertainty_threshold
            correction = step_size * v_hat
            refined_actions = torch.where(mask, refined_actions - correction, refined_actions)
        if refinement_stats is not None:
            self._last_refinement_stats = refinement_stats
        return refined_actions

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        path = os.path.join(
            weights_path,
            "model_state_dict.pth" if sv_name is None else f"{sv_name}.pth",
        )
        self.load_state_dict(torch.load(path, weights_only=True))
        log.info("Loaded pre-trained model")

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        os.makedirs(store_path, exist_ok=True)
        path = os.path.join(
            store_path, "model_state_dict.pth" if sv_name is None else f"{sv_name}.pth"
        )
        torch.save(self.state_dict(), path)
        log.info(f"Model saved to: {store_path}")

    def store_model_scaler(self, store_path: str, sv_name=None) -> None:
        save_path = os.path.join(
            store_path, "model_scaler.pkl" if sv_name is None else sv_name
        )
        with open(save_path, "wb") as f:
            pickle.dump(self.scaler, f)
        log.info(f"Model scaler saved to: {save_path}")

    def load_model_scaler(self, weights_path: str, sv_name=None) -> None:
        if sv_name is None:
            sv_name = "model_scaler.pkl"

        with open(os.path.join(weights_path, sv_name), "rb") as f:
            self.scaler = pickle.load(f)
        log.info("Loaded model scaler")

    def get_params(self):

        total_params = sum(p.numel() for p in self.parameters())

        if wandb.run is not None:
            wandb.log({"model parameters": total_params})

        log.info("The model has a total amount of {} parameters".format(total_params))

    @property
    def get_model_state_dict(self) -> dict:
        return self.state_dict()

    @property
    def get_scaler(self) -> Scaler:
        if self.scaler is None:
            raise AttributeError("Scaler has not been set. Use set_scaler() first.")
        return self.scaler

    @property
    def get_model_state(self) -> tuple[dict, Scaler]:
        if self.scaler is None:
            raise AttributeError("Scaler has not been set. Use set_scaler() first.")
        return (self.state_dict(), self.get_scaler)

    def recover_model_state(self, model_state, scaler):
        self.load_state_dict(model_state)
        self.set_scaler(scaler)

    def configure_optimizers(self):
        """
        Initialize optimizers and learning rate schedulers based on model configuration.
        """
        optim_groups = [
            {"params": self.model.model.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
        ]

        optim_groups.extend([
            {"params": self.img_encoder.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
        ])


        optimizer = torch.optim.AdamW(optim_groups, lr=self.optimizer_config.learning_rate,
                                      betas=self.optimizer_config.betas)

        # Optionally initialize the scheduler
        if self.use_lr_scheduler:
            scheduler = TriStageLRScheduler(optimizer, self.lr_scheduler)

            return optimizer, scheduler

        else:
            return optimizer

    def forward(self, obs_dict, actions=None):

        # with torch.no_grad():
        obs_embed, lang_embed = self._input_embeddings(obs_dict)

        if self.training and actions is not None:
            v_hat, s_hat, v_target, time_loss_pairs = self.model(actions, obs_embed, lang_embed)
            return {
                "v_hat": v_hat,
                "s_hat": s_hat,
                "v_target": v_target,
                "time_loss_pairs": time_loss_pairs,
            }

        noise_actions = torch.randn((len(obs_embed), self.action_seq_len, self.action_dim), device=self.device)
        pred_act_seq = self.model.generate_actions(noise_actions, obs_embed, lang_embed, sample_steps=self.sampling_steps)

        return pred_act_seq
