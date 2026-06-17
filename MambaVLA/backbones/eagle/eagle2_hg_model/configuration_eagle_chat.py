# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy

from transformers import LlamaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Eagle2ChatConfig(PretrainedConfig):
    model_type = "eagle_chat"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        select_layer=-1,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        mlp_checkpoint=True,
        pre_feature_reduction=False,
        keep_aspect_ratio=False,
        vocab_size=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing Vision Encoders with default values.")

        if llm_config is None:
            llm_config = {}
            logger.info("llm_config is None. Initializing the LLM config with default values")

        # Handle the case where llm_config is empty or doesn't have architectures
        if not llm_config or "architectures" not in llm_config:
            if not llm_config:
                # Create a minimal default llm config for Llama
                llm_config = {
                    "architectures": ["LlamaForCausalLM"],
                    "hidden_size": 2048,
                    "intermediate_size": 8192,
                    "num_hidden_layers": 24,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 32,
                    "vocab_size": 49164,
                    "max_position_embeddings": 8192,
                    "hidden_act": "silu",
                    "initializer_range": 0.02,
                    "rms_norm_eps": 1e-05,
                    "use_cache": True,
                    "torch_dtype": "bfloat16",
                    "rope_theta": 130000,
                    "attention_dropout": 0.0,
                    "bos_token_id": 1,
                    "eos_token_id": 2,
                    "pad_token_id": 2,
                }
            else:
                raise ValueError("llm_config is not empty but missing 'architectures' key")

        # Handle the case where vision_config is empty or doesn't have model_type
        if not vision_config or "model_type" not in vision_config:
            # If vision_config is empty or missing model_type, we can't proceed
            # This typically happens during to_dict() calls with default parameters
            if not vision_config:
                # Create a minimal default vision config for siglip
                vision_config = {
                    "model_type": "siglip_vision_model",
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "num_channels": 3,
                    "image_size": 224,
                    "patch_size": 14,
                    "hidden_act": "gelu_pytorch_tanh",
                    "layer_norm_eps": 1e-06,
                    "drop_path_rate": 0.1,
                    "attention_dropout": 0.0,
                    "initializer_range": 0.02,
                    "initializer_factor": 1.0,
                    "use_cache": True,
                }
            else:
                raise ValueError("vision_config is not empty but missing 'model_type' key")

        if vision_config["model_type"] == "siglip_vision_model":
            self.vision_config = SiglipVisionConfig(**vision_config)
        else:
            raise ValueError("Unsupported model_type: {}".format(vision_config["model_type"]))

        if llm_config["architectures"][0] == "LlamaForCausalLM":
            self.llm_config = LlamaConfig(**llm_config)
        else:
            raise ValueError("Unsupported architecture: {}".format(llm_config["architectures"][0]))
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.mlp_checkpoint = mlp_checkpoint
        self.pre_feature_reduction = pre_feature_reduction
        self.keep_aspect_ratio = keep_aspect_ratio
        self.vocab_size = self.llm_config.vocab_size
        logger.info(f"keep_aspect_ratio: {self.keep_aspect_ratio}")
        logger.info(f"vision_select_layer: {self.select_layer}")
        logger.info(f"min_dynamic_patch: {self.min_dynamic_patch}")
        logger.info(f"max_dynamic_patch: {self.max_dynamic_patch}")

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        output["model_type"] = self.__class__.model_type
        output["use_backbone_lora"] = self.use_backbone_lora
        output["use_llm_lora"] = self.use_llm_lora
        output["select_layer"] = self.select_layer
        output["force_image_size"] = self.force_image_size
        output["downsample_ratio"] = self.downsample_ratio
        output["template"] = self.template
        output["dynamic_image_size"] = self.dynamic_image_size
        output["use_thumbnail"] = self.use_thumbnail
        output["min_dynamic_patch"] = self.min_dynamic_patch
        output["max_dynamic_patch"] = self.max_dynamic_patch
        output["keep_aspect_ratio"] = self.keep_aspect_ratio

        return output
