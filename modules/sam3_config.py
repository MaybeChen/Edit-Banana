"""Configuration helpers for SAM3 prompt groups."""

import os
import yaml
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any

from prompts.arrow import ARROW_PROMPT
from prompts.background import BACKGROUND_PROMPT
from prompts.shape import SHAPE_PROMPT
from prompts.image import IMAGE_PROMPT

# ======================== 提示词分组枚举 ========================
class PromptGroup(Enum):
    """提示词分组"""
    IMAGE = "image"          # 图片类（需要转base64）
    ARROW = "arrow"          # 箭头类（需要方向检测）
    BASIC_SHAPE = "shape"    # 基本图形（需要取色矢量化）
    BACKGROUND = "background"  # 背景/容器类


@dataclass
class PromptGroupConfig:
    """提示词组配置"""
    name: str                           # 组名
    prompts: List[str] = field(default_factory=list)  # 该组的提示词
    score_threshold: float = 0.5        # 置信度阈值
    min_area: int = 100                 # 最小面积
    priority: int = 1                   # 去重优先级（越高越优先保留）
    description: str = ""               # 描述
    
    def add_prompt(self, prompt: str):
        """添加提示词"""
        if prompt not in self.prompts:
            self.prompts.append(prompt)
    
    def remove_prompt(self, prompt: str):
        """移除提示词"""
        if prompt in self.prompts:
            self.prompts.remove(prompt)


# ======================== 配置加载器 ========================
class ConfigLoader:
    """从config.yaml加载词组配置"""
    
    _instance = None
    _config = None
    _config_path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_config_path(cls) -> str:
        """获取配置文件路径"""
        if cls._config_path is None:
            cls._config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config", "config.yaml"
            )
        return cls._config_path
    
    @classmethod
    def load_config(cls, force_reload: bool = False) -> dict:
        """加载配置文件"""
        if cls._config is None or force_reload:
            config_path = cls.get_config_path()
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    cls._config = yaml.safe_load(f)
            else:
                print(f"[ConfigLoader] Config not found: {config_path}, using defaults")
                cls._config = cls._get_default_config()
        return cls._config
    
    @classmethod
    def _get_default_config(cls) -> dict:
        """获取默认配置（当config.yaml不存在时使用）"""
        return {
            'sam3': {
                'checkpoint_path': '',
                'bpe_path': '',
                'use_vlm_prompts': True,
                'vlm_prompt_max_per_group': 6,
            },
            'prompt_groups': {
                'image': {
                    'name': '图片类',
                    'prompts': ['icon', 'picture', 'logo', 'chart'],
                    'score_threshold': 0.5,
                    'min_area': 100,
                    'priority': 2,
                },
                'arrow': {
                    'name': '箭头类',
                    'prompts': ['arrow', 'line', 'connector'],
                    'score_threshold': 0.40,
                    'min_area': 40,
                    'priority': 4,
                },
                'shape': {
                    'name': '基本图形',
                    'prompts': ['rectangle', 'rounded rectangle', 'diamond', 'ellipse'],
                    'score_threshold': 0.5,
                    'min_area': 150,
                    'priority': 3,
                },
                'background': {
                    'name': '背景容器',
                    'prompts': ['section_panel', 'title bar', 'container'],
                    'score_threshold': 0.25,  # 降低阈值以检测更多背景色块
                    'min_area': 400,
                    'priority': 1,
                },
            },
            'text_filter': {
                'blacklist': ['text', 'word', 'label'],
                'keywords': ['text', 'word'],
            },
            'deduplication': {
                'iou_threshold': 0.7,
                'arrow_iou_threshold': 0.85,
            },
        }
    
    @staticmethod
    def _merge_prompts(*prompt_lists: List[str]) -> List[str]:
        """Merge prompt lists while preserving order and removing duplicates."""
        merged = []
        seen = set()
        for prompts in prompt_lists:
            for prompt in prompts or []:
                prompt = str(prompt).strip()
                if not prompt or prompt in seen:
                    continue
                seen.add(prompt)
                merged.append(prompt)
        return merged

    @classmethod
    def get_prompt_groups(cls) -> Dict[PromptGroup, PromptGroupConfig]:
        """从配置文件加载词组配置"""
        config = cls.load_config()
        prompt_groups_config = config.get('prompt_groups', {})
        
        result = {}
        
        # 映射配置键到枚举
        key_to_enum = {
            'image': PromptGroup.IMAGE,
            'arrow': PromptGroup.ARROW,
            'shape': PromptGroup.BASIC_SHAPE,
            'background': PromptGroup.BACKGROUND,
        }
        prompt_mapping = {
            'image': IMAGE_PROMPT,
            'arrow': ARROW_PROMPT,
            'shape': SHAPE_PROMPT,
            'background': BACKGROUND_PROMPT,
        }
        
        
        for key, enum_val in key_to_enum.items():
            if key in prompt_groups_config:
                group_cfg = prompt_groups_config.get(key, {})
                # Prompt text defaults live in prompts/*.py. Config can add
                # project-specific prompts without losing the broad defaults that
                # preserve recall; set replace_default_prompts: true to replace.
                default_prompts = list(prompt_mapping.get(key, []))
                configured_prompts = list(group_cfg.get('prompts') or [])
                extra_prompts = list(group_cfg.get('extra_prompts') or [])
                if group_cfg.get('replace_default_prompts'):
                    prompts = configured_prompts or default_prompts
                else:
                    prompts = cls._merge_prompts(default_prompts, configured_prompts, extra_prompts)
                # 从config.yaml读取其他配置（阈值、面积、优先级等）
                result[enum_val] = PromptGroupConfig(
                    name=group_cfg.get('name', key),
                    prompts=prompts,
                    score_threshold=group_cfg.get('score_threshold', 0.5),
                    min_area=group_cfg.get('min_area', 100),
                    priority=group_cfg.get('priority', 1),
                    description=group_cfg.get('description', ''),
                )
        
        return result
    
    @classmethod
    def get_text_filter(cls) -> dict:
        """获取文字过滤配置"""
        config = cls.load_config()
        return config.get('text_filter', {'blacklist': [], 'keywords': []})
    
    @classmethod
    def get_deduplication_config(cls) -> dict:
        """获取去重配置"""
        config = cls.load_config()
        return config.get('deduplication', {
            'iou_threshold': 0.7,
            'arrow_iou_threshold': 0.85,
        })
    
    @classmethod
    def get_drawio_styles(cls) -> dict:
        """获取DrawIO样式配置"""
        config = cls.load_config()
        return config.get('drawio_styles', {})
    
    @classmethod
    def get_sam3_config(cls) -> dict:
        """获取SAM3配置"""
        config = cls.load_config()
        return config.get('sam3', {})

    @classmethod
    def get_multimodal_config(cls) -> dict:
        """获取多模态/VLM配置"""
        config = cls.load_config()
        return config.get('multimodal', {})
