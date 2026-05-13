# config.py - Configurație pentru Llama-3.2-3B-Instruct

import os
import torch
import logging

class Config:
    # Setări de bază
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'content-rating-super-secret-key-2025'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///ratings.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload settings
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max

    #HuggingFaceToken
    HUGGINGFACE_TOKEN = "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"
    # Model Configuration - Llama-3.2-3B-Instruct
    LLM_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
    LLM_CACHE_DIR = "./model_cache"
    
    # Model settings pentru Llama-3.2-3B - FIX warning-uri
    LLM_MAX_NEW_TOKENS = 512
    LLM_TEMPERATURE = 0.3
    LLM_DO_SAMPLE = True  # IMPORTANT: True pentru sampling parameters
    LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Memory optimization pentru 64GB RAM
    LLM_TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
    LLM_LOAD_IN_8BIT = False
    LLM_LOAD_IN_4BIT = False
    LLM_LOW_CPU_MEM_USAGE = True
    LLM_DEVICE_MAP = "auto" if torch.cuda.is_available() else None
    
    # Performance settings
    LLM_MAX_LENGTH = 8192
    LLM_TRUST_REMOTE_CODE = True
    LLM_USE_FAST_TOKENIZER = True
    LLM_ATTN_IMPLEMENTATION = "eager"
    
    # Generation settings - doar când do_sample=True
    LLM_TOP_P = 0.9
    LLM_TOP_K = 50
    LLM_REPETITION_PENALTY = 1.05
    LLM_PAD_TOKEN_ID = None
    
    # Web Search cu Tavily
    USE_WEB_SEARCH = True
    TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY')
    WEB_SEARCH_MAX_RESULTS = 3
    
    # Logging
    LOG_LEVEL = logging.INFO
    DEBUG_LLM = os.environ.get('DEBUG_LLM', 'false').lower() == 'true'
    DEBUG_WEB_SEARCH = os.environ.get('DEBUG_WEB_SEARCH', 'false').lower() == 'true'
    
    # Optimization flags
    LLM_USE_CACHE = True
    LLM_CLEAN_UP_TOKENIZATION_SPACES = True
    
    # Prompt settings pentru Llama-3.2-3B
    LLAMA_USE_SYSTEM_PROMPT = True
    LLAMA_PROMPT_FORMAT = "llama3"
    
    # Performance expectations
    EXPECTED_LOAD_TIME = 90
    EXPECTED_GENERATION_TIME = 15