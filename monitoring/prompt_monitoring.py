import wandb
from settings import settings
from typing import Optional, Dict
from contextlib import contextmanager

class PromptMonitoringManager:
    @classmethod
    def initialize(cls):
        """Khởi tạo WandB run nếu chưa tồn tại."""
        if wandb.run is None:  # Kiểm tra run toàn cục
            wandb.init(
                project=settings.WANDB_PROJECT,
                entity=settings.WANDB_ENTITY,
                config={"api_key": settings.WANDB_API_KEY},
                reinit=True  # Cho phép khởi tạo lại nếu cần
            )
        return wandb.run
    
    @classmethod
    def finish(cls):
        """Kết thúc WandB run nếu đang hoạt động."""
        if wandb.run is not None:
            wandb.finish()

    @classmethod
    @contextmanager
    def wandb_session(cls):
        """Context manager để quản lý WandB run."""
        run = cls.initialize()
        try:
            yield run
        finally:
            pass  # Có thể gọi cls.finish() nếu muốn tự động kết thúc

    @classmethod
    def log(
        cls,
        prompt: str,
        output: str,
        prompt_template: Optional[str] = None,
        prompt_template_variables: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        # Khởi tạo WandB run
        run = cls.initialize()

        # Chuẩn bị metadata
        metadata = metadata or {}
        metadata = {
            "model": settings.MODEL_TYPE,
            **metadata,
        }

        # Ghi log prompt, output và metadata
        log_data = {
            "prompt": prompt,
            "output": output,
            "prompt_template": prompt_template,
            "prompt_template_variables": prompt_template_variables,
            "metadata": metadata,
        }
        run.log(log_data)

    @classmethod
    def log_chain(cls, query: str, response: str, eval_output: str):
        run = cls.initialize()

        # Log từng bước trong chuỗi
        run.log({"user_query": query}, step=0)
        run.log({
            "twin_response_input": query,
            "twin_response_output": response
        }, step=1)
        run.log({
            "gpt3.5_eval_input": eval_output,
            "gpt3.5_eval_output": response
        }, step=2)
        run.log({
            "response": response,
            "eval_output": eval_output
        }, step=3)