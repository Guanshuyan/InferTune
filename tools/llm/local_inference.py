"""本地 LLM 推理接口及批量限流辅助工具。"""

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Union

import requests


def chat_completion(
    messages,
    *,
    model: str | None = None,
    url: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict:
    """发起单次聊天请求，并返回网关的原始响应。"""
    model_name = model or os.getenv("LLM_MODEL", "deepseek-v3.2-exp")
    request_url = url or "http://ai-llm-gateway.amap.com/open_api/v1/chat"
    return _one_call(messages, model=model_name, url=request_url, extra_body=extra_body)


def chat_text(
    messages,
    *,
    model: str | None = None,
    url: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """发起单次聊天请求，并直接返回文本内容。"""
    response = chat_completion(messages, model=model, url=url, extra_body=extra_body)
    return response["choices"][0]["message"]["content"]


def read_json(path: Union[str, Path], *, encoding: str = "utf-8") -> Any:
    """读取 JSON 文件并返回解析后的对象。"""
    path = Path(path)
    with path.open("r", encoding=encoding) as file:
        return json.load(file)


def write_json(obj: Any, path: Union[str, Path], *, encoding: str = "utf-8", indent: int = 2) -> None:
    """将对象写入 JSON 文件。"""
    path = Path(path)
    with path.open("w", encoding=encoding) as file:
        json.dump(obj, file, ensure_ascii=False, indent=indent)


def _one_call(
    messages,
    model: str = "deepseek-v3.2-exp",
    url: str = "http://ai-llm-gateway.amap.com/open_api/v1/chat",
    extra_body: dict[str, Any] | None = None,
):
    """直接调用内部网关接口，供单次和批量请求复用。"""
    # 历史上这里接过不同网关，暂时保留当前默认地址。
    headers = {
        "Authorization": "Bearer ELbaRWLsyMPNZ2n61GN2f9Zm",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.8,
    }
    if extra_body:
        data.update(extra_body)

    # 统一在这里发请求，便于上层复用 tool-calling、批量推理等能力。
    resp = requests.post(url, headers=headers, json=data, timeout=120)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {
            "_non_json": True,
            "status_code": resp.status_code,
            "text": resp.text,
        }


class _SlidingWindowRateLimiter:
    """线程安全的 QPS/QPM 滑动窗口限流器。"""

    def __init__(self, qps: int = 100, qpm: int = 600):
        """初始化秒级与分钟级两个窗口。"""
        self.qps = int(qps)
        self.qpm = int(qpm)
        self._window_sec = deque()  # 秒级窗口 [(ts, count), ...]
        self._window_min = deque()  # 分钟级窗口 [(ts, count), ...]
        self._sum_sec = 0
        self._sum_min = 0
        self._lock = threading.Lock()

    def acquire(self):
        """阻塞直到当前请求通过速率限制。"""
        while True:
            now = time.monotonic()
            with self._lock:
                # 清理秒级窗口（1 秒）。
                cutoff_sec = now - 1.0
                while self._window_sec and self._window_sec[0][0] <= cutoff_sec:
                    _, cnt = self._window_sec.popleft()
                    self._sum_sec -= cnt

                # 清理分钟级窗口（60 秒）。
                cutoff_min = now - 60.0
                while self._window_min and self._window_min[0][0] <= cutoff_min:
                    _, cnt = self._window_min.popleft()
                    self._sum_min -= cnt

                if self._sum_sec < self.qps and self._sum_min < self.qpm:
                    self._window_sec.append((now, 1))
                    self._window_min.append((now, 1))
                    self._sum_sec += 1
                    self._sum_min += 1
                    return

                wait_sec = 0.01
                if self._sum_sec >= self.qps and self._window_sec:
                    wait_sec = max(wait_sec, (self._window_sec[0][0] + 1.0) - now)
                if self._sum_min >= self.qpm and self._window_min:
                    wait_sec = max(wait_sec, (self._window_min[0][0] + 60.0) - now)

            time.sleep(max(wait_sec, 0.001))


class _SlidingWindowTokenRateLimiter:
    """线程安全的 TPM 滑动窗口限流器。"""

    def __init__(self, tpm: int):
        """初始化 token 预算窗口。"""
        if tpm <= 0:
            raise ValueError("tpm 必须为正整数")
        self.tpm = int(tpm)
        self._window = deque()  # [(ts, tokens), ...]
        self._sum_tokens = 0
        self._lock = threading.Lock()

    def acquire(self, tokens: int):
        """阻塞直到窗口内还有足够的 token 预算。"""
        if tokens <= 0:
            return
        tokens = int(tokens)
        while True:
            now = time.monotonic()
            with self._lock:
                cutoff = now - 60.0
                while self._window and self._window[0][0] <= cutoff:
                    _, tok = self._window.popleft()
                    self._sum_tokens -= tok

                if self._sum_tokens + tokens <= self.tpm:
                    self._window.append((now, tokens))
                    self._sum_tokens += tokens
                    return

                if self._window:
                    wait_sec = (self._window[0][0] + 60.0) - now
                else:
                    wait_sec = 0.01
            time.sleep(max(wait_sec, 0.001))


def _estimate_token_budget(messages, max_tokens: int = 4096) -> int:
    """粗略估算一次请求会占用的 token 预算。"""
    try:
        serialized = json.dumps(messages, ensure_ascii=False)
    except Exception:
        serialized = str(messages)
    # 不引入 tokenizer 依赖，使用偏保守的字符数估算。
    prompt_tokens_est = max(1, len(serialized) // 2)
    return int(prompt_tokens_est + max_tokens)


def async_api_request(
    items,
    upper_bound,
    *,
    qps: int | None = None,
    qpm: int | None = None,
    tpm: int | None = None,
    model: str | None = None,
):
    """并发调用聊天接口，并附带 QPS/QPM/TPM 限流。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    res = [None] * len(items)
    errors = []

    # 优先使用函数入参，其次读取环境变量，最后退回默认值。
    env_qps = os.getenv("LLM_QPS")
    env_qpm = os.getenv("LLM_QPM")
    env_tpm = os.getenv("LLM_TPM")
    qps_v = int(qps if qps is not None else (env_qps if env_qps else 100))
    qpm_v = int(qpm if qpm is not None else (env_qpm if env_qpm else 600))
    # tpm=0 表示关闭 TPM 限流；默认给一个很大值，近似不限制 token。
    tpm_v = int(tpm if tpm is not None else (env_tpm if env_tpm else 5_000_000))
    # 模型选择同样遵循“入参优先、环境变量次之”的原则。
    model_v = model if model is not None else os.getenv("LLM_MODEL", "deepseek-v3.2-exp")

    rate_limiter = _SlidingWindowRateLimiter(qps=qps_v, qpm=qpm_v)
    token_limiter = _SlidingWindowTokenRateLimiter(tpm=tpm_v) if tpm_v > 0 else None

    def _limited_call(item):
        """在限流保护下执行一次请求。"""
        # 先控制请求数，再控制 token 预算，避免批量请求打满网关。
        rate_limiter.acquire()
        if token_limiter is not None:
            token_limiter.acquire(_estimate_token_budget(item, max_tokens=4096))
        return _one_call(item, model=model_v)

    with ThreadPoolExecutor(max_workers=min(int(upper_bound), len(items) or 1)) as executor:
        future_to_idx = {executor.submit(_limited_call, item): i for i, item in enumerate(items)}
        for future in as_completed(future_to_idx):
            index = future_to_idx[future]
            try:
                res[index] = future.result()["choices"][0]["message"]["content"]
            except Exception as exc:
                errors.append((index, repr(exc)))
                res[index] = {"_error": repr(exc)}
    return res, errors


def retry_errors_recursive(
    original_items: list,
    res: list,
    errors: list,
    *,
    max_retries: int = 3,
    retry_budget: int = 10,
    qps: int | None = 20,
    qpm: int | None = 200,
    tpm: int | None = None,
    model: str | None = None,
    _current_depth: int = 0,
):
    """递归重试失败请求，并把恢复成功的结果原地回填。"""
    if not errors:
        return res, [], {"cur_retries": 0, "total_retried": 0, "recovered": 0, "still_failed": 0}

    if _current_depth >= max_retries:
        print(f"[Retry] 已达最大重试次数 {max_retries}，剩余 {len(errors)} 个错误未恢复")
        return res, errors, {"cur_retries": _current_depth, "total_retried": 0, "recovered": 0, "still_failed": len(errors)}

    # 提取失败项索引，并缩小重试批次，避免再次雪崩。
    failed_indices = [idx for idx, _ in errors]
    print(len(failed_indices))
    failed_items = [original_items[idx] for idx in failed_indices]

    print(f"[Retry] 第 {_current_depth + 1}/{max_retries} 轮重试，共 {len(failed_items)} 个错误项，budget={retry_budget}")

    # 使用更保守的并发预算重跑失败子集。
    retry_res, retry_errors = async_api_request(
        failed_items,
        retry_budget,
        qps=qps,
        qpm=qpm,
        tpm=tpm,
        model=model,
    )

    # 统计本轮恢复情况，并把成功结果回填到原始位置。
    recovered_count = 0
    new_errors = []
    for retry_index, (original_idx, retry_result) in enumerate(zip(failed_indices, retry_res)):
        if retry_result is not None and not (isinstance(retry_result, dict) and retry_result.get("_error")):
            res[original_idx] = retry_result
            recovered_count += 1
        else:
            error_info = next((err for idx, err in retry_errors if idx == retry_index), repr(retry_result))
            new_errors.append((original_idx, error_info))

    print(f"[Retry] 第 {_current_depth + 1} 轮完成：恢复 {recovered_count} 个，剩余 {len(new_errors)} 个错误")

    if new_errors:
        _, remaining_errors, sub_stats = retry_errors_recursive(
            original_items,
            res,
            new_errors,
            max_retries=max_retries,
            retry_budget=retry_budget,
            qps=qps,
            qpm=qpm,
            tpm=tpm,
            model=model,
            _current_depth=_current_depth + 1,
        )

        total_recovered = recovered_count + sub_stats["recovered"]
        return res, remaining_errors, {
            "cur_retries": max(_current_depth + 1, sub_stats.get("cur_retries", 0)),
            "total_retried": len(failed_items) + sub_stats["total_retried"],
            "recovered": total_recovered,
            "still_failed": len(remaining_errors),
        }

    return res, [], {
        "cur_retries": _current_depth + 1,
        "total_retried": len(failed_items),
        "recovered": recovered_count,
        "still_failed": 0,
    }


def main():
    """保留的批量推理脚本入口，主要用于历史压测场景。"""
    path = "/Users/carlo/dev/verify/native1.0Optimal/functioncall_super/pre_inference/pre_inference_optimal_0107_1.json"
    prompts = read_json(path)
    print(f"输入的长度为{len(prompts)}")

    t0 = time.perf_counter()
    res, errors = async_api_request(prompts, 50)
    dt = time.perf_counter() - t0
    print(f"耗时 {dt:.2f} 秒")
    print(len(errors))
    print(errors)

    output_path = "/Users/carlo/dev/verify/native1.0Optimal/functioncall_super/post_inference"
    write_json(res, os.path.join(output_path, "inference_results_260108.json"))


if __name__ == "__main__":
    main()
