"""ProTeGi 提示优化使用示例。

演示如何使用 ProTeGi 模块自动优化一个情感分类 prompt。
默认使用内部网关（GatewayClient），也可切换为 OpenAIClient。

使用方式:
    python examples/prompt_optimize_demo.py
"""

from infertune.llm import GatewayClient
from infertune.evaluator import AccuracyEvaluator
from infertune.prompt_optimizer.optimizer import PromptOptimizer
from infertune.config import ProTeGiConfig


def main():
    # 1. 初始化 LLM（默认使用内部网关）
    llm = GatewayClient(
        model="deepseek-v3.2-exp",
        api_url="<LLM_API_URL>",
        api_key="<LLM_API_KEY>",
        temperature=0.8,
    )
    # 如需使用 OpenAI 兼容 API，替换为:
    # from infertune.llm import OpenAIClient
    # llm = OpenAIClient(model="gpt-4o", api_key="your-key")

    # 2. 准备训练和评估数据
    train_data = [
        {"input": "这部电影太棒了，演员演技在线", "label": "positive"},
        {"input": "剧情拖沓，浪费时间", "label": "negative"},
        {"input": "画面精美，配乐动人", "label": "positive"},
        {"input": "烂片一部，不推荐", "label": "negative"},
        {"input": "故事感人，值得一看", "label": "positive"},
        {"input": "特效五毛，剧本稀烂", "label": "negative"},
        {"input": "节奏紧凑，全程无尿点", "label": "positive"},
        {"input": "看了开头就想走", "label": "negative"},
    ]

    eval_data = [
        {"input": "非常好看的一部作品", "label": "positive"},
        {"input": "失望透顶", "label": "negative"},
        {"input": "超出预期，强烈推荐", "label": "positive"},
        {"input": "无聊至极", "label": "negative"},
    ]

    # 3. 配置优化参数（小规模 demo 用较小的参数）
    config = ProTeGiConfig(
        beam_width=2,
        search_depth=3,
        minibatch_size=8,
        num_gradients=2,
        num_edits_per_gradient=1,
        num_monte_carlo=1,
        selection_method="ucb",
    )

    # 4. 执行优化（会先打印调用次数估算）
    optimizer = PromptOptimizer(llm, config)
    evaluator = AccuracyEvaluator()

    initial_prompt = "判断以下文本的情感是正面还是负面，只输出 positive 或 negative"

    print(f"初始 prompt: {initial_prompt}")
    print("开始优化...\n")

    result = optimizer.optimize(
        initial_prompt,
        train_data,
        eval_data,
        evaluator,
        verbose=True,
    )

    # 5. 输出结果
    print(f"\n{'='*50}")
    print(f"初始分数: {result['initial_score']:.4f}")
    print(f"最终分数: {result['final_score']:.4f}")
    print(f"提升: {result['improvement']:+.4f}")
    print(f"\n最优 prompt:\n{result['best_prompt']}")


if __name__ == "__main__":
    main()
