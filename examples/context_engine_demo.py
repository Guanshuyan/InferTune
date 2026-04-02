"""ACE 上下文工程使用示例。

演示如何使用 ACE 模块通过 offline 和 online 两种模式优化上下文。
默认使用内部网关（GatewayClient），也可切换为 OpenAIClient。

使用方式:
    python examples/context_engine_demo.py
"""

from infertune.llm import GatewayClient
from infertune.evaluator import AccuracyEvaluator
from infertune.context_engine.engine import ContextEngine
from infertune.config import ACEConfig


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

    # 2. 准备数据
    train_data = [
        {"input": "苹果公司发布了新款 iPhone", "label": "科技"},
        {"input": "国足在世预赛中战胜对手", "label": "体育"},
        {"input": "央行宣布降息 25 个基点", "label": "财经"},
        {"input": "新型疫苗通过三期临床试验", "label": "科技"},
        {"input": "NBA 总决赛今晚开打", "label": "体育"},
        {"input": "A 股三大指数集体收涨", "label": "财经"},
    ]

    eval_data = [
        {"input": "特斯拉发布全自动驾驶更新", "label": "科技"},
        {"input": "奥运会开幕式精彩纷呈", "label": "体育"},
        {"input": "美联储维持利率不变", "label": "财经"},
    ]

    system_prompt = "请将以下新闻标题分类为：科技、体育、财经。只输出类别名称。"

    # ---- Offline 模式 ----
    print("=" * 50)
    print("Offline 模式：在训练数据上迭代优化 Playbook")
    print("=" * 50)

    config = ACEConfig(
        mode="offline",
        max_epochs=2,
        max_reflector_rounds=2,
        dedup_threshold=0.8,
    )
    engine = ContextEngine(llm, config)
    evaluator = AccuracyEvaluator()

    result = engine.optimize_offline(
        train_data,
        system_prompt=system_prompt,
        evaluator=evaluator,
        eval_data=eval_data,
        verbose=True,
    )

    print(f"\nPlaybook 大小: {result['num_bullets']} 条策略")
    print(f"Epoch 分数: {result['epoch_scores']}")
    print(f"\n生成的上下文:\n{result['context'][:500]}...")

    # ---- Online 模式 ----
    print("\n" + "=" * 50)
    print("Online 模式：逐条处理并实时更新 Playbook")
    print("=" * 50)

    online_config = ACEConfig(
        mode="online",
        max_reflector_rounds=2,
    )
    online_engine = ContextEngine(llm, online_config)

    test_inputs = [
        {"input": "华为发布新一代芯片", "label": "科技"},
        {"input": "中超联赛第 10 轮战报", "label": "体育"},
        {"input": "比特币价格突破新高", "label": "财经"},
    ]

    for item in test_inputs:
        result = online_engine.optimize_online(
            item["input"], label=item["label"], system_prompt=system_prompt,
        )
        print(f"输入: {item['input']}")
        print(f"  回答: {result['answer']} | 正确: {result['correct']} | Playbook: {result['playbook_size']} 条")

    # 保存 Playbook 供后续使用
    online_engine.playbook.save("playbook_output.json")
    print(f"\nPlaybook 已保存到 playbook_output.json")


if __name__ == "__main__":
    main()
