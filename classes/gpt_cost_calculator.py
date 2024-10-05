import openai
import tiktoken

class GPTCostCalculator:
    def __init__(self, model_name):
        """
        クラスを初期化し、モデルごとのトークン単価を設定。
        :param model_name: 使用するモデル名（例: "gpt-4"）
        """
        self.model_name = model_name
        self.input_texts = []
        self.output_texts = []
        self.token_usage = []

        # モデルごとのトークン単価を設定
        self.model_token_costs = {
            'gpt-4o': {'input': 5 / 1000000, 'output': 15 / 1000000},
            'gpt-4o-mini': {'input': 0.15 / 1000000, 'output': 0.6 / 1000000},
        }

        # 指定されたモデルのトークナイザーを取得
        self.encoding = tiktoken.encoding_for_model(model_name)

        # モデル名に対応するトークン単価を取得
        if model_name not in self.model_token_costs:
            raise ValueError(f"Model '{model_name}' is not supported.")

        self.token_cost = self.model_token_costs[model_name]

    def add_input_text(self, text):
        """
        入力文を記録するメソッド。
        :param text: 入力された文章
        """
        self.input_texts.append(text)

    def add_output_text(self, text):
        """
        出力文を記録するメソッド。
        :param text: 出力された文章
        """
        self.output_texts.append(text)

    def count_tokens(self, text):
        """
        テキストのトークン数を計算するメソッド。
        :param text: トークン数を計算する文章
        :return: トークン数
        """
        tokens = self.encoding.encode(text)
        return len(tokens)

    def calculate_cost(self):
        """
        入出力トークン数に基づいてコストを計算するメソッド。
        2と3で記録された全ての入力文と出力文を対象にする。
        :return: 計算されたコスト
        """
        total_input_tokens = sum([self.count_tokens(text) for text in self.input_texts])
        total_output_tokens = sum([self.count_tokens(text) for text in self.output_texts])

        # 計算されたトークン数をリストに記録
        self.token_usage.append({'input_tokens': total_input_tokens, 'output_tokens': total_output_tokens})

        # コストを計算
        input_cost = total_input_tokens * self.token_cost['input']
        output_cost = total_output_tokens * self.token_cost['output']
        total_cost = input_cost + output_cost

        return total_cost

    def get_token_usage(self):
        """
        トークン使用量の履歴を返すメソッド。
        :return: トークン使用量のリスト
        """
        return self.token_usage
