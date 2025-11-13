from absl.testing import absltest
from einops import rearrange
from keras import ops
from keras.src import testing
from keras_hub import models
from typing_extensions import override


class ScoreLogits(testing.TestCase):
    hf_model: str
    prompt: str
    llm: models.CausalLM

    @override
    def setUp(self):
        super().setUp()
        self.hf_model = "hf://meta-llama/Llama-3.2-1B-Instruct"
        self.llm = models.CausalLM.from_preset(self.hf_model)
        self.prompt = "the quick brown fox"

    def test_score_method(self):
        response = self.llm.generate(self.prompt)
        outputs = self.llm.preprocessor.generate_preprocess(response)
        tokens = rearrange(outputs["token_ids"], "max_length -> () max_length")
        scores = self.llm.score(tokens)
        self.assertTupleEqual((1, 1024, 128256), ops.shape(scores))


if __name__ == "__main__":
    absltest.main()
